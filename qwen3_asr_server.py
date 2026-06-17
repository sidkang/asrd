#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=1.0",
#   "mlx-qwen3-asr[serve]>=0.3.5",
#   "uvicorn>=0.20.0",
#   "websockets>=12.0",
# ]
# ///

"""
# voxt-qwen3-asr

Single-file local OpenAI-compatible ASR server for VoxT on Apple Silicon/MLX,
powered by `mlx-qwen3-asr`.

## Run

```bash
uv run qwen3_asr_server.py
```

Defaults:

- endpoint: `http://127.0.0.1:5100/v1/audio/transcriptions`
- realtime endpoint: `ws://127.0.0.1:5100/v1/realtime`
- bind: `0.0.0.0:5100`
- API key: `password`
- final model: `mlx-community/Qwen3-ASR-1.7B-6bit`
- preview model: `mlx-community/Qwen3-ASR-0.6B-4bit`
- model cache: `./.cache`
- Hugging Face endpoint: `https://hf-mirror.com`
- idle unload: `30s`
- preview streaming context window: `30s`

## VoxT settings

```text
Provider: OpenAI Transcribe
Endpoint: http://127.0.0.1:5100/v1/audio/transcriptions
API Key: password
Model: mlx-community/Qwen3-ASR-1.7B-6bit
Chunk Pseudo Realtime Preview: On
```

Realtime WebSocket:

```text
Provider: Aliyun Qwen Realtime ASR
Endpoint: ws://127.0.0.1:5100/v1/realtime
API Key: password
Model: qwen3-asr-flash-realtime
Audio: PCM16 mono 16k
```

For LAN use, replace `127.0.0.1` with the server machine IP.

## Notes

- VoxT preview uploads named `voxt-openai-preview-*.wav` use the preview model.
- VoxT HTTP previews are handled as stateful streaming/delta ASR internally.
- Final uploads use the final model.
- ASR runs in a worker process, loaded lazily and killed after idle timeout.
- Requests are serialized to avoid MLX thread/stream issues.
- Busy preview requests are skipped with an empty result to avoid backlog.
- Logs include speed and RTF, e.g. `speed=14.11x rtf=0.071`.

Useful commands:

```bash
uv run qwen3_asr_server.py --readme
uv run qwen3_asr_server.py --unload-after-sec 0
uv run qwen3_asr_server.py --preview-max-new-tokens 64
HF_ENDPOINT=https://huggingface.co uv run qwen3_asr_server.py
```
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import json
import multiprocessing as mp
import os
import queue
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from huggingface_hub import snapshot_download


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5100
DEFAULT_API_KEY = "password"
DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-6bit"
DEFAULT_PREVIEW_MODEL = "mlx-community/Qwen3-ASR-0.6B-4bit"
DEFAULT_DTYPE = "float16"
DEFAULT_MAX_FILE_SIZE_MB = 512
DEFAULT_PREVIEW_MAX_NEW_TOKENS = 96
DEFAULT_PREVIEW_MAX_AUDIO_SECONDS = 30.0
DEFAULT_UNLOAD_AFTER_SEC = 30.0
WORKER_PIDFILE = Path("/tmp/voxt-qwen3-asr-worker.pid")

LANGUAGE_ALIASES = {
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "zh-hk": "Chinese",
    "cmn-hans-cn": "Chinese",
    "cmn-hant-tw": "Chinese",
    "yue-hant-hk": "Chinese",
    "en-us": "English",
    "en-gb": "English",
    "ja-jp": "Japanese",
    "ko-kr": "Korean",
    "de-de": "German",
    "fr-fr": "French",
    "es-es": "Spanish",
    "es-419": "Spanish",
    "it-it": "Italian",
    "pt-br": "Portuguese",
    "pt-pt": "Portuguese",
    "nl-nl": "Dutch",
    "ru-ru": "Russian",
    "ar-sa": "Arabic",
    "ar-eg": "Arabic",
    "hi-in": "Hindi",
    "tr-tr": "Turkish",
}

DTYPE_NAMES = {"float16", "float32", "bfloat16"}


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    api_key: str
    model: str
    model_path: str
    preview_model: str
    preview_model_path: str
    dtype: str
    max_file_size_mb: int
    unload_after_sec: float
    preview_max_audio_seconds: float
    preview_max_new_tokens: int | None
    final_max_new_tokens: int | None


@dataclass
class State:
    inference_lock: asyncio.Lock | None = None
    ws_lock: asyncio.Lock | None = None
    worker_process: mp.Process | None = None
    request_queue: mp.Queue | None = None
    response_queue: mp.Queue | None = None
    ws_session: Any | None = None
    ws_executor: concurrent.futures.ThreadPoolExecutor | None = None
    active_ws_count: int = 0
    unload_task: asyncio.Task | None = None
    last_used_at: float = 0.0


@dataclass
class PreviewStreamSession:
    stream_state: Any
    last_sample_count: int
    first_head: np.ndarray
    last_tail: np.ndarray
    last_text: str
    language: str | None
    prompt: str
    max_new_tokens: int | None
    last_used_at: float
    reset_count: int = 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.readme:
        print(__doc__.strip())
        return 0
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg was not found; non-WAV audio may fail.", file=sys.stderr)

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    configure_hf_cache(cache_dir)

    print("Preparing model cache...", flush=True)
    model_path = prepare_model(args.model)
    preview_model_path = prepare_model(args.preview_model)
    print_startup_summary(args, model_path, preview_model_path)

    import uvicorn

    app = create_app(
        Config(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            model=args.model,
            model_path=model_path,
            preview_model=args.preview_model,
            preview_model_path=preview_model_path,
            dtype=args.dtype,
            unload_after_sec=args.unload_after_sec,
            preview_max_audio_seconds=args.preview_max_audio_seconds,
            max_file_size_mb=args.max_file_size_mb,
            preview_max_new_tokens=args.preview_max_new_tokens,
            final_max_new_tokens=args.final_max_new_tokens,
        )
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local OpenAI-compatible Qwen3-ASR server for VoxT.")
    parser.add_argument("--readme", action="store_true", help="Print the embedded README and exit.")
    parser.add_argument("--host", default=os.environ.get("ASR_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ASR_PORT", DEFAULT_PORT)))
    parser.add_argument("--api-key", default=os.environ.get("MLX_ASR_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--model", default=os.environ.get("ASR_MODEL", DEFAULT_MODEL), help="Final transcription model.")
    parser.add_argument(
        "--preview-model",
        default=os.environ.get("ASR_PREVIEW_MODEL", DEFAULT_PREVIEW_MODEL),
        help="Model used for VoxT Chunk Pseudo Realtime Preview uploads.",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPE_NAMES), default=os.environ.get("ASR_DTYPE", DEFAULT_DTYPE))
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("ASR_MODEL_CACHE_DIR", str(Path.cwd() / ".cache")),
        help="Model cache root. Defaults to ./.cache in the current directory.",
    )
    parser.add_argument("--max-file-size-mb", type=int, default=int(os.environ.get("ASR_MAX_FILE_SIZE_MB", DEFAULT_MAX_FILE_SIZE_MB)))
    parser.add_argument(
        "--unload-after-sec",
        type=float,
        default=float(os.environ.get("ASR_UNLOAD_AFTER_SEC", DEFAULT_UNLOAD_AFTER_SEC)),
        help="Unload models after this many idle seconds. Use 0 to keep models loaded.",
    )
    parser.add_argument(
        "--preview-max-new-tokens",
        type=optional_positive_int,
        default=optional_positive_int(os.environ.get("ASR_PREVIEW_MAX_NEW_TOKENS", str(DEFAULT_PREVIEW_MAX_NEW_TOKENS))),
        help="Cap generated tokens for VoxT pseudo-realtime preview uploads. HTTP streaming preview uses a conservative 16-32 token per-chunk cap derived from this value. Use 0/none to disable sync final-style caps.",
    )
    parser.add_argument(
        "--preview-max-audio-seconds",
        type=float,
        default=float(os.environ.get("ASR_PREVIEW_MAX_AUDIO_SECONDS", DEFAULT_PREVIEW_MAX_AUDIO_SECONDS)),
        help="Maximum audio context retained by HTTP preview streaming. Use 0 to disable the context cap.",
    )
    parser.add_argument(
        "--final-max-new-tokens",
        type=optional_positive_int,
        default=optional_positive_int(os.environ.get("ASR_FINAL_MAX_NEW_TOKENS", "0")),
        help="Cap generated tokens for final uploads. Default uses mlx-qwen3-asr auto sizing.",
    )
    parser.add_argument("--log-level", default=os.environ.get("ASR_LOG_LEVEL", "info"))
    return parser.parse_args(argv)


def optional_positive_int(value: str | int | None) -> int | None:
    if value is None or isinstance(value, int) and value <= 0:
        return None
    text = str(value).strip().lower()
    if text in {"", "0", "none", "auto", "false"}:
        return None
    parsed = int(text)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer, 0, or none")
    return parsed


def configure_hf_cache(cache_dir: Path) -> None:
    hf_home = Path(os.environ.get("HF_HOME", cache_dir / "huggingface")).expanduser().resolve()
    hub_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home / "hub")).expanduser().resolve()
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def prepare_model(model: str) -> str:
    local = Path(model).expanduser()
    if local.exists() and (local / "config.json").exists():
        ensure_quantization_config(local, model)
        return str(local.resolve())

    allow_patterns = ["*.json", "*.safetensors", "*.txt", "*.model"]
    try:
        path = Path(snapshot_download(repo_id=model, allow_patterns=allow_patterns, local_files_only=True))
        ensure_quantization_config(path, model)
        return str(path)
    except Exception:
        print(f"Model {model} not fully available in local cache; downloading...", flush=True)

    path = Path(snapshot_download(repo_id=model, allow_patterns=allow_patterns))
    ensure_quantization_config(path, model)
    return str(path)


def ensure_quantization_config(model_path: Path, model_name: str) -> None:
    bits = quantization_bits_from_name(model_name)
    if bits is None:
        return
    qconf = model_path / "quantization_config.json"
    if qconf.exists():
        current = read_json(qconf)
        if current.get("bits") != bits:
            print(f"WARNING: {qconf} declares bits={current.get('bits')}, but model name suggests bits={bits}; trusting existing config.", file=sys.stderr)
        return
    qconf.write_text(json.dumps({"bits": bits, "group_size": 64}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {qconf}", file=sys.stderr)


def quantization_bits_from_name(name: str) -> int | None:
    match = re.search(r"(?:^|[-_/])(4|6|8)bit(?:$|[-_/])", name.lower())
    return int(match.group(1)) if match else None


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def create_app(config: Config) -> FastAPI:
    state = State(
        inference_lock=asyncio.Lock(),
        ws_lock=asyncio.Lock(),
        ws_executor=concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="voxt-asr-ws"),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if config.unload_after_sec > 0:
            state.unload_task = asyncio.create_task(unload_idle_ws_session_loop(state, config))
        yield
        if state.unload_task is not None:
            state.unload_task.cancel()
            try:
                await state.unload_task
            except asyncio.CancelledError:
                pass
        await unload_ws_session(state, config, reason="shutdown", force=True)
        if state.ws_executor is not None:
            state.ws_executor.shutdown(wait=True, cancel_futures=True)
        stop_worker(state, reason="shutdown")

    app = FastAPI(title="voxt-qwen3-asr", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        worker_alive = worker_process_alive(state)
        idle_for = time.monotonic() - state.last_used_at if state.last_used_at else None
        idle_remaining = None
        if (worker_alive or state.ws_session is not None) and idle_for is not None and config.unload_after_sec > 0:
            idle_remaining = max(0.0, config.unload_after_sec - idle_for)
        return {
            "status": "ok",
            "model": config.model,
            "preview_model": config.preview_model,
            "worker_alive": worker_alive,
            "models_loaded": worker_alive,
            "ws_model_loaded": state.ws_session is not None,
            "active_ws_count": state.active_ws_count,
            "busy": state.inference_lock.locked() if state.inference_lock else False,
            "last_used_at": state.last_used_at or None,
            "idle_for_sec": round(idle_for, 3) if idle_for is not None else None,
            "idle_remaining_sec": round(idle_remaining, 3) if idle_remaining is not None else None,
            "unload_after_sec": config.unload_after_sec,
        }

    @app.get("/v1/models")
    async def models(request: Request) -> dict:
        check_auth(request, config.api_key)
        return {"object": "list", "data": [model_info(config.model), model_info(config.preview_model)]}

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket):
        if not websocket_auth_ok(websocket, config.api_key):
            await websocket.close(code=1008, reason="Invalid API key")
            return
        if state.ws_lock is None or state.ws_executor is None:
            await websocket.close(code=1013, reason="ASR runtime is not ready")
            return

        await websocket.accept()
        state.active_ws_count += 1
        client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
        print(f"Realtime WS connected client={client} active={state.active_ws_count}", flush=True)
        stream_state = None
        last_text = ""
        language = None
        prompt = ""
        max_new_tokens = config.final_max_new_tokens
        session_configured = False
        fun_task_id = ""
        audio_buffer: list[np.ndarray] = []
        buffered_samples = 0
        feed_threshold_samples = 32000  # 2s at 16 kHz; avoid per-100ms decode backlog.
        try:
            while True:
                received = await websocket.receive()
                if "bytes" in received and received["bytes"] is not None:
                    if not session_configured or stream_state is None:
                        await websocket_send_error(websocket, "session_not_configured", "Send run-task/session.update before audio")
                        continue
                    audio = pcm16_bytes_to_float32(received["bytes"])
                    audio_buffer.append(audio)
                    buffered_samples += int(audio.size)
                    if buffered_samples >= feed_threshold_samples:
                        chunk = np.concatenate(audio_buffer) if len(audio_buffer) > 1 else audio_buffer[0]
                        print(f"Realtime WS feed client={client} samples={chunk.size} seconds={chunk.size / 16000:.2f}", flush=True)
                        audio_buffer = []
                        buffered_samples = 0
                        stream_state = await ws_feed_audio(state, config, chunk, stream_state)
                        text = stream_state.text or ""
                        if text and text != last_text:
                            last_text = text
                            await send_aliyun_fun_result(websocket, fun_task_id, text, sentence_end=False)
                    continue
                message = received.get("text")
                if message is None:
                    continue
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    await websocket_send_error(websocket, "invalid_json", "Message must be JSON")
                    continue

                header = event.get("header") if isinstance(event.get("header"), dict) else {}
                action = str(header.get("action") or "")
                event_type = event.get("type")
                if event_type != "input_audio_buffer.append":
                    print(
                        f"Realtime WS event client={client} type={event_type!r} action={action!r} "
                        f"keys={list(event.keys())} raw={message[:500]!r}",
                        flush=True,
                    )
                if action == "run-task":
                    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
                    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
                    fun_task_id = str(header.get("task_id") or new_event_id())
                    requested_model = payload.get("model")
                    selected_model = select_realtime_model(str(requested_model or ""), config)
                    hints = parameters.get("language_hints") if isinstance(parameters.get("language_hints"), list) else []
                    language = normalize_language(str(hints[0])) if hints else None
                    print(
                        f"Realtime WS aliyun fun run-task client={client} task_id={fun_task_id} "
                        f"request_model={requested_model!r} selected_model={selected_model!r} language={language!r}",
                        flush=True,
                    )
                    stream_state = await ws_init_stream(state, config, context=prompt, language=language, max_new_tokens=max_new_tokens)
                    session_configured = True
                    await send_aliyun_fun_event(websocket, fun_task_id, "task-started")
                elif action == "finish-task":
                    if stream_state is None:
                        stream_state = await ws_init_stream(state, config, context=prompt, language=language, max_new_tokens=max_new_tokens)
                    if buffered_samples > 0:
                        chunk = np.concatenate(audio_buffer) if len(audio_buffer) > 1 else audio_buffer[0]
                        print(f"Realtime WS feed-final client={client} samples={chunk.size} seconds={chunk.size / 16000:.2f}", flush=True)
                        audio_buffer = []
                        buffered_samples = 0
                        stream_state = await ws_feed_audio(state, config, chunk, stream_state)
                    stream_state = await ws_finish_stream(state, config, stream_state)
                    text = stream_state.text or last_text
                    if text:
                        await send_aliyun_fun_result(websocket, fun_task_id, text, sentence_end=True)
                    await send_aliyun_fun_event(websocket, fun_task_id, "task-finished")
                    await websocket.close(code=1000)
                    return
                elif event_type == "session.update":
                    session = event.get("session") if isinstance(event.get("session"), dict) else {}
                    language = normalize_language(session.get("language") or event.get("language"))
                    prompt = str(session.get("prompt") or session.get("context") or event.get("prompt") or "")
                    requested_model = session.get("model") or event.get("model")
                    selected_model = select_realtime_model(str(requested_model or ""), config)
                    print(
                        f"Realtime WS session.update client={client} request_model={requested_model!r} "
                        f"selected_model={selected_model!r} language={language!r} prompt_chars={len(prompt)}",
                        flush=True,
                    )
                    stream_state = await ws_init_stream(state, config, context=prompt, language=language, max_new_tokens=max_new_tokens)
                    session_configured = True
                    await websocket.send_json(
                        {
                            "type": "session.updated",
                            "event_id": new_event_id(),
                            "session": {
                                "model": selected_model,
                                "input_audio_format": "pcm16",
                                "sample_rate": 16000,
                            },
                        }
                    )
                elif event_type == "input_audio_buffer.append":
                    if not session_configured or stream_state is None:
                        await websocket_send_error(websocket, "session_not_configured", "Send session.update before audio")
                        continue
                    audio_b64 = str(event.get("audio") or event.get("delta") or "")
                    try:
                        audio = decode_pcm16_base64(audio_b64)
                    except ValueError as exc:
                        await websocket_send_error(websocket, "invalid_audio", str(exc))
                        continue
                    audio_buffer.append(audio)
                    buffered_samples += int(audio.size)
                    if buffered_samples >= feed_threshold_samples:
                        chunk = np.concatenate(audio_buffer) if len(audio_buffer) > 1 else audio_buffer[0]
                        print(f"Realtime WS feed client={client} samples={chunk.size} seconds={chunk.size / 16000:.2f}", flush=True)
                        audio_buffer = []
                        buffered_samples = 0
                        stream_state = await ws_feed_audio(state, config, chunk, stream_state)
                        text = stream_state.text or ""
                        if text and text != last_text:
                            delta = text[len(last_text) :] if text.startswith(last_text) else text
                            last_text = text
                            await websocket.send_json(
                                {
                                    "type": "conversation.item.input_audio_transcription.text",
                                    "event_id": new_event_id(),
                                    "text": text,
                                    "delta": delta,
                                }
                            )
                elif event_type == "session.finish":
                    if stream_state is None:
                        stream_state = await ws_init_stream(state, config, context=prompt, language=language, max_new_tokens=max_new_tokens)
                    if buffered_samples > 0:
                        chunk = np.concatenate(audio_buffer) if len(audio_buffer) > 1 else audio_buffer[0]
                        print(f"Realtime WS feed-final client={client} samples={chunk.size} seconds={chunk.size / 16000:.2f}", flush=True)
                        audio_buffer = []
                        buffered_samples = 0
                        stream_state = await ws_feed_audio(state, config, chunk, stream_state)
                    stream_state = await ws_finish_stream(state, config, stream_state)
                    text = stream_state.text or ""
                    if text and text != last_text:
                        await websocket.send_json(
                            {
                                "type": "conversation.item.input_audio_transcription.text",
                                "event_id": new_event_id(),
                                "text": text,
                                "delta": text[len(last_text) :] if text.startswith(last_text) else text,
                            }
                        )
                    await websocket.send_json(
                        {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "event_id": new_event_id(),
                            "text": text,
                            "transcript": text,
                        }
                    )
                    await websocket.send_json({"type": "session.finished", "event_id": new_event_id()})
                    await websocket.close(code=1000)
                    return
                else:
                    await websocket_send_error(websocket, "unsupported_event", f"Unsupported event type: {event_type}")
        except WebSocketDisconnect:
            print(f"Realtime WS disconnected client={client}", flush=True)
        finally:
            state.active_ws_count = max(0, state.active_ws_count - 1)
            state.last_used_at = time.monotonic()
            print(f"Realtime WS closed client={client} active={state.active_ws_count}", flush=True)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: Optional[str] = Form(None),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: Optional[str] = Form("json"),
    ):
        check_auth(request, config.api_key)
        if state.inference_lock is None:
            raise HTTPException(status_code=503, detail="ASR runtime is not ready")

        is_preview = should_use_preview_model(model, file.filename, config)
        if is_preview and state.inference_lock.locked():
            print(f"Skipping preview while ASR worker is busy filename={file.filename!r}", flush=True)
            return JSONResponse({"text": ""})

        suffix = Path(file.filename or "upload.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="voxt_asr_")
        tmp_path = tmp.name
        try:
            size = await write_upload_to_temp(file, tmp, config.max_file_size_mb)
            fmt = (response_format or "json").strip().lower()
            if fmt not in {"json", "text", "verbose_json"}:
                raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")

            lang = normalize_language(language)
            selected_model = config.preview_model if is_preview else config.model
            max_new_tokens = config.preview_max_new_tokens if is_preview else config.final_max_new_tokens
            audio_duration = audio_duration_seconds(tmp_path)
            preview_window_log = ""
            if audio_duration and is_preview:
                preview_window_log = f"audio={audio_duration:.2f}s streaming_context={preview_context_label(config)} "
            print(
                "Transcription request "
                f"kind={'preview' if is_preview else 'final'} filename={file.filename!r} bytes={size} "
                f"{preview_window_log}"
                f"request_model={model!r} selected_model={selected_model!r} "
                f"language={language!r}->{lang!r} prompt_chars={len(prompt or '')} format={fmt!r}",
                flush=True,
            )

            started = time.perf_counter()
            async with state.inference_lock:
                state.last_used_at = time.monotonic()
                result = await transcribe_in_worker(
                    state,
                    config,
                    audio_path=tmp_path,
                    preview=is_preview,
                    audio_duration=audio_duration,
                    language=lang,
                    prompt=prompt or "",
                    max_new_tokens=max_new_tokens,
                    timeout_sec=transcription_timeout_seconds(audio_duration),
                )
                state.last_used_at = time.monotonic()
            elapsed = time.perf_counter() - started
            text = (result.get("text") or "").strip()
            print_efficiency(audio_duration, elapsed, len(text), is_preview=is_preview, max_new_tokens=max_new_tokens, metadata=result)

            if fmt == "text":
                return PlainTextResponse(text)
            if fmt == "verbose_json":
                return JSONResponse({"text": text, "task": "transcribe", "language": lang, "duration": audio_duration, "segments": []})
            return JSONResponse({"text": text})
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": type(exc).__name__}})
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    return app


async def ws_init_stream(state: State, config: Config, *, context: str, language: str | None, max_new_tokens: int | None):
    if state.ws_lock is None or state.ws_executor is None:
        raise RuntimeError("Realtime ASR runtime is not ready")
    loop = asyncio.get_running_loop()
    async with state.ws_lock:
        result = await loop.run_in_executor(
            state.ws_executor,
            ws_init_stream_sync,
            state,
            config,
            context,
            language,
            max_new_tokens,
        )
        state.last_used_at = time.monotonic()
        return result


async def ws_feed_audio(state: State, config: Config, audio: np.ndarray, stream_state: Any):
    if state.ws_lock is None or state.ws_executor is None:
        raise RuntimeError("Realtime ASR runtime is not ready")
    loop = asyncio.get_running_loop()
    async with state.ws_lock:
        result = await loop.run_in_executor(state.ws_executor, ws_feed_audio_sync, state, config, audio, stream_state)
        state.last_used_at = time.monotonic()
        return result


async def ws_finish_stream(state: State, config: Config, stream_state: Any):
    if state.ws_lock is None or state.ws_executor is None:
        raise RuntimeError("Realtime ASR runtime is not ready")
    loop = asyncio.get_running_loop()
    async with state.ws_lock:
        result = await loop.run_in_executor(state.ws_executor, ws_finish_stream_sync, state, config, stream_state)
        state.last_used_at = time.monotonic()
        return result


def ws_init_stream_sync(state: State, config: Config, context: str, language: str | None, max_new_tokens: int | None):
    return get_ws_session_sync(state, config).init_streaming(
        context=context,
        language=language,
        sample_rate=16000,
        max_new_tokens=max_new_tokens,
    )


def ws_feed_audio_sync(state: State, config: Config, audio: np.ndarray, stream_state: Any):
    return get_ws_session_sync(state, config).feed_audio(audio, stream_state)


def ws_finish_stream_sync(state: State, config: Config, stream_state: Any):
    return get_ws_session_sync(state, config).finish_streaming(stream_state)


def get_ws_session_sync(state: State, config: Config):
    if state.ws_session is not None:
        return state.ws_session
    from mlx_qwen3_asr.session import Session

    dtype = {"float16": mx.float16, "float32": mx.float32, "bfloat16": mx.bfloat16}.get(config.dtype, mx.float16)
    load_started = time.perf_counter()
    print(f"Loading realtime ASR model from {config.model_path} dtype={config.dtype}", flush=True)
    state.ws_session = Session(config.model_path, dtype=dtype)
    print(f"Realtime ASR model loaded elapsed={time.perf_counter() - load_started:.2f}s", flush=True)
    return state.ws_session


async def unload_idle_ws_session_loop(state: State, config: Config) -> None:
    while True:
        await asyncio.sleep(min(max(config.unload_after_sec / 2, 1.0), 10.0))
        if state.ws_session is None or state.active_ws_count > 0 or state.last_used_at <= 0:
            continue
        idle_for = time.monotonic() - state.last_used_at
        if idle_for >= config.unload_after_sec:
            await unload_ws_session(state, config, reason=f"{config.unload_after_sec}s idle")


async def unload_ws_session(state: State, config: Config, *, reason: str, force: bool = False) -> None:
    if state.ws_lock is None or state.ws_executor is None:
        return
    async with state.ws_lock:
        if state.ws_session is None or state.active_ws_count > 0:
            return
        if not force and config.unload_after_sec > 0 and state.last_used_at > 0:
            idle_for = time.monotonic() - state.last_used_at
            if idle_for < config.unload_after_sec:
                return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(state.ws_executor, unload_ws_session_sync, state, reason)


def unload_ws_session_sync(state: State, reason: str) -> None:
    if state.ws_session is None:
        return
    print(f"Unloading realtime ASR model after {reason}", flush=True)
    state.ws_session = None
    clear_mlx_cache()


def clear_mlx_cache() -> None:
    for clear in (mx.clear_cache, getattr(mx, "metal", None) and mx.metal.clear_cache):
        if clear:
            try:
                clear()
            except Exception:
                pass


def websocket_auth_ok(websocket: WebSocket, api_key: str) -> bool:
    if websocket.headers.get("authorization", "") == f"Bearer {api_key}":
        return True
    if websocket.headers.get("x-api-key", "") == api_key:
        return True
    return websocket.query_params.get("api_key") == api_key


def select_realtime_model(request_model: str, config: Config) -> str:
    requested = request_model.strip()
    if requested == config.model:
        return config.model
    if requested and requested == config.preview_model:
        print("Realtime preview model requested; using final model because streaming uses the final session", flush=True)
    elif requested and requested != config.model:
        print(f"Realtime model {requested!r} requested; using final model {config.model!r}", flush=True)
    return config.model


def decode_pcm16_base64(audio_base64: str) -> np.ndarray:
    if not audio_base64:
        return np.array([], dtype=np.float32)
    try:
        raw = base64.b64decode(audio_base64, validate=True)
    except Exception as exc:
        raise ValueError("audio must be base64-encoded PCM16") from exc
    return pcm16_bytes_to_float32(raw)


def pcm16_bytes_to_float32(raw: bytes) -> np.ndarray:
    if len(raw) % 2:
        raw = raw[:-1]
    if not raw:
        return np.array([], dtype=np.float32)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


async def send_aliyun_fun_event(websocket: WebSocket, task_id: str, event: str) -> None:
    await websocket.send_json({"header": {"task_id": task_id, "event": event}, "payload": {}})


async def send_aliyun_fun_result(websocket: WebSocket, task_id: str, text: str, *, sentence_end: bool) -> None:
    await websocket.send_json(
        {
            "header": {"task_id": task_id, "event": "result-generated"},
            "payload": {"output": {"sentence": {"text": text, "sentence_end": sentence_end}}},
        }
    )


async def websocket_send_error(websocket: WebSocket, code: str, message: str) -> None:
    await websocket.send_json({"type": "error", "event_id": new_event_id(), "error": {"code": code, "message": message}})


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def model_info(model_id: str) -> dict:
    return {"id": model_id, "object": "model", "created": 0, "owned_by": "local"}


def worker_process_alive(state: State) -> bool:
    process = state.worker_process
    return bool(process and process.is_alive())


def is_worker_alive(state: State) -> bool:
    process = state.worker_process
    if process is None:
        return False
    if process.is_alive():
        return True
    remove_worker_pidfile(process.pid)
    state.worker_process = None
    state.request_queue = None
    state.response_queue = None
    return False


def ensure_worker_started(state: State, config: Config) -> None:
    if is_worker_alive(state) and state.request_queue is not None and state.response_queue is not None:
        return
    cleanup_stale_worker_pidfile()
    state.request_queue = None
    state.response_queue = None
    state.worker_process = None

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()
    process = ctx.Process(
        target=worker_main,
        args=(config, request_queue, response_queue),
        name="voxt-qwen3-asr-worker",
    )
    process.daemon = True
    process.start()
    write_worker_pidfile(process.pid)
    state.request_queue = request_queue
    state.response_queue = response_queue
    state.worker_process = process
    print(f"Started ASR worker pid={process.pid}", flush=True)


def stop_worker(state: State, *, reason: str) -> None:
    process = state.worker_process
    if process is None:
        return
    print(f"Stopping ASR worker after {reason} pid={process.pid}", flush=True)
    if state.request_queue is not None:
        try:
            state.request_queue.put({"type": "shutdown"})
        except Exception:
            pass
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    remove_worker_pidfile(process.pid)
    state.worker_process = None
    state.request_queue = None
    state.response_queue = None


async def transcribe_in_worker(
    state: State,
    config: Config,
    *,
    audio_path: str,
    preview: bool,
    audio_duration: float | None,
    language: str | None,
    prompt: str,
    max_new_tokens: int | None,
    timeout_sec: float,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        transcribe_in_worker_sync,
        state,
        config,
        audio_path,
        preview,
        audio_duration,
        language,
        prompt,
        max_new_tokens,
        timeout_sec,
    )


def transcribe_in_worker_sync(
    state: State,
    config: Config,
    audio_path: str,
    preview: bool,
    audio_duration: float | None,
    language: str | None,
    prompt: str,
    max_new_tokens: int | None,
    timeout_sec: float,
) -> dict[str, Any]:
    ensure_worker_started(state, config)
    if state.request_queue is None or state.response_queue is None:
        raise RuntimeError("ASR worker did not start")

    request_id = f"{time.time_ns()}"
    state.request_queue.put(
        {
            "type": "transcribe",
            "id": request_id,
            "audio_path": audio_path,
            "preview": preview,
            "audio_duration": audio_duration,
            "language": language,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
    )
    if not is_worker_alive(state):
        raise RuntimeError("ASR worker exited before accepting the request")

    deadline = time.monotonic() + timeout_sec
    while True:
        if time.monotonic() >= deadline:
            stop_worker(state, reason=f"transcription timeout ({timeout_sec:.1f}s)")
            raise RuntimeError(f"ASR worker timed out after {timeout_sec:.1f}s")
        try:
            response = state.response_queue.get(timeout=1)
        except queue.Empty:
            if not is_worker_alive(state):
                raise RuntimeError("ASR worker exited before returning a result")
            continue
        if response.get("id") != request_id:
            continue
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "ASR worker failed")
        return response


def cleanup_stale_worker_pidfile() -> None:
    try:
        pid = int(WORKER_PIDFILE.read_text().strip())
    except Exception:
        return
    if pid <= 0 or pid == os.getpid():
        remove_worker_pidfile(pid)
        return
    try:
        os.kill(pid, 0)
    except OSError:
        remove_worker_pidfile(pid)
        return
    print(f"Terminating stale ASR worker pid={pid}", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            remove_worker_pidfile(pid)
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    remove_worker_pidfile(pid)


def write_worker_pidfile(pid: int | None) -> None:
    if not pid:
        return
    try:
        WORKER_PIDFILE.write_text(f"{pid}\n", encoding="utf-8")
    except OSError:
        pass


def remove_worker_pidfile(pid: int | None = None) -> None:
    try:
        if pid is not None and WORKER_PIDFILE.exists():
            current = int(WORKER_PIDFILE.read_text().strip())
            if current != pid:
                return
        WORKER_PIDFILE.unlink(missing_ok=True)
    except Exception:
        pass


def transcription_timeout_seconds(audio_duration: float | None) -> float:
    if audio_duration and audio_duration > 0:
        return max(300.0, audio_duration * 10.0)
    return 300.0


def worker_main(config: Config, request_queue: mp.Queue, response_queue: mp.Queue) -> None:
    from mlx_qwen3_asr.session import Session

    final_session = None
    preview_session = None
    preview_stream: PreviewStreamSession | None = None
    dtype = {"float16": mx.float16, "float32": mx.float32, "bfloat16": mx.bfloat16}.get(config.dtype, mx.float16)
    timeout = config.unload_after_sec if config.unload_after_sec > 0 else None

    def clear_mlx_cache() -> None:
        for clear in (mx.clear_cache, getattr(mx, "metal", None) and mx.metal.clear_cache):
            if clear:
                try:
                    clear()
                except Exception:
                    pass

    def get_session(preview: bool):
        nonlocal final_session, preview_session
        if preview:
            if preview_session is not None:
                return preview_session
            if config.preview_model_path == config.model_path and final_session is not None:
                preview_session = final_session
                return preview_session
            load_started = time.perf_counter()
            print(f"Loading preview ASR model from {config.preview_model_path} dtype={config.dtype}", flush=True)
            preview_session = Session(config.preview_model_path, dtype=dtype)
            print(f"Preview ASR model loaded elapsed={time.perf_counter() - load_started:.2f}s", flush=True)
            return preview_session

        if final_session is not None:
            return final_session
        if config.preview_model_path == config.model_path and preview_session is not None:
            final_session = preview_session
            return final_session
        load_started = time.perf_counter()
        print(f"Loading final ASR model from {config.model_path} dtype={config.dtype}", flush=True)
        final_session = Session(config.model_path, dtype=dtype)
        print(f"Final ASR model loaded elapsed={time.perf_counter() - load_started:.2f}s", flush=True)
        return final_session

    def preview_chunk_max_new_tokens(request_cap: int | None) -> int:
        # mlx-qwen3-asr streaming applies explicit max_new_tokens per internal
        # decode turn, not as a whole-stream budget. Keep preview conservative
        # so the old sync preview cap does not multiply by every 0.5s chunk.
        if request_cap is None:
            return 32
        return max(16, min(request_cap, 32))

    def init_preview_stream(language: str | None, prompt: str, max_new_tokens: int | None):
        return get_session(True).init_streaming(
            context=prompt,
            language=language,
            sample_rate=16000,
            chunk_size_sec=0.5,
            max_context_sec=preview_context_seconds(config),
            finalization_mode="latency",
            max_new_tokens=max_new_tokens,
        )

    def audio_diff_ok(reference: np.ndarray, candidate: np.ndarray, threshold: float = 0.02) -> bool:
        count = min(reference.size, candidate.size)
        if count <= 0:
            return False
        return float(np.mean(np.abs(reference[:count] - candidate[:count]))) <= threshold

    def handle_preview_message(message: dict[str, Any]) -> dict[str, Any]:
        nonlocal preview_stream
        sample_rate = 16000
        overlap_samples = sample_rate
        silence_rms = 0.003
        tolerance_samples = int(0.25 * sample_rate)
        language = message.get("language")
        prompt = message.get("prompt") or ""
        chunk_cap = preview_chunk_max_new_tokens(message.get("max_new_tokens"))
        duration = message.get("audio_duration")
        estimated_total = int(round(duration * sample_rate)) if duration and duration > 0 else None
        reset_reason = None
        feed_started = time.perf_counter()

        if (
            preview_stream is None
            or preview_stream.language != language
            or preview_stream.prompt != prompt
            or preview_stream.max_new_tokens != chunk_cap
        ):
            reset_reason = "new_stream" if preview_stream is None else "request_context_changed"

        delta: np.ndarray
        if reset_reason is None and estimated_total is not None:
            previous_total = preview_stream.last_sample_count
            if estimated_total <= previous_total:
                if estimated_total < previous_total - tolerance_samples:
                    reset_reason = "audio_shrank"
                else:
                    head = decode_audio_float32(message["audio_path"], sample_rate=sample_rate, start_sec=0, duration_sec=1.0)
                    if not audio_diff_ok(preview_stream.first_head, head):
                        reset_reason = "same_length_head_mismatch"
                    else:
                        preview_stream.last_used_at = time.monotonic()
                        return {
                            "text": preview_stream.last_text,
                            "preview_streaming": True,
                            "preview_delta_samples": 0,
                            "preview_total_samples": previous_total,
                            "preview_stream_chars": len(preview_stream.last_text),
                        }

        if reset_reason is None:
            previous_total = preview_stream.last_sample_count
            start_sample = max(0, previous_total - overlap_samples)
            decode_duration = None
            if estimated_total is not None:
                decode_duration = max(0.1, (estimated_total - start_sample) / sample_rate)
            decoded = decode_audio_float32(message["audio_path"], sample_rate=sample_rate, start_sec=start_sample / sample_rate, duration_sec=decode_duration)
            expected_overlap = previous_total - start_sample
            overlap = decoded[:expected_overlap]
            tail_reference = preview_stream.last_tail[-expected_overlap:]
            tail_rms = float(np.sqrt(np.mean(np.square(tail_reference)))) if tail_reference.size else 0.0
            if tail_rms >= silence_rms:
                if not audio_diff_ok(tail_reference, overlap):
                    reset_reason = "tail_mismatch"
            else:
                head = decode_audio_float32(message["audio_path"], sample_rate=sample_rate, start_sec=0, duration_sec=1.0)
                if not audio_diff_ok(preview_stream.first_head, head):
                    reset_reason = "silent_tail_head_mismatch"
            delta = decoded[expected_overlap:] if reset_reason is None else np.array([], dtype=np.float32)
            observed_total = start_sample + decoded.size
        else:
            observed_total = 0
            delta = np.array([], dtype=np.float32)

        if reset_reason is not None:
            if not (observed_total > 0 and decoded.size >= max(0, observed_total - tolerance_samples)):
                decoded = decode_audio_float32(message["audio_path"], sample_rate=sample_rate)
            chunk_state = init_preview_stream(language, prompt, chunk_cap)
            reset_count = (preview_stream.reset_count + 1) if preview_stream is not None else 1
            preview_stream = PreviewStreamSession(
                stream_state=chunk_state,
                last_sample_count=0,
                first_head=decoded[:overlap_samples].copy(),
                last_tail=np.array([], dtype=np.float32),
                last_text="",
                language=language,
                prompt=prompt,
                max_new_tokens=chunk_cap,
                last_used_at=time.monotonic(),
                reset_count=reset_count,
            )
            delta = decoded
            observed_total = decoded.size

        if delta.size > 0:
            preview_stream.stream_state = get_session(True).feed_audio(delta, preview_stream.stream_state)
            preview_stream.last_text = preview_stream.stream_state.text or ""
            preview_stream.last_sample_count = observed_total
            if preview_stream.first_head.size == 0:
                preview_stream.first_head = decode_audio_float32(message["audio_path"], sample_rate=sample_rate, start_sec=0, duration_sec=1.0)[:overlap_samples].copy()
            tail_source = delta if delta.size >= overlap_samples else decode_audio_float32(message["audio_path"], sample_rate=sample_rate, start_sec=max(0, (observed_total - overlap_samples) / sample_rate), duration_sec=overlap_samples / sample_rate)
            preview_stream.last_tail = tail_source[-overlap_samples:].copy()
        preview_stream.last_used_at = time.monotonic()

        elapsed = time.perf_counter() - feed_started
        total_samples = preview_stream.last_sample_count
        print(
            "Preview stream "
            f"reset={reset_reason or 'no'} total={total_samples / sample_rate:.2f}s delta={delta.size / sample_rate:.2f}s "
            f"feed_elapsed={elapsed:.2f}s chars={len(preview_stream.last_text)}",
            flush=True,
        )
        response = {
            "text": preview_stream.last_text,
            "preview_streaming": True,
            "preview_delta_samples": int(delta.size),
            "preview_total_samples": int(total_samples),
            "preview_stream_chars": len(preview_stream.last_text),
        }
        if reset_reason is not None:
            response["preview_reset_reason"] = reset_reason
        return response

    try:
        while True:
            try:
                message = request_queue.get(timeout=timeout)
            except queue.Empty:
                print(f"ASR worker exiting after {config.unload_after_sec}s idle", flush=True)
                break

            if message.get("type") == "shutdown":
                print("ASR worker exiting after shutdown", flush=True)
                break
            if message.get("type") != "transcribe":
                continue

            try:
                if bool(message["preview"]):
                    try:
                        result = handle_preview_message(message)
                    except Exception as preview_exc:
                        print(
                            f"WARNING: preview streaming failed; falling back to sync preview transcription: {preview_exc}",
                            flush=True,
                        )
                        fallback_started = time.perf_counter()
                        fallback = get_session(True).transcribe(
                            message["audio_path"],
                            language=message.get("language"),
                            context=message.get("prompt") or "",
                            return_timestamps=False,
                            return_chunks=False,
                            max_new_tokens=message.get("max_new_tokens"),
                        )
                        result = {
                            "text": fallback.text or "",
                            "preview_streaming": False,
                            "preview_fallback_reason": type(preview_exc).__name__,
                            "preview_fallback_elapsed": time.perf_counter() - fallback_started,
                        }
                    response_queue.put({"id": message["id"], "ok": True, **result})
                    continue

                preview_stream = None
                result = get_session(False).transcribe(
                    message["audio_path"],
                    language=message.get("language"),
                    context=message.get("prompt") or "",
                    return_timestamps=False,
                    return_chunks=False,
                    max_new_tokens=message.get("max_new_tokens"),
                )
                response_queue.put({"id": message["id"], "ok": True, "text": result.text or ""})
            except Exception as exc:
                traceback.print_exc()
                response_queue.put({"id": message.get("id"), "ok": False, "error": str(exc)})
    finally:
        preview_stream = None
        preview_session = None
        final_session = None
        clear_mlx_cache()


def check_auth(request: Request, api_key: str) -> None:
    if request.headers.get("authorization", "") != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def normalize_language(language: Optional[str]) -> Optional[str]:
    if not language or not language.strip():
        return None
    value = language.strip()
    return LANGUAGE_ALIASES.get(value.lower().replace("_", "-"), value)


async def write_upload_to_temp(file: UploadFile, tmp, max_file_size_mb: int) -> int:
    max_bytes = max_file_size_mb * 1024 * 1024
    size = 0
    try:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(status_code=413, detail=f"File too large: > {max_file_size_mb} MB")
            tmp.write(chunk)
        tmp.flush()
    finally:
        tmp.close()
    return size


def is_voxt_preview_upload(filename: str | None) -> bool:
    return bool(filename and filename.startswith("voxt-openai-preview-"))


def should_use_preview_model(request_model: str | None, filename: str | None, config: Config) -> bool:
    # VoxT preview uploads still send the configured final model, so the
    # preview filename is the strongest signal. For non-VoxT clients, an
    # explicit preview model form value can also select the preview path.
    if is_voxt_preview_upload(filename):
        return True
    return bool(request_model and request_model.strip() == config.preview_model)


def audio_duration_seconds(path: str) -> float | None:
    return wav_duration_seconds(path) or ffprobe_duration_seconds(path)


def decode_audio_float32(path: str, sample_rate: int = 16000, start_sec: float | None = None, duration_sec: float | None = None) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        if start_sec is None and duration_sec is None:
            return decode_wav_float32(path, sample_rate=sample_rate)
        raise RuntimeError("ffmpeg is required to decode partial preview audio")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    wav_input = Path(path).suffix.lower() == ".wav"
    # Input-side seeking is exact and cheap for WAV, which is what VoxT's HTTP
    # preview uploads use. For compressed formats, use output-side seeking for
    # sample-accurate overlap checks even though it may be slower.
    if wav_input and start_sec is not None and start_sec > 0:
        command.extend(["-ss", f"{start_sec:.6f}"])
    command.extend(["-i", path])
    if not wav_input and start_sec is not None and start_sec > 0:
        command.extend(["-ss", f"{start_sec:.6f}"])
    if duration_sec is not None and duration_sec > 0:
        command.extend(["-t", f"{duration_sec:.6f}"])
    command.extend(["-ar", str(sample_rate), "-ac", "1", "-f", "f32le", "pipe:1"])
    timeout = max(15.0, (duration_sec or audio_duration_seconds(path) or 1.0) + 10.0)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)
    if not result.stdout:
        return np.array([], dtype=np.float32)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def decode_wav_float32(path: str, sample_rate: int = 16000) -> np.ndarray:
    try:
        with wave.open(path, "rb") as wav:
            channels = wav.getnchannels()
            rate = wav.getframerate()
            width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
    except Exception as exc:
        raise RuntimeError("ffmpeg is unavailable and fallback WAV decoding failed") from exc

    if rate != sample_rate:
        raise RuntimeError(f"ffmpeg is unavailable and WAV sample rate is {rate}, expected {sample_rate}")
    if width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"ffmpeg is unavailable and WAV sample width {width} is unsupported")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32, copy=False)


def preview_context_seconds(config: Config) -> float:
    return config.preview_max_audio_seconds if config.preview_max_audio_seconds > 0 else 86400.0


def preview_context_label(config: Config) -> str:
    return f"{config.preview_max_audio_seconds:.2f}s" if config.preview_max_audio_seconds > 0 else "unlimited"


def wav_duration_seconds(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate > 0 else None
    except Exception:
        return None


def ffprobe_duration_seconds(path: str) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=True,
        )
        value = float(result.stdout.strip())
        return value if value > 0 else None
    except Exception:
        return None


def print_efficiency(audio_duration: float | None, elapsed: float, text_chars: int, *, is_preview: bool, max_new_tokens: int | None, metadata: dict[str, Any] | None = None) -> None:
    kind = "preview" if is_preview else "final"
    token_cap = f" max_new_tokens={max_new_tokens}" if max_new_tokens is not None else ""
    if is_preview and metadata and metadata.get("preview_streaming"):
        delta_samples = int(metadata.get("preview_delta_samples") or 0)
        total_samples = int(metadata.get("preview_total_samples") or 0)
        reset = metadata.get("preview_reset_reason")
        delta_duration = delta_samples / 16000 if delta_samples > 0 else 0.0
        total_duration = total_samples / 16000 if total_samples > 0 else (audio_duration or 0.0)
        reset_log = f" reset={reset}" if reset else ""
        if delta_duration > 0 and elapsed > 0:
            speed = delta_duration / elapsed
            rtf = elapsed / delta_duration
            print(
                f"Transcription done kind={kind} total={total_duration:.2f}s delta={delta_duration:.2f}s elapsed={elapsed:.2f}s "
                f"speed={speed:.2f}x rtf={rtf:.3f} chars={text_chars}{token_cap}{reset_log}",
                flush=True,
            )
        else:
            print(f"Transcription done kind={kind} total={total_duration:.2f}s delta=0.00s elapsed={elapsed:.2f}s chars={text_chars}{token_cap}{reset_log}", flush=True)
        return
    if audio_duration and audio_duration > 0 and elapsed > 0:
        speed = audio_duration / elapsed
        rtf = elapsed / audio_duration
        print(f"Transcription done kind={kind} audio={audio_duration:.2f}s elapsed={elapsed:.2f}s speed={speed:.2f}x rtf={rtf:.3f} chars={text_chars}{token_cap}", flush=True)
    else:
        print(f"Transcription done kind={kind} elapsed={elapsed:.2f}s chars={text_chars}{token_cap}", flush=True)


def masked_api_key(api_key: str) -> str:
    if not api_key:
        return "not set"
    if api_key == DEFAULT_API_KEY:
        return DEFAULT_API_KEY
    return "configured"


def print_startup_summary(args: argparse.Namespace, model_path: str, preview_model_path: str) -> None:
    print(
        f"""
Starting voxt-qwen3-asr

Server:
  Host:              {args.host}
  Port:              {args.port}
  Health:            http://127.0.0.1:{args.port}/health
  VoxT endpoint:     http://<host-ip>:{args.port}/v1/audio/transcriptions
  Realtime WS:       ws://<host-ip>:{args.port}/v1/realtime

Model:
  Final model:       {args.model}
  Final loaded from: {model_path}
  Preview model:     {args.preview_model}
  Preview from:      {preview_model_path}
  DType:             {args.dtype}
  Cache:             {os.environ['HUGGINGFACE_HUB_CACHE']}
  HF endpoint:       {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}

VoxT Remote ASR:
  Provider:          OpenAI Transcribe
  Endpoint:          http://<host-ip>:{args.port}/v1/audio/transcriptions
  API Key:           {masked_api_key(args.api_key)}
  Model:             {args.model}
  Preview:           Enable "Chunk Pseudo Realtime Preview" in VoxT if desired

VoxT Realtime ASR:
  Provider:          Aliyun Qwen Realtime ASR
  Endpoint:          ws://<host-ip>:{args.port}/v1/realtime
  API Key:           {masked_api_key(args.api_key)}
  Model:             qwen3-asr-flash-realtime (served by {args.model})
  Audio:             PCM16 mono 16k
  Unload after:      {args.unload_after_sec}s idle (0 = keep loaded)
""".strip(),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
