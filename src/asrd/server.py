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
# asrd

Local OpenAI-compatible ASR daemon for Apple Silicon/MLX,
powered by `mlx-qwen3-asr`. The main service lives in `asrd.server`, with
small support modules under `src/asrd/`.

## Run

```bash
uv run asrd serve
```

Defaults:

- endpoint: `http://127.0.0.1:5100/v1/audio/transcriptions`
- realtime endpoint: `ws://127.0.0.1:5100/v1/realtime`
- bind: `127.0.0.1:5100`
- API key: `password`
- final model: `mlx-community/Qwen3-ASR-1.7B-8bit`
- preview model: `mlx-community/Qwen3-ASR-0.6B-6bit`
- model cache: `~/.cache/huggingface/hub`
- Hugging Face endpoint: `https://hf-mirror.com`
- idle unload: `30s`
- HTTP preview: 0.6B last-20s chunk, 1.4s cadence
- HTTP/WS correction: 1.7B forced-aligner rolling windows

## VoxT settings

```text
Provider: OpenAI Transcribe
Endpoint: http://127.0.0.1:5100/v1/audio/transcriptions
API Key: password
Model: mlx-community/Qwen3-ASR-1.7B-8bit
Chunk Pseudo Realtime Preview: On
```

Realtime WebSocket:

```text
Provider: Aliyun Bailian ASR
Endpoint: ws://127.0.0.1:5100/v1/realtime
API Key: password
Model: qwen3-asr-flash-realtime
Audio: PCM16 mono 16k
```

For LAN use, replace `127.0.0.1` with the server machine IP.

## Notes

- VoxT preview uploads named `voxt-openai-preview-*.wav` use the preview model on the last 20s chunk.
- VoxT HTTP previews also advance the same 1.7B rolling correction cache used by WS.
- HTTP final uploads with matching VoxT preview cache use bounded final-tail/catch-up windows.
- Plain Whisper/OpenAI HTTP uploads without preview cache use full one-shot final transcription.
- Realtime preview/correction models are loaded lazily and killed after idle timeout.
- Busy HTTP preview requests return cached text instead of clearing preview text.
- Logs include speed and RTF, e.g. `speed=14.11x rtf=0.071`.
- Use `--debug` to preserve related audio and compact JSON metadata sidecars under `debug_dir`.

Useful commands:

```bash
uv run asrd serve --readme
uv run asrd serve --unload-after-sec 0
uv run asrd serve --debug
uv run asrd service install
HF_ENDPOINT=https://huggingface.co uv run asrd serve
```
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import multiprocessing as mp
import os
import queue
import re
import signal
import shutil
import sys
import tempfile
import time
import traceback
import uuid
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from huggingface_hub import snapshot_download

from .audio import (
    audio_diff_ok,
    audio_duration_seconds,
    audio_slice_seconds,
    decode_audio_float32,
    decode_pcm16_base64,
    full_audio_from_parts,
    is_voxt_preview_upload,
    pcm16_bytes_to_float32,
)
from .util import masked_api_key, new_event_id, read_json, safe_debug_filename


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5100
DEFAULT_API_KEY = "password"
DEFAULT_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
DEFAULT_PREVIEW_MODEL = "mlx-community/Qwen3-ASR-0.6B-6bit"
DEFAULT_DTYPE = "float16"
DEFAULT_MAX_FILE_SIZE_MB = 512
DEFAULT_UNLOAD_AFTER_SEC = 30.0
DEFAULT_DEBUG_DIR = str(Path.home() / "Library" / "Logs" / "com.sid.asrd.debug")
ASR_SAMPLE_RATE = 16000
CHUNK_PREVIEW_WINDOW_SEC = 20.0
WS_CHUNK_PREVIEW_UPDATE_SEC = 2.0
HTTP_CHUNK_PREVIEW_UPDATE_SEC = 1.4
WS_FEED_BATCH_SAMPLES = int(ASR_SAMPLE_RATE * WS_CHUNK_PREVIEW_UPDATE_SEC)
ROLLING_CORRECTION_WINDOW_SEC = 40.0
ROLLING_CORRECTION_STRIDE_SEC = 10.0
ROLLING_CORRECTION_COMMIT_POSITION_SEC = 20.0
ROLLING_CORRECTION_COMMIT_SPAN_SEC = 10.0
FINAL_CATCHUP_MAX_WINDOW_SEC = 60.0
FINAL_CATCHUP_LEFT_CONTEXT_SEC = 10.0
FINAL_CATCHUP_RIGHT_CONTEXT_SEC = 10.0
REALTIME_FINAL_WAIT_TIMEOUT_SEC = 4.5
HTTP_PREVIEW_SESSION_HEAD_SEC = 1.0
HTTP_PREVIEW_SESSION_TAIL_SEC = 1.0
HTTP_PREVIEW_SESSION_MATCH_RMS = 0.02
HTTP_PREVIEW_SESSION_MATCH_TOLERANCE_SEC = 0.25
TEXT_BOUNDARY_PUNCT = set("，。！？；：、,.!?;:（）()[]【】{}《》<>“”\"'‘’…—-·/\\|@#$%^&*_+=~`")
ASR_CONTEXT_TERM_KEYS = {
    "hotwords",
    "keywords",
    "phrases",
    "terms",
    "vocabulary",
    "custom_vocabulary",
    "customVocabulary",
    "contextual_phrases",
    "contextualPhrases",
}
ASR_CONTEXT_ITEM_KEYS = {"text", "term", "word", "phrase", "name", "value"}
ASR_CONTEXT_MAX_TERMS = 80
ASR_CONTEXT_MAX_CHARS = 2000
CORRECTION_WORKER_PIDFILE = Path("/tmp/asrd-correction-worker.pid")

LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "zh-hans": "Chinese",
    "zh-hant": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "zh-hk": "Chinese",
    "cmn-hans-cn": "Chinese",
    "cmn-hant-tw": "Chinese",
    "yue-hant-hk": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "ko": "Korean",
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
    debug: bool
    debug_dir: str


@dataclass
class State:
    ws_lock: asyncio.Lock | None = None
    ws_correction_lock: asyncio.Lock | None = None
    http_realtime_lock: asyncio.Lock | None = None
    ws_correction_process: mp.Process | None = None
    ws_correction_request_queue: mp.Queue | None = None
    ws_correction_response_queue: mp.Queue | None = None
    ws_correction_responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    ws_session: Any | None = None
    http_realtime_session: Any | None = None
    http_preview_inflight: bool = False
    ws_executor: concurrent.futures.ThreadPoolExecutor | None = None
    active_ws_count: int = 0
    active_http_count: int = 0
    unload_task: asyncio.Task | None = None
    last_used_at: float = 0.0


@dataclass
class WsRollingCorrectionSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    next_window_start_sec: float = 0.0
    stable_until_sec: float = 0.0
    scheduled_until_sec: float = 0.0
    stable_text: str = ""
    pending_request_ids: set[str] = field(default_factory=set)
    pending_requests: dict[str, dict[str, Any]] = field(default_factory=dict)
    ready_results: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class WsChunkPreviewState:
    text: str = ""
    window_start_sec: float = 0.0
    window_end_sec: float = 0.0


@dataclass
class HttpRealtimeSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    language: str | None = None
    prompt: str = ""
    first_head: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    last_tail: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    last_sample_count: int = 0
    last_preview_audio_sec: float = 0.0
    last_seen_at: float = 0.0
    preview_state: WsChunkPreviewState = field(default_factory=WsChunkPreviewState)
    rolling_correction: WsRollingCorrectionSession = field(default_factory=WsRollingCorrectionSession)
    last_response_text: str = ""
    final_requested: bool = False


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
            max_file_size_mb=args.max_file_size_mb,
            debug=args.debug,
            debug_dir=args.debug_dir,
        )
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, ws_ping_interval=None)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local OpenAI-compatible Qwen3-ASR server.")
    parser.add_argument("--readme", action="store_true", help="Print the embedded README and exit.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Final transcription model.")
    parser.add_argument(
        "--preview-model",
        default=DEFAULT_PREVIEW_MODEL,
        help="Model used for HTTP chunk preview and realtime partials.",
    )
    parser.add_argument("--dtype", choices=sorted(DTYPE_NAMES), default=os.environ.get("ASR_DTYPE", DEFAULT_DTYPE))
    parser.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache"),
        help="Model cache root. Defaults to ~/.cache so models use the global Hugging Face cache.",
    )
    parser.add_argument("--max-file-size-mb", type=int, default=int(os.environ.get("ASR_MAX_FILE_SIZE_MB", DEFAULT_MAX_FILE_SIZE_MB)))
    parser.add_argument(
        "--unload-after-sec",
        type=float,
        default=DEFAULT_UNLOAD_AFTER_SEC,
        help="Unload models after this many idle seconds. Use 0 to keep models loaded.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Log returned text and preserve final uploaded audio for debugging.",
    )
    parser.add_argument(
        "--debug-dir",
        default=DEFAULT_DEBUG_DIR,
        help=f"Directory for --debug artifacts. Defaults to {DEFAULT_DEBUG_DIR}.",
    )
    parser.add_argument("--log-level", default=os.environ.get("ASR_LOG_LEVEL", "info"))
    return parser.parse_args(argv)


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


def create_app(config: Config) -> FastAPI:
    state = State(
        ws_lock=asyncio.Lock(),
        ws_correction_lock=asyncio.Lock(),
        http_realtime_lock=asyncio.Lock(),
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
        stop_ws_correction_worker(state, reason="shutdown")

    app = FastAPI(title="asrd", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        correction_worker_alive = ws_correction_worker_alive(state)
        idle_for = time.monotonic() - state.last_used_at if state.last_used_at else None
        idle_remaining = None
        if (correction_worker_alive or state.ws_session is not None) and idle_for is not None and config.unload_after_sec > 0:
            idle_remaining = max(0.0, config.unload_after_sec - idle_for)
        return {
            "status": "ok",
            "model": config.model,
            "preview_model": config.preview_model,
            "preview_model_loaded": state.ws_session is not None,
            "correction_worker_alive": correction_worker_alive,
            "models_loaded": state.ws_session is not None or correction_worker_alive,
            "sample_rate": ASR_SAMPLE_RATE,
            "chunk_preview_window_sec": CHUNK_PREVIEW_WINDOW_SEC,
            "ws_chunk_preview_update_sec": WS_CHUNK_PREVIEW_UPDATE_SEC,
            "http_chunk_preview_update_sec": HTTP_CHUNK_PREVIEW_UPDATE_SEC,
            "rolling_correction_window_sec": ROLLING_CORRECTION_WINDOW_SEC,
            "rolling_correction_stride_sec": ROLLING_CORRECTION_STRIDE_SEC,
            "rolling_correction_commit_position_sec": ROLLING_CORRECTION_COMMIT_POSITION_SEC,
            "rolling_correction_commit_span_sec": ROLLING_CORRECTION_COMMIT_SPAN_SEC,
            "final_catchup_max_window_sec": FINAL_CATCHUP_MAX_WINDOW_SEC,
            "realtime_final_wait_timeout_sec": REALTIME_FINAL_WAIT_TIMEOUT_SEC,
            "http_cached_audio_sec": round(state.http_realtime_session.last_sample_count / ASR_SAMPLE_RATE, 3) if state.http_realtime_session else 0.0,
            "http_cached_stable_until_sec": round(state.http_realtime_session.rolling_correction.stable_until_sec, 3) if state.http_realtime_session else 0.0,
            "active_ws_count": state.active_ws_count,
            "active_http_count": state.active_http_count,
            "busy": bool(state.http_realtime_lock and state.http_realtime_lock.locked()),
            "last_used_at": state.last_used_at or None,
            "idle_for_sec": round(idle_for, 3) if idle_for is not None else None,
            "idle_remaining_sec": round(idle_remaining, 3) if idle_remaining is not None else None,
            "unload_after_sec": config.unload_after_sec,
            "debug": config.debug,
            "debug_dir": config.debug_dir if config.debug else None,
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
        preview_state: WsChunkPreviewState | None = None
        last_text = ""
        language = None
        prompt = ""
        ws_request_model = ""
        context_terms_count = 0
        session_configured = False
        fun_task_id = ""
        audio_buffer: list[np.ndarray] = []
        full_audio_parts: list[np.ndarray] = []
        ws_debug_audio_saved = False
        ws_debug_started_at = time.monotonic()
        ws_debug_chunks: list[list[float]] = []
        ws_debug_sample_cursor = 0
        rolling_correction = WsRollingCorrectionSession()
        send_lock = asyncio.Lock()
        preview_task: asyncio.Task | None = None
        preview_generation = 0
        final_requested = False
        buffered_samples = 0
        feed_threshold_samples = WS_FEED_BATCH_SAMPLES  # 2s at 16 kHz; avoid per-100ms decode backlog.

        async def send_partial_text(text: str, *, sentence_end: bool) -> None:
            nonlocal last_text
            if not text or text == last_text or final_requested:
                return
            if fun_task_id:
                last_text = text
                if config.debug:
                    log_debug_realtime_text(protocol="aliyun-fun", text=text, sentence_end=sentence_end)
                async with send_lock:
                    await send_aliyun_fun_result(websocket, fun_task_id, text, sentence_end=sentence_end)
                return

            delta = text[len(last_text) :] if text.startswith(last_text) else text
            last_text = text
            if config.debug:
                log_debug_realtime_text(protocol="aliyun-qwen", text=text, delta=delta, sentence_end=sentence_end)
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "conversation.item.input_audio_transcription.text",
                        "event_id": new_event_id(),
                        "text": text,
                        "delta": delta,
                    }
                )

        async def send_current_merged_partial() -> None:
            if preview_state is None:
                return
            text = merge_stable_and_live_window(
                rolling_correction.stable_text,
                preview_state.text,
                preview_state.window_start_sec,
                preview_state.window_end_sec,
                rolling_correction.stable_until_sec,
            )
            await send_partial_text(text, sentence_end=False)

        def current_best_realtime_text(full_audio: np.ndarray) -> tuple[str, float]:
            total_audio_sec = full_audio.size / ASR_SAMPLE_RATE if full_audio.size else 0.0
            if rolling_correction.stable_text and rolling_correction.stable_until_sec >= total_audio_sec - 0.05:
                return rolling_correction.stable_text, total_audio_sec
            if preview_state is None:
                return last_text, total_audio_sec
            text = merge_stable_and_live_window(
                rolling_correction.stable_text,
                preview_state.text,
                preview_state.window_start_sec,
                preview_state.window_end_sec,
                rolling_correction.stable_until_sec,
            )
            return text or last_text, total_audio_sec

        async def send_finish_interim_text(text: str, *, protocol: str) -> None:
            nonlocal last_text
            if not text:
                print(f"Realtime WS final interim skipped client={client} protocol={protocol} reason=empty-text", flush=True)
                return
            repeated = text == last_text
            if fun_task_id:
                if config.debug and not repeated:
                    log_debug_realtime_text(protocol="aliyun-fun", text=text, sentence_end=False)
                async with send_lock:
                    await send_aliyun_fun_result(websocket, fun_task_id, text, sentence_end=False)
                last_text = text
                print(
                    f"Realtime WS final interim text sent client={client} protocol=aliyun-fun "
                    f"chars={len(text)} repeated={repeated}",
                    flush=True,
                )
                return

            delta = "" if repeated else (text[len(last_text) :] if text.startswith(last_text) else text)
            if config.debug and not repeated:
                log_debug_realtime_text(protocol="aliyun-qwen", text=text, delta=delta, sentence_end=False)
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "conversation.item.input_audio_transcription.text",
                        "event_id": new_event_id(),
                        "text": text,
                        "delta": delta,
                    }
                )
            last_text = text
            print(
                f"Realtime WS final interim text sent client={client} protocol=aliyun-qwen "
                f"chars={len(text)} delta_chars={len(delta)} repeated={repeated}",
                flush=True,
            )

        async def cleanup_timed_out_final_corrections(*, protocol: str) -> None:
            if state.ws_correction_lock is None or not rolling_correction.pending_request_ids:
                return
            print(
                f"Realtime WS final timeout cleanup client={client} protocol={protocol} "
                f"pending={len(rolling_correction.pending_request_ids)}",
                flush=True,
            )
            async with state.ws_correction_lock:
                cancel_pending_ws_corrections(state, rolling_correction, reason="final-timeout")

        def discard_preview_tasks() -> None:
            nonlocal preview_generation
            preview_generation += 1

        def record_debug_ws_chunk(audio: np.ndarray) -> None:
            nonlocal ws_debug_sample_cursor
            if not config.debug:
                return
            if audio.size <= 0:
                return
            start_sample = ws_debug_sample_cursor
            end_sample = start_sample + int(audio.size)
            ws_debug_sample_cursor = end_sample
            received_elapsed = round(time.monotonic() - ws_debug_started_at, 3)
            ws_debug_chunks.append([len(ws_debug_chunks), start_sample, end_sample, received_elapsed])

        def preserve_debug_ws_audio_once(*, protocol: str, final_text: str = "") -> None:
            nonlocal ws_debug_audio_saved
            if not config.debug or ws_debug_audio_saved:
                return
            ws_debug_audio_saved = True
            preserve_debug_ws_audio(
                config,
                full_audio_from_parts(full_audio_parts),
                client=client,
                protocol=protocol,
                request_model=ws_request_model,
                preview_model=config.preview_model,
                final_model=config.model,
                language=language,
                chunks=ws_debug_chunks,
                final_text=final_text,
            )

        def schedule_chunk_preview(full_audio: np.ndarray) -> None:
            nonlocal preview_task, preview_generation, preview_state
            if final_requested or full_audio.size <= 0:
                return
            if preview_task is not None and not preview_task.done():
                return
            preview_generation += 1
            generation = preview_generation

            async def run_preview() -> None:
                nonlocal preview_state
                try:
                    updated = await ws_update_chunk_preview(
                        state,
                        config,
                        full_audio,
                        language=language,
                        prompt=prompt,
                    )
                    if final_requested or generation != preview_generation:
                        return
                    preview_state = updated
                    await send_current_merged_partial()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"Realtime WS preview task failed error={type(exc).__name__}: {exc}", flush=True)

            preview_task = asyncio.create_task(run_preview())

        try:
            while True:
                received = await websocket.receive()
                if "bytes" in received and received["bytes"] is not None:
                    if not session_configured or preview_state is None:
                        await websocket_send_error(websocket, "session_not_configured", "Send run-task/session.update before audio")
                        continue
                    audio = pcm16_bytes_to_float32(received["bytes"])
                    record_debug_ws_chunk(audio)
                    if audio.size:
                        full_audio_parts.append(audio)
                    audio_buffer.append(audio)
                    buffered_samples += int(audio.size)
                    if buffered_samples >= feed_threshold_samples:
                        chunk = np.concatenate(audio_buffer) if len(audio_buffer) > 1 else audio_buffer[0]
                        print(f"Realtime WS feed client={client} samples={chunk.size} seconds={chunk.size / ASR_SAMPLE_RATE:.2f}", flush=True)
                        audio_buffer = []
                        buffered_samples = 0
                        full_audio = full_audio_from_parts(full_audio_parts)
                        schedule_chunk_preview(full_audio)
                        await ws_update_rolling_correction(
                            state,
                            config,
                            rolling_correction,
                            full_audio,
                            language=language,
                            prompt=prompt,
                            final=False,
                        )
                        await send_current_merged_partial()
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
                    requested_model = str(payload.get("model") or websocket.query_params.get("model") or "").strip()
                    ws_request_model = requested_model
                    selected_model = config.preview_model
                    if requested_model and requested_model != config.preview_model:
                        print(
                            f"Realtime protocol model {requested_model!r} requested; using preview={config.preview_model!r} correction={config.model!r}",
                            flush=True,
                        )
                    hints = parameters.get("language_hints") if isinstance(parameters.get("language_hints"), list) else []
                    language = normalize_language(str(hints[0])) if hints else None
                    prompt, context_terms_count = build_asr_context_prompt(
                        "",
                        asr_context_terms_from_sources(parameters, payload, event, dict(websocket.query_params)),
                    )
                    print(
                        f"Realtime WS aliyun fun run-task client={client} task_id={fun_task_id} "
                        f"request_model={requested_model!r} selected_model={selected_model!r} "
                        f"language={language!r} context_terms={context_terms_count} prompt_chars={len(prompt)}",
                        flush=True,
                    )
                    preview_state = WsChunkPreviewState()
                    session_configured = True
                    await send_aliyun_fun_event(websocket, fun_task_id, "task-started")
                elif action == "finish-task":
                    final_requested = True
                    discard_preview_tasks()
                    if preview_state is None:
                        preview_state = WsChunkPreviewState()
                    if buffered_samples > 0:
                        chunk = np.concatenate(audio_buffer) if len(audio_buffer) > 1 else audio_buffer[0]
                        print(f"Realtime WS feed-final client={client} samples={chunk.size} seconds={chunk.size / ASR_SAMPLE_RATE:.2f}", flush=True)
                        audio_buffer = []
                        buffered_samples = 0
                    full_audio = full_audio_from_parts(full_audio_parts)
                    interim_text, total_audio_sec = current_best_realtime_text(full_audio)
                    await send_finish_interim_text(interim_text, protocol="aliyun-fun")
                    _, final_timed_out = await ws_update_final_correction_with_deadline(
                        state,
                        config,
                        rolling_correction,
                        full_audio,
                        language=language,
                        prompt=prompt,
                        timeout_sec=REALTIME_FINAL_WAIT_TIMEOUT_SEC,
                        client=client,
                        protocol="aliyun-fun",
                    )
                    text, total_audio_sec = current_best_realtime_text(full_audio)
                    print(
                        f"Realtime WS final ready client={client} protocol=aliyun-fun audio={total_audio_sec:.2f}s "
                        f"stable_until={rolling_correction.stable_until_sec:.2f}s text_chars={len(text)} "
                        f"timed_out={final_timed_out}",
                        flush=True,
                    )
                    preserve_debug_ws_audio_once(protocol="aliyun-fun", final_text=text)
                    final_send_stage = "result"
                    try:
                        if text:
                            if config.debug:
                                log_debug_realtime_text(protocol="aliyun-fun", text=text, sentence_end=True)
                            await send_aliyun_fun_result(websocket, fun_task_id, text, sentence_end=True)
                            print(f"Realtime WS final result sent client={client} protocol=aliyun-fun chars={len(text)}", flush=True)
                        else:
                            print(f"Realtime WS final result skipped client={client} protocol=aliyun-fun reason=empty-text", flush=True)
                        final_send_stage = "task-finished"
                        await send_aliyun_fun_event(websocket, fun_task_id, "task-finished")
                        print(f"Realtime WS final task-finished sent client={client} protocol=aliyun-fun", flush=True)
                        final_send_stage = "close"
                        await websocket.close(code=1000)
                        print(f"Realtime WS final close sent client={client} protocol=aliyun-fun", flush=True)
                    except Exception as exc:
                        print(
                            f"Realtime WS final send failed client={client} protocol=aliyun-fun stage={final_send_stage} "
                            f"error={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        raise
                    finally:
                        if final_timed_out:
                            await cleanup_timed_out_final_corrections(protocol="aliyun-fun")
                    return
                elif event_type == "session.update":
                    session = event.get("session") if isinstance(event.get("session"), dict) else {}
                    transcription = session.get("input_audio_transcription") if isinstance(session.get("input_audio_transcription"), dict) else {}
                    language = normalize_language(session.get("language") or transcription.get("language") or event.get("language"))
                    base_prompt = first_text_value(session.get("prompt"), session.get("context"), event.get("prompt"), event.get("context"))
                    prompt, context_terms_count = build_asr_context_prompt(
                        base_prompt,
                        asr_context_terms_from_sources(transcription, session, event, dict(websocket.query_params)),
                    )
                    requested_model = str(
                        session.get("model")
                        or transcription.get("model")
                        or event.get("model")
                        or websocket.query_params.get("model")
                        or ""
                    ).strip()
                    ws_request_model = requested_model
                    selected_model = config.preview_model
                    if requested_model and requested_model != config.preview_model:
                        print(
                            f"Realtime protocol model {requested_model!r} requested; using preview={config.preview_model!r} correction={config.model!r}",
                            flush=True,
                        )
                    print(
                        f"Realtime WS session.update client={client} request_model={requested_model!r} "
                        f"selected_model={selected_model!r} language={language!r} "
                        f"context_terms={context_terms_count} prompt_chars={len(prompt)}",
                        flush=True,
                    )
                    preview_state = WsChunkPreviewState()
                    session_configured = True
                    await websocket.send_json(
                        {
                            "type": "session.updated",
                            "event_id": new_event_id(),
                            "session": {
                                "model": selected_model,
                                "input_audio_format": "pcm16",
                                "sample_rate": ASR_SAMPLE_RATE,
                            },
                        }
                    )
                elif event_type == "input_audio_buffer.append":
                    if not session_configured or preview_state is None:
                        await websocket_send_error(websocket, "session_not_configured", "Send session.update before audio")
                        continue
                    audio_b64 = str(event.get("audio") or event.get("delta") or "")
                    try:
                        audio = decode_pcm16_base64(audio_b64)
                    except ValueError as exc:
                        await websocket_send_error(websocket, "invalid_audio", str(exc))
                        continue
                    record_debug_ws_chunk(audio)
                    if audio.size:
                        full_audio_parts.append(audio)
                    audio_buffer.append(audio)
                    buffered_samples += int(audio.size)
                    if buffered_samples >= feed_threshold_samples:
                        chunk = np.concatenate(audio_buffer) if len(audio_buffer) > 1 else audio_buffer[0]
                        print(f"Realtime WS feed client={client} samples={chunk.size} seconds={chunk.size / ASR_SAMPLE_RATE:.2f}", flush=True)
                        audio_buffer = []
                        buffered_samples = 0
                        full_audio = full_audio_from_parts(full_audio_parts)
                        schedule_chunk_preview(full_audio)
                        await ws_update_rolling_correction(
                            state,
                            config,
                            rolling_correction,
                            full_audio,
                            language=language,
                            prompt=prompt,
                            final=False,
                        )
                        await send_current_merged_partial()
                elif event_type == "session.finish":
                    final_requested = True
                    discard_preview_tasks()
                    if preview_state is None:
                        preview_state = WsChunkPreviewState()
                    if buffered_samples > 0:
                        chunk = np.concatenate(audio_buffer) if len(audio_buffer) > 1 else audio_buffer[0]
                        print(f"Realtime WS feed-final client={client} samples={chunk.size} seconds={chunk.size / ASR_SAMPLE_RATE:.2f}", flush=True)
                        audio_buffer = []
                        buffered_samples = 0
                    full_audio = full_audio_from_parts(full_audio_parts)
                    interim_text, total_audio_sec = current_best_realtime_text(full_audio)
                    await send_finish_interim_text(interim_text, protocol="aliyun-qwen")
                    _, final_timed_out = await ws_update_final_correction_with_deadline(
                        state,
                        config,
                        rolling_correction,
                        full_audio,
                        language=language,
                        prompt=prompt,
                        timeout_sec=REALTIME_FINAL_WAIT_TIMEOUT_SEC,
                        client=client,
                        protocol="aliyun-qwen",
                    )
                    text, total_audio_sec = current_best_realtime_text(full_audio)
                    should_send_text = bool(text and text != last_text)
                    print(
                        f"Realtime WS final ready client={client} protocol=aliyun-qwen audio={total_audio_sec:.2f}s "
                        f"stable_until={rolling_correction.stable_until_sec:.2f}s text_chars={len(text)} "
                        f"last_chars={len(last_text)} send_text={should_send_text} timed_out={final_timed_out}",
                        flush=True,
                    )
                    preserve_debug_ws_audio_once(protocol="aliyun-qwen", final_text=text)
                    final_send_stage = "text"
                    try:
                        if should_send_text:
                            delta = text[len(last_text) :] if text.startswith(last_text) else text
                            if config.debug:
                                log_debug_realtime_text(protocol="aliyun-qwen", text=text, delta=delta, sentence_end=True)
                            await websocket.send_json(
                                {
                                    "type": "conversation.item.input_audio_transcription.text",
                                    "event_id": new_event_id(),
                                    "text": text,
                                    "delta": delta,
                                }
                            )
                            print(
                                f"Realtime WS final text sent client={client} protocol=aliyun-qwen chars={len(text)} delta_chars={len(delta)}",
                                flush=True,
                            )
                        else:
                            print(
                                f"Realtime WS final text skipped client={client} protocol=aliyun-qwen reason={'empty-text' if not text else 'unchanged'}",
                                flush=True,
                            )
                        final_send_stage = "completed"
                        await websocket.send_json(
                            {
                                "type": "conversation.item.input_audio_transcription.completed",
                                "event_id": new_event_id(),
                                "text": text,
                                "transcript": text,
                            }
                        )
                        print(f"Realtime WS final completed sent client={client} protocol=aliyun-qwen chars={len(text)}", flush=True)
                        final_send_stage = "session.finished"
                        await websocket.send_json({"type": "session.finished", "event_id": new_event_id()})
                        print(f"Realtime WS final session.finished sent client={client} protocol=aliyun-qwen", flush=True)
                        final_send_stage = "close"
                        await websocket.close(code=1000)
                        print(f"Realtime WS final close sent client={client} protocol=aliyun-qwen", flush=True)
                    except Exception as exc:
                        print(
                            f"Realtime WS final send failed client={client} protocol=aliyun-qwen stage={final_send_stage} "
                            f"error={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        raise
                    finally:
                        if final_timed_out:
                            await cleanup_timed_out_final_corrections(protocol="aliyun-qwen")
                    return
                else:
                    await websocket_send_error(websocket, "unsupported_event", f"Unsupported event type: {event_type}")
        except WebSocketDisconnect:
            print(f"Realtime WS disconnected client={client}", flush=True)
        except RuntimeError as exc:
            if "disconnect message" not in str(exc):
                raise
            print(f"Realtime WS disconnected client={client} reason=disconnect-message", flush=True)
        finally:
            final_requested = True
            discard_preview_tasks()
            preserve_debug_ws_audio_once(protocol="aliyun-fun" if fun_task_id else "aliyun-qwen", final_text=last_text)
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
        if state.http_realtime_lock is None or state.ws_lock is None or state.ws_executor is None:
            raise HTTPException(status_code=503, detail="ASR runtime is not ready")

        request_model = require_non_empty_model(model)
        is_preview = is_voxt_preview_upload(file.filename)
        state.active_http_count += 1

        suffix = Path(file.filename or "upload.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="asrd_")
        tmp_path = tmp.name
        debug_audio_path: Path | None = None
        try:
            size = await write_upload_to_temp(file, tmp, config.max_file_size_mb)
            fmt = (response_format or "json").strip().lower()
            if fmt not in {"json", "text", "verbose_json"}:
                raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")

            lang = normalize_language(language)
            selected_model = config.preview_model if is_preview else config.model
            audio_duration = audio_duration_seconds(tmp_path)
            if config.debug:
                debug_audio_path = preserve_debug_final_audio(config, tmp_path, file.filename)
            preview_window_log = ""
            if audio_duration and is_preview:
                preview_window_log = f"audio={audio_duration:.2f}s preview_window={CHUNK_PREVIEW_WINDOW_SEC:.2f}s preview_update={HTTP_CHUNK_PREVIEW_UPDATE_SEC:.2f}s "
            print(
                "Transcription request "
                f"kind={'preview' if is_preview else 'final'} filename={file.filename!r} bytes={size} "
                f"{preview_window_log}"
                f"request_model={request_model!r} selected_model={selected_model!r} "
                f"language={language!r}->{lang!r} prompt_chars={len(prompt or '')} format={fmt!r}",
                flush=True,
            )

            started = time.perf_counter()
            full_audio = decode_audio_float32(tmp_path, sample_rate=ASR_SAMPLE_RATE)
            if audio_duration is None and full_audio.size:
                audio_duration = full_audio.size / ASR_SAMPLE_RATE
            if is_preview:
                result = await http_realtime_preview_transcription(
                    state,
                    config,
                    full_audio,
                    language=lang,
                    prompt=prompt or "",
                )
            else:
                result = await http_realtime_final_transcription(
                    state,
                    config,
                    full_audio,
                    language=lang,
                    prompt=prompt or "",
                )
            elapsed = time.perf_counter() - started
            text = (result.get("text") or "").strip()
            print_efficiency(audio_duration, elapsed, len(text), is_preview=is_preview, metadata=result)
            if config.debug:
                log_debug_transcription_text(is_preview=is_preview, text=text)
                if debug_audio_path is not None:
                    write_debug_transcription_metadata(
                        debug_audio_path,
                        protocol="whisper-chunk" if is_preview else "whisper",
                        request_model=request_model,
                        preview_model=config.preview_model,
                        final_model=config.model,
                        language=lang,
                        audio_duration=audio_duration,
                        audio_samples=int(full_audio.size),
                        text=text,
                    )

            if fmt == "text":
                return PlainTextResponse(text)
            if fmt == "verbose_json":
                return JSONResponse({"text": text, "task": "transcribe", "language": lang, "duration": audio_duration, "segments": []})
            return JSONResponse({"text": text})
        except HTTPException:
            raise
        except Exception as exc:
            if debug_audio_path is not None and not debug_audio_path.with_suffix(".json").exists():
                try:
                    debug_audio_path.unlink(missing_ok=True)
                except Exception:
                    pass
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": type(exc).__name__}})
        finally:
            state.active_http_count = max(0, state.active_http_count - 1)
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    return app


async def ws_update_chunk_preview(
    state: State,
    config: Config,
    full_audio: np.ndarray,
    *,
    language: str | None,
    prompt: str,
) -> WsChunkPreviewState:
    if state.ws_lock is None or state.ws_executor is None:
        raise RuntimeError("Realtime ASR runtime is not ready")
    loop = asyncio.get_running_loop()
    async with state.ws_lock:
        result = await loop.run_in_executor(
            state.ws_executor,
            ws_update_chunk_preview_sync,
            state,
            config,
            full_audio,
            language,
            prompt,
        )
        state.last_used_at = time.monotonic()
        return result


async def ws_update_final_correction_with_deadline(
    state: State,
    config: Config,
    correction: WsRollingCorrectionSession,
    full_audio: np.ndarray,
    *,
    language: str | None,
    prompt: str,
    timeout_sec: float,
    client: str,
    protocol: str,
) -> tuple[bool, bool]:
    audio_sec = full_audio.size / ASR_SAMPLE_RATE if full_audio.size else 0.0
    effective_timeout_sec = max(0.1, timeout_sec)
    started = time.perf_counter()
    try:
        updated = await asyncio.wait_for(
            ws_update_rolling_correction(
                state,
                config,
                correction,
                full_audio,
                language=language,
                prompt=prompt,
                final=True,
            ),
            timeout=effective_timeout_sec,
        )
        return updated, False
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started
        print(
            f"Realtime WS final correction timeout client={client} protocol={protocol} "
            f"timeout={effective_timeout_sec:.2f}s elapsed={elapsed:.2f}s audio={audio_sec:.2f}s "
            f"stable_until={correction.stable_until_sec:.2f}s pending={len(correction.pending_request_ids)}",
            flush=True,
        )
        return False, True


async def ws_update_rolling_correction(
    state: State,
    config: Config,
    correction: WsRollingCorrectionSession,
    full_audio: np.ndarray,
    *,
    language: str | None,
    prompt: str,
    final: bool,
) -> bool:
    if full_audio.size <= 0:
        return False
    if state.ws_correction_lock is None:
        raise RuntimeError("Realtime correction runtime is not ready")

    async with state.ws_correction_lock:
        updated = await ws_drain_correction_results(state, correction, wait_for_pending=False)
        if final:
            pending_windows = list(correction.pending_requests.values())
            had_pending = bool(correction.pending_request_ids)
            if correction.pending_request_ids:
                cancel_pending_ws_corrections(state, correction, reason="final-catchup")
            if ws_schedule_final_catchup_corrections(
                state,
                config,
                correction,
                full_audio,
                prefer_catchup_window=had_pending,
                pending_windows=pending_windows,
                language=language,
                prompt=prompt,
            ):
                updated = True
            updated = await ws_drain_correction_results(state, correction, wait_for_pending=True) or updated
        else:
            if ws_schedule_rolling_correction_windows(
                state,
                config,
                correction,
                full_audio,
                language=language,
                prompt=prompt,
            ):
                updated = True
            updated = await ws_drain_correction_results(state, correction, wait_for_pending=False) or updated

        state.last_used_at = time.monotonic()
        return updated


def update_http_session_audio(session: HttpRealtimeSession, full_audio: np.ndarray) -> None:
    head_samples = int(round(HTTP_PREVIEW_SESSION_HEAD_SEC * ASR_SAMPLE_RATE))
    tail_samples = int(round(HTTP_PREVIEW_SESSION_TAIL_SEC * ASR_SAMPLE_RATE))
    if full_audio.size > 0 and session.first_head.size == 0:
        session.first_head = full_audio[: min(head_samples, full_audio.size)].copy()
    if full_audio.size > 0:
        session.last_tail = full_audio[max(0, full_audio.size - tail_samples) :].copy()
    else:
        session.last_tail = np.array([], dtype=np.float32)
    session.last_sample_count = int(full_audio.size)
    session.last_seen_at = time.monotonic()


def http_session_reset_reason(
    session: HttpRealtimeSession | None,
    full_audio: np.ndarray,
    *,
    language: str | None,
    prompt: str,
) -> str | None:
    if session is None:
        return "new_session"
    if session.final_requested:
        return "after_final"
    if session.language != language or session.prompt != prompt:
        return "request_context_changed"
    tolerance_samples = int(round(HTTP_PREVIEW_SESSION_MATCH_TOLERANCE_SEC * ASR_SAMPLE_RATE))
    if full_audio.size + tolerance_samples < session.last_sample_count:
        return "audio_shrank"
    if session.first_head.size > 0:
        if full_audio.size < session.first_head.size:
            return "head_too_short"
        if not audio_diff_ok(session.first_head, full_audio[: session.first_head.size]):
            return "head_mismatch"
    if session.last_tail.size > 0 and full_audio.size >= session.last_sample_count:
        start = max(0, session.last_sample_count - session.last_tail.size)
        candidate = full_audio[start : start + session.last_tail.size]
        if candidate.size != session.last_tail.size:
            return "tail_missing"
        if not audio_diff_ok(session.last_tail, candidate):
            return "tail_mismatch"
    return None


def http_session_matches_audio(
    session: HttpRealtimeSession | None,
    full_audio: np.ndarray,
    *,
    language: str | None,
    prompt: str,
) -> bool:
    if session is None or session.final_requested or session.last_sample_count <= 0:
        return False
    return http_session_reset_reason(session, full_audio, language=language, prompt=prompt) is None


def http_session_merged_text(session: HttpRealtimeSession) -> str:
    total_sec = session.last_sample_count / ASR_SAMPLE_RATE if session.last_sample_count else 0.0
    correction = session.rolling_correction
    if correction.stable_text and correction.stable_until_sec >= total_sec - 0.05:
        return correction.stable_text
    text = merge_stable_and_live_window(
        correction.stable_text,
        session.preview_state.text,
        session.preview_state.window_start_sec,
        session.preview_state.window_end_sec,
        correction.stable_until_sec,
    )
    return text or session.last_response_text


def http_cached_response_text(state: State) -> str:
    session = state.http_realtime_session
    return session.last_response_text if session is not None else ""


async def cancel_http_session_corrections(state: State, session: HttpRealtimeSession | None, *, reason: str) -> None:
    if session is None or not session.rolling_correction.pending_request_ids:
        return
    if state.ws_correction_lock is None:
        return
    async with state.ws_correction_lock:
        cancel_pending_ws_corrections(state, session.rolling_correction, reason=reason)


async def http_realtime_preview_transcription(
    state: State,
    config: Config,
    full_audio: np.ndarray,
    *,
    language: str | None,
    prompt: str,
) -> dict[str, Any]:
    if state.http_realtime_lock is None:
        raise RuntimeError("HTTP realtime runtime is not ready")
    if state.http_realtime_lock.locked():
        text = http_cached_response_text(state)
        state.last_used_at = time.monotonic()
        print(f"HTTP realtime preview skipped busy cached_chars={len(text)}", flush=True)
        return {"text": text, "http_realtime": True, "http_preview_cached": True, "http_preview_skip_reason": "busy"}

    total_sec = full_audio.size / ASR_SAMPLE_RATE if full_audio.size else 0.0
    session_id = ""
    should_run_preview = False
    async with state.http_realtime_lock:
        session = state.http_realtime_session
        reset_reason = http_session_reset_reason(session, full_audio, language=language, prompt=prompt)
        if reset_reason is not None:
            await cancel_http_session_corrections(state, session, reason=f"http-preview-{reset_reason}")
            session = HttpRealtimeSession(language=language, prompt=prompt)
            state.http_realtime_session = session
            print(f"HTTP realtime preview session reset reason={reset_reason} session={session.session_id}", flush=True)

        update_http_session_audio(session, full_audio)
        await ws_update_rolling_correction(
            state,
            config,
            session.rolling_correction,
            full_audio,
            language=language,
            prompt=prompt,
            final=False,
        )

        if state.http_preview_inflight:
            text = http_session_merged_text(session)
            session.last_response_text = text
            state.last_used_at = time.monotonic()
            return {
                "text": text,
                "http_realtime": True,
                "http_preview_cached": True,
                "http_preview_skip_reason": "preview_inflight",
                "http_audio_sec": total_sec,
            }

        should_run_preview = (
            total_sec > 0
            and (
                not session.preview_state.text
                or total_sec - session.last_preview_audio_sec >= HTTP_CHUNK_PREVIEW_UPDATE_SEC - 0.05
            )
        )
        if not should_run_preview:
            text = http_session_merged_text(session)
            session.last_response_text = text
            state.last_used_at = time.monotonic()
            return {
                "text": text,
                "http_realtime": True,
                "http_preview_cached": True,
                "http_preview_skip_reason": "cadence",
                "http_audio_sec": total_sec,
            }

        state.http_preview_inflight = True
        session_id = session.session_id

    preview_state = WsChunkPreviewState()
    preview_error: Exception | None = None
    try:
        preview_state = await ws_update_chunk_preview(
            state,
            config,
            full_audio,
            language=language,
            prompt=prompt,
        )
    except Exception as exc:
        preview_error = exc

    async with state.http_realtime_lock:
        session = state.http_realtime_session
        if session is None or session.session_id != session_id:
            state.http_preview_inflight = False
            text = http_cached_response_text(state)
            if preview_error is not None:
                raise preview_error
            return {"text": text, "http_realtime": True, "http_preview_cached": True, "http_preview_skip_reason": "session_changed"}
        state.http_preview_inflight = False
        if preview_error is not None:
            raise preview_error
        if not session.final_requested and preview_state.window_end_sec + 0.05 >= session.preview_state.window_end_sec:
            session.preview_state = preview_state
            session.last_preview_audio_sec = preview_state.window_end_sec
        text = http_session_merged_text(session)
        session.last_response_text = text
        state.last_used_at = time.monotonic()
        return {
            "text": text,
            "http_realtime": True,
            "http_preview_cached": False,
            "http_audio_sec": total_sec,
            "http_preview_window_start_sec": preview_state.window_start_sec,
            "http_preview_window_end_sec": preview_state.window_end_sec,
            "http_stable_until_sec": session.rolling_correction.stable_until_sec,
        }


async def http_realtime_final_transcription(
    state: State,
    config: Config,
    full_audio: np.ndarray,
    *,
    language: str | None,
    prompt: str,
) -> dict[str, Any]:
    if state.http_realtime_lock is None:
        raise RuntimeError("HTTP realtime runtime is not ready")
    total_sec = full_audio.size / ASR_SAMPLE_RATE if full_audio.size else 0.0

    async with state.http_realtime_lock:
        session = state.http_realtime_session
        matched = http_session_matches_audio(session, full_audio, language=language, prompt=prompt)
        if not matched:
            await cancel_http_session_corrections(state, session, reason="http-final-new-session")
            state.http_realtime_session = None
            state.http_preview_inflight = False

    if not matched:
        return await http_full_final_transcription(
            state,
            config,
            full_audio,
            language=language,
            prompt=prompt,
        )

    async with state.http_realtime_lock:
        session = state.http_realtime_session
        if session is None or not http_session_matches_audio(session, full_audio, language=language, prompt=prompt):
            return await http_full_final_transcription(
                state,
                config,
                full_audio,
                language=language,
                prompt=prompt,
            )
        session.final_requested = True
        update_http_session_audio(session, full_audio)
        await ws_update_rolling_correction(
            state,
            config,
            session.rolling_correction,
            full_audio,
            language=language,
            prompt=prompt,
            final=True,
        )
        text = http_session_merged_text(session)
        session.last_response_text = text
        state.http_preview_inflight = False
        state.last_used_at = time.monotonic()
        print(
            f"HTTP realtime final strategy=rolling-final-tail matched_session=True audio={total_sec:.2f}s "
            f"stable_until={session.rolling_correction.stable_until_sec:.2f}s chars={len(text)}",
            flush=True,
        )
        return {
            "text": text,
            "http_realtime": True,
            "http_final_strategy": "rolling-final-tail",
            "http_final_matched_session": True,
            "http_audio_sec": total_sec,
            "http_stable_until_sec": session.rolling_correction.stable_until_sec,
        }


async def http_full_final_transcription(
    state: State,
    config: Config,
    full_audio: np.ndarray,
    *,
    language: str | None,
    prompt: str,
) -> dict[str, Any]:
    response = await transcribe_full_in_correction_worker(
        state,
        config,
        full_audio,
        language=language,
        prompt=prompt,
    )
    text = str(response.get("text") or "")
    audio_sec = full_audio.size / ASR_SAMPLE_RATE if full_audio.size else 0.0
    print(
        f"HTTP realtime final strategy=full-one-shot matched_session=False audio={audio_sec:.2f}s "
        f"elapsed={float(response.get('elapsed_sec') or 0.0):.2f}s chars={len(text)}",
        flush=True,
    )
    return {
        "text": text,
        "http_realtime": True,
        "http_final_strategy": "full-one-shot",
        "http_final_matched_session": False,
        "http_audio_sec": audio_sec,
        "http_full_elapsed_sec": float(response.get("elapsed_sec") or 0.0),
    }


async def transcribe_full_in_correction_worker(
    state: State,
    config: Config,
    full_audio: np.ndarray,
    *,
    language: str | None,
    prompt: str,
) -> dict[str, Any]:
    if full_audio.size <= 0:
        return {"ok": True, "text": "", "elapsed_sec": 0.0}
    if state.ws_correction_lock is None:
        raise RuntimeError("Realtime correction runtime is not ready")

    async with state.ws_correction_lock:
        ensure_ws_correction_worker_started(state, config)
        if state.ws_correction_request_queue is None:
            raise RuntimeError("Realtime correction worker did not start")

        request_id = f"full_{time.time_ns()}_{uuid.uuid4().hex}"
        audio_sec = full_audio.size / ASR_SAMPLE_RATE
        state.ws_correction_request_queue.put(
            {
                "type": "full-transcribe",
                "id": request_id,
                "audio": full_audio.astype(np.float32, copy=False),
                "language": language,
                "prompt": prompt,
            }
        )
        print(f"HTTP full one-shot queued audio={audio_sec:.2f}s pending=1", flush=True)
        while True:
            ws_collect_correction_worker_responses(state)
            response = state.ws_correction_responses.pop(request_id, None)
            if response is not None:
                if not response.get("ok"):
                    raise RuntimeError(response.get("error") or "Full transcription failed")
                state.last_used_at = time.monotonic()
                return response
            if not is_ws_correction_worker_alive(state):
                raise RuntimeError("Realtime correction worker exited before returning full transcription")
            await asyncio.sleep(0.05)


def cancel_pending_ws_corrections(state: State, correction: WsRollingCorrectionSession, *, reason: str) -> None:
    if not correction.pending_request_ids:
        return
    pending = sorted(correction.pending_request_ids)
    print(
        f"Realtime WS correction cancel reason={reason} pending={len(pending)} "
        f"stable_until={correction.stable_until_sec:.2f}s",
        flush=True,
    )
    stop_ws_correction_worker(state, reason=reason)
    for request_id in pending:
        state.ws_correction_responses.pop(request_id, None)
        correction.ready_results.pop(request_id, None)
    correction.pending_request_ids.clear()
    correction.pending_requests.clear()
    correction.scheduled_until_sec = correction.stable_until_sec


def ws_schedule_final_catchup_corrections(
    state: State,
    config: Config,
    correction: WsRollingCorrectionSession,
    full_audio: np.ndarray,
    *,
    prefer_catchup_window: bool,
    pending_windows: list[dict[str, Any]],
    language: str | None,
    prompt: str,
) -> bool:
    total_sec = full_audio.size / ASR_SAMPLE_RATE
    commit_start_sec = correction.stable_until_sec
    if total_sec <= 0 or commit_start_sec >= total_sec - 0.05:
        return False

    if not prefer_catchup_window and total_sec - commit_start_sec <= ROLLING_CORRECTION_WINDOW_SEC + 0.05:
        window_start_sec = max(0.0, total_sec - ROLLING_CORRECTION_WINDOW_SEC)
        return ws_submit_correction_request(
            state,
            config,
            correction,
            full_audio,
            window_start_sec=window_start_sec,
            window_end_sec=total_sec,
            commit_start_sec=commit_start_sec,
            commit_end_sec=total_sec,
            reason="final-tail",
            language=language,
            prompt=prompt,
        )

    inherited_window_start_sec: float | None = None
    if prefer_catchup_window and pending_windows:
        candidate_starts = [
            float(item.get("window_start_sec") or 0.0)
            for item in pending_windows
            if float(item.get("commit_end_sec") or 0.0) > commit_start_sec + 0.05
        ]
        if candidate_starts:
            inherited_window_start_sec = max(0.0, min(candidate_starts))

    scheduled = False
    planned_start = commit_start_sec
    max_window_sec = max(ROLLING_CORRECTION_WINDOW_SEC, FINAL_CATCHUP_MAX_WINDOW_SEC)
    left_context_sec = max(0.0, FINAL_CATCHUP_LEFT_CONTEXT_SEC)
    right_context_sec = max(0.0, FINAL_CATCHUP_RIGHT_CONTEXT_SEC)
    while planned_start < total_sec - 0.05:
        if inherited_window_start_sec is not None:
            window_start_sec = inherited_window_start_sec
            inherited_window_start_sec = None
        else:
            window_start_sec = max(0.0, planned_start - left_context_sec)
        window_end_sec = min(total_sec, window_start_sec + max_window_sec)
        if window_end_sec >= total_sec - 0.05:
            commit_end_sec = total_sec
        else:
            commit_end_sec = max(planned_start, window_end_sec - right_context_sec)
        if commit_end_sec <= planned_start + 0.05:
            commit_end_sec = min(total_sec, planned_start + max(1.0, max_window_sec - left_context_sec - right_context_sec))
        if commit_end_sec <= planned_start + 0.05:
            break
        if ws_submit_correction_request(
            state,
            config,
            correction,
            full_audio,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            commit_start_sec=planned_start,
            commit_end_sec=commit_end_sec,
            reason="final-catchup" if commit_end_sec < total_sec - 0.05 else "final-tail",
            language=language,
            prompt=prompt,
        ):
            scheduled = True
        planned_start = commit_end_sec
    return scheduled


def ws_schedule_rolling_correction_windows(
    state: State,
    config: Config,
    correction: WsRollingCorrectionSession,
    full_audio: np.ndarray,
    *,
    language: str | None,
    prompt: str,
) -> bool:
    total_sec = full_audio.size / ASR_SAMPLE_RATE
    if total_sec <= 0:
        return False

    scheduled = False
    window_sec = ROLLING_CORRECTION_WINDOW_SEC
    stride_sec = ROLLING_CORRECTION_STRIDE_SEC
    commit_position_sec = ROLLING_CORRECTION_COMMIT_POSITION_SEC
    commit_span_sec = ROLLING_CORRECTION_COMMIT_SPAN_SEC

    while correction.next_window_start_sec + window_sec <= total_sec + 1e-6:
        window_start_sec = correction.next_window_start_sec
        window_end_sec = window_start_sec + window_sec
        if correction.scheduled_until_sec <= 0.02 and window_start_sec <= 0.02:
            commit_start_sec = 0.0
            commit_end_sec = min(window_end_sec, commit_position_sec + commit_span_sec)
        else:
            commit_start_sec = max(correction.scheduled_until_sec, window_start_sec + commit_position_sec)
            commit_end_sec = min(window_end_sec, window_start_sec + commit_position_sec + commit_span_sec)
        if ws_submit_correction_request(
            state,
            config,
            correction,
            full_audio,
            window_start_sec=window_start_sec,
            window_end_sec=window_end_sec,
            commit_start_sec=commit_start_sec,
            commit_end_sec=commit_end_sec,
            reason="rolling",
            language=language,
            prompt=prompt,
        ):
            scheduled = True
        correction.next_window_start_sec += stride_sec
    return scheduled


def ws_submit_correction_request(
    state: State,
    config: Config,
    correction: WsRollingCorrectionSession,
    full_audio: np.ndarray,
    *,
    window_start_sec: float,
    window_end_sec: float,
    commit_start_sec: float,
    commit_end_sec: float,
    reason: str,
    language: str | None,
    prompt: str,
) -> bool:
    if commit_end_sec <= commit_start_sec + 0.02:
        return False
    clip = audio_slice_seconds(full_audio, window_start_sec, window_end_sec, sample_rate=ASR_SAMPLE_RATE)
    if clip.size <= 0:
        return False
    ensure_ws_correction_worker_started(state, config)
    if state.ws_correction_request_queue is None:
        raise RuntimeError("Realtime correction worker did not start")

    request_id = f"corr_{time.time_ns()}_{uuid.uuid4().hex}"
    state.ws_correction_request_queue.put(
        {
            "type": "correction",
            "id": request_id,
            "session_id": correction.session_id,
            "reason": reason,
            "audio": clip.astype(np.float32, copy=False),
            "window_start_sec": window_start_sec,
            "window_end_sec": window_end_sec,
            "commit_start_sec": commit_start_sec,
            "commit_end_sec": commit_end_sec,
            "language": language,
            "prompt": prompt,
        }
    )
    correction.pending_request_ids.add(request_id)
    correction.pending_requests[request_id] = {
        "reason": reason,
        "window_start_sec": window_start_sec,
        "window_end_sec": window_end_sec,
        "commit_start_sec": commit_start_sec,
        "commit_end_sec": commit_end_sec,
    }
    correction.scheduled_until_sec = max(correction.scheduled_until_sec, commit_end_sec)
    print(
        f"Realtime WS correction queued reason={reason} window={window_start_sec:.2f}-{window_end_sec:.2f}s "
        f"commit={commit_start_sec:.2f}-{commit_end_sec:.2f}s pending={len(correction.pending_request_ids)}",
        flush=True,
    )
    return True


async def ws_drain_correction_results(
    state: State,
    correction: WsRollingCorrectionSession,
    *,
    wait_for_pending: bool,
) -> bool:
    updated = False
    while True:
        ws_collect_correction_worker_responses(state)
        updated = ws_apply_correction_results(state, correction) or updated
        if not wait_for_pending or not correction.pending_request_ids:
            return updated
        if not is_ws_correction_worker_alive(state):
            raise RuntimeError("Realtime correction worker exited before returning a result")
        await asyncio.sleep(0.05)


def ws_collect_correction_worker_responses(state: State) -> None:
    response_queue = state.ws_correction_response_queue
    if response_queue is None:
        return
    while True:
        try:
            response = response_queue.get_nowait()
        except queue.Empty:
            return
        response_id = str(response.get("id") or "")
        if response_id:
            state.ws_correction_responses[response_id] = response


def ws_apply_correction_results(state: State, correction: WsRollingCorrectionSession) -> bool:
    for request_id in list(correction.pending_request_ids):
        response = state.ws_correction_responses.pop(request_id, None)
        if response is None:
            continue
        correction.pending_request_ids.remove(request_id)
        correction.pending_requests.pop(request_id, None)
        correction.ready_results[request_id] = response

    updated = False
    while correction.ready_results:
        request_id, response = min(
            correction.ready_results.items(),
            key=lambda item: (float(item[1].get("commit_start_sec") or 0.0), float(item[1].get("commit_end_sec") or 0.0)),
        )
        commit_start_sec = float(response.get("commit_start_sec") or 0.0)
        commit_end_sec = float(response.get("commit_end_sec") or commit_start_sec)
        if commit_start_sec > correction.stable_until_sec + 0.05:
            break
        correction.ready_results.pop(request_id, None)

        reason = str(response.get("reason") or "correction")
        window_start_sec = float(response.get("window_start_sec") or 0.0)
        window_end_sec = float(response.get("window_end_sec") or 0.0)
        elapsed = float(response.get("elapsed_sec") or 0.0)
        if not response.get("ok"):
            print(
                f"Realtime WS correction failed reason={reason} window={window_start_sec:.2f}-{window_end_sec:.2f}s "
                f"commit={commit_start_sec:.2f}-{commit_end_sec:.2f}s error={response.get('error')!r}",
                flush=True,
            )
            continue

        piece = str(response.get("piece") or "")
        if piece:
            correction.stable_text = append_corrected_text(correction.stable_text, piece)
        else:
            print(
                f"Realtime WS correction empty reason={reason} window={window_start_sec:.2f}-{window_end_sec:.2f}s "
                f"commit={commit_start_sec:.2f}-{commit_end_sec:.2f}s elapsed={elapsed:.2f}s",
                flush=True,
            )
        correction.stable_until_sec = max(correction.stable_until_sec, commit_end_sec)
        updated = True
        print(
            f"Realtime WS correction applied reason={reason} window={window_start_sec:.2f}-{window_end_sec:.2f}s "
            f"commit={commit_start_sec:.2f}-{commit_end_sec:.2f}s elapsed={elapsed:.2f}s "
            f"piece_chars={len(piece)} stable_until={correction.stable_until_sec:.2f}s stable_chars={len(correction.stable_text)}",
            flush=True,
        )
    return updated


def ws_update_chunk_preview_sync(
    state: State,
    config: Config,
    full_audio: np.ndarray,
    language: str | None,
    prompt: str,
) -> WsChunkPreviewState:
    total_sec = full_audio.size / ASR_SAMPLE_RATE if full_audio.size else 0.0
    if total_sec <= 0:
        return WsChunkPreviewState()
    window_start_sec = max(0.0, total_sec - CHUNK_PREVIEW_WINDOW_SEC)
    clip = audio_slice_seconds(full_audio, window_start_sec, total_sec, sample_rate=ASR_SAMPLE_RATE)
    started = time.perf_counter()
    result = get_ws_session_sync(state, config).transcribe(
        clip,
        context=prompt,
        language=language,
        return_timestamps=False,
        return_chunks=False,
    )
    elapsed = time.perf_counter() - started
    text = result.text or ""
    print(
        f"Realtime WS chunk preview window={window_start_sec:.2f}-{total_sec:.2f}s "
        f"elapsed={elapsed:.2f}s chars={len(text)}",
        flush=True,
    )
    return WsChunkPreviewState(text=text, window_start_sec=window_start_sec, window_end_sec=total_sec)


def get_ws_session_sync(state: State, config: Config):
    if state.ws_session is not None:
        return state.ws_session
    from mlx_qwen3_asr.session import Session

    dtype = {"float16": mx.float16, "float32": mx.float32, "bfloat16": mx.bfloat16}.get(config.dtype, mx.float16)
    load_started = time.perf_counter()
    print(f"Loading realtime ASR model from {config.preview_model_path} dtype={config.dtype}", flush=True)
    state.ws_session = Session(config.preview_model_path, dtype=dtype)
    print(f"Realtime ASR model loaded elapsed={time.perf_counter() - load_started:.2f}s", flush=True)
    return state.ws_session


def corrected_text_span(raw_text: str, segments: list[dict[str, Any]], start_sec: float, end_sec: float) -> str:
    if not raw_text:
        return ""
    if not segments:
        return raw_text.strip() if start_sec <= 0.02 else ""

    norm_start = 0
    norm_end = 0
    for segment in segments:
        text = str(segment.get("text") or "")
        seg_start = float(segment.get("start") or 0.0)
        seg_end = float(segment.get("end") or seg_start)
        midpoint = (seg_start + seg_end) / 2.0
        length = normalized_text_len(text)
        if midpoint < start_sec - 1e-6:
            norm_start += length
        if midpoint < end_sec - 1e-6:
            norm_end += length
    if norm_end <= norm_start:
        return ""
    return raw_span_by_normalized_counts(raw_text, norm_start, norm_end)


def normalized_text_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace() and ch not in TEXT_BOUNDARY_PUNCT)


def normalized_text(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace() and ch not in TEXT_BOUNDARY_PUNCT)


def raw_span_by_normalized_counts(raw_text: str, start_count: int, end_count: int) -> str:
    current = 0
    raw_start: int | None = None
    raw_end: int | None = None
    for index, ch in enumerate(raw_text):
        if ch.isspace() or ch in TEXT_BOUNDARY_PUNCT:
            continue
        if raw_start is None and current >= start_count:
            raw_start = index
        current += 1
        if current >= end_count:
            raw_end = index + 1
            break
    if raw_start is None:
        return ""
    if raw_end is None:
        raw_end = len(raw_text)
    while raw_end < len(raw_text) and (raw_text[raw_end].isspace() or raw_text[raw_end] in TEXT_BOUNDARY_PUNCT):
        raw_end += 1
    return raw_text[raw_start:raw_end].strip()


def append_corrected_text(stable_text: str, piece: str) -> str:
    piece = piece.strip()
    if not piece:
        return stable_text
    if not stable_text:
        return piece
    return stable_text + piece


def merge_stable_and_live_window(
    stable_text: str,
    live_text: str,
    live_start_sec: float,
    live_end_sec: float,
    stable_until_sec: float,
) -> str:
    if not stable_text:
        return live_text
    if not live_text:
        return stable_text
    if live_end_sec <= live_start_sec + 0.02:
        return stable_text + live_text

    # Prefer text overlap when the 0.6B live chunk and 1.7B stable prefix agree.
    stable_norm = normalized_text(stable_text)
    live_norm = normalized_text(live_text)
    max_overlap = min(len(stable_norm), len(live_norm), 120)
    for count in range(max_overlap, 5, -1):
        if stable_norm.endswith(live_norm[:count]):
            return stable_text + live_text[raw_cut_after_normalized_count(live_text, count) :]

    # Fallback to audio-time proportional cutting. The live chunk is only a
    # temporary preview tail, so avoiding visible duplicate text matters more
    # than perfectly cutting every character.
    if stable_until_sec <= live_start_sec + 0.05:
        return stable_text + live_text
    if stable_until_sec >= live_end_sec - 0.05:
        return stable_text
    overlap_ratio = (stable_until_sec - live_start_sec) / (live_end_sec - live_start_sec)
    drop_count = int(round(max(0.0, min(1.0, overlap_ratio)) * normalized_text_len(live_text)))
    return stable_text + live_text[raw_cut_after_normalized_count(live_text, drop_count) :]


def raw_cut_after_normalized_count(text: str, count: int) -> int:
    if count <= 0:
        return 0
    seen = 0
    cut = len(text)
    for index, ch in enumerate(text):
        if ch.isspace() or ch in TEXT_BOUNDARY_PUNCT:
            continue
        seen += 1
        if seen >= count:
            cut = index + 1
            break
    while cut < len(text) and (text[cut].isspace() or text[cut] in TEXT_BOUNDARY_PUNCT):
        cut += 1
    return cut


async def unload_idle_ws_session_loop(state: State, config: Config) -> None:
    while True:
        await asyncio.sleep(min(max(config.unload_after_sec / 2, 1.0), 10.0))
        if (state.ws_session is None and not ws_correction_worker_alive(state)) or state.active_ws_count > 0 or state.active_http_count > 0 or state.last_used_at <= 0:
            continue
        idle_for = time.monotonic() - state.last_used_at
        if idle_for >= config.unload_after_sec:
            await unload_ws_session(state, config, reason=f"{config.unload_after_sec}s idle")


async def unload_ws_session(state: State, config: Config, *, reason: str, force: bool = False) -> None:
    if state.ws_lock is None or state.ws_executor is None:
        return
    async with state.ws_lock:
        if (state.ws_session is None and not ws_correction_worker_alive(state)) or state.active_ws_count > 0 or state.active_http_count > 0:
            return
        if not force and config.unload_after_sec > 0 and state.last_used_at > 0:
            idle_for = time.monotonic() - state.last_used_at
            if idle_for < config.unload_after_sec:
                return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(state.ws_executor, unload_ws_session_sync, state, reason)
        stop_ws_correction_worker(state, reason=reason)


def unload_ws_session_sync(state: State, reason: str) -> None:
    if state.ws_session is not None:
        print(f"Unloading realtime ASR model after {reason}", flush=True)
        state.ws_session = None
    state.http_realtime_session = None
    state.http_preview_inflight = False
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


def model_info(model_id: str) -> dict:
    return {"id": model_id, "object": "model", "created": 0, "owned_by": "local"}


def ws_correction_worker_alive(state: State) -> bool:
    process = state.ws_correction_process
    return bool(process and process.is_alive())


def is_ws_correction_worker_alive(state: State) -> bool:
    process = state.ws_correction_process
    if process is None:
        return False
    if process.is_alive():
        return True
    remove_pidfile(CORRECTION_WORKER_PIDFILE, process.pid)
    state.ws_correction_process = None
    state.ws_correction_request_queue = None
    state.ws_correction_response_queue = None
    return False


def ensure_ws_correction_worker_started(state: State, config: Config) -> None:
    if is_ws_correction_worker_alive(state) and state.ws_correction_request_queue is not None and state.ws_correction_response_queue is not None:
        return
    cleanup_stale_pidfile(CORRECTION_WORKER_PIDFILE, label="Realtime correction worker")
    state.ws_correction_request_queue = None
    state.ws_correction_response_queue = None
    state.ws_correction_process = None

    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()
    process = ctx.Process(
        target=ws_correction_worker_main,
        args=(config, request_queue, response_queue),
        name="asrd-correction-worker",
    )
    process.daemon = True
    process.start()
    write_pidfile(CORRECTION_WORKER_PIDFILE, process.pid)
    state.ws_correction_request_queue = request_queue
    state.ws_correction_response_queue = response_queue
    state.ws_correction_process = process
    print(f"Started realtime correction worker pid={process.pid}", flush=True)


def stop_ws_correction_worker(state: State, *, reason: str) -> None:
    process = state.ws_correction_process
    if process is None:
        return
    print(f"Stopping realtime correction worker after {reason} pid={process.pid}", flush=True)
    if state.ws_correction_request_queue is not None:
        try:
            state.ws_correction_request_queue.put({"type": "shutdown"})
        except Exception:
            pass
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    remove_pidfile(CORRECTION_WORKER_PIDFILE, process.pid)
    state.ws_correction_process = None
    state.ws_correction_request_queue = None
    state.ws_correction_response_queue = None


def cleanup_stale_pidfile(pidfile: Path, *, label: str) -> None:
    try:
        pid = int(pidfile.read_text().strip())
    except Exception:
        return
    if pid <= 0 or pid == os.getpid():
        remove_pidfile(pidfile, pid)
        return
    try:
        os.kill(pid, 0)
    except OSError:
        remove_pidfile(pidfile, pid)
        return
    print(f"Terminating stale {label} pid={pid}", flush=True)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            remove_pidfile(pidfile, pid)
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    remove_pidfile(pidfile, pid)


def write_pidfile(pidfile: Path, pid: int | None) -> None:
    if not pid:
        return
    try:
        pidfile.write_text(f"{pid}\n", encoding="utf-8")
    except OSError:
        pass


def remove_pidfile(pidfile: Path, pid: int | None = None) -> None:
    try:
        if pid is not None and pidfile.exists():
            current = int(pidfile.read_text().strip())
            if current != pid:
                return
        pidfile.unlink(missing_ok=True)
    except Exception:
        pass


def ws_correction_worker_main(config: Config, request_queue: mp.Queue, response_queue: mp.Queue) -> None:
    from mlx_qwen3_asr.forced_aligner import ForcedAligner
    from mlx_qwen3_asr.session import Session

    correction_session = None
    forced_aligner = None
    dtype = {"float16": mx.float16, "float32": mx.float32, "bfloat16": mx.bfloat16}.get(config.dtype, mx.float16)
    timeout = config.unload_after_sec if config.unload_after_sec > 0 else None

    def clear_mlx_cache() -> None:
        for clear in (mx.clear_cache, getattr(mx, "metal", None) and mx.metal.clear_cache):
            if clear:
                try:
                    clear()
                except Exception:
                    pass

    def get_correction_session():
        nonlocal correction_session
        if correction_session is not None:
            return correction_session
        load_started = time.perf_counter()
        print(f"Loading realtime correction ASR model from {config.model_path} dtype={config.dtype}", flush=True)
        correction_session = Session(config.model_path, dtype=dtype)
        print(f"Realtime correction ASR model loaded elapsed={time.perf_counter() - load_started:.2f}s", flush=True)
        return correction_session

    def get_forced_aligner():
        nonlocal forced_aligner
        if forced_aligner is not None:
            return forced_aligner
        load_started = time.perf_counter()
        print(f"Loading realtime correction forced aligner dtype={config.dtype}", flush=True)
        forced_aligner = ForcedAligner(dtype=dtype)
        print(f"Realtime correction forced aligner ready elapsed={time.perf_counter() - load_started:.2f}s", flush=True)
        return forced_aligner

    try:
        while True:
            try:
                message = request_queue.get(timeout=timeout)
            except queue.Empty:
                print(f"Realtime correction worker exiting after {config.unload_after_sec}s idle", flush=True)
                break

            if message.get("type") == "shutdown":
                print("Realtime correction worker exiting after shutdown", flush=True)
                break
            if message.get("type") == "full-transcribe":
                request_id = message.get("id")
                started = time.perf_counter()
                try:
                    audio = message["audio"]
                    result = get_correction_session().transcribe(
                        audio,
                        context=message.get("prompt") or "",
                        language=message.get("language"),
                        return_timestamps=False,
                        return_chunks=False,
                        )
                    response_queue.put(
                        {
                            "id": request_id,
                            "ok": True,
                            "text": result.text or "",
                            "elapsed_sec": time.perf_counter() - started,
                            "audio_sec": len(audio) / ASR_SAMPLE_RATE if hasattr(audio, "__len__") else 0.0,
                            "reason": "full-one-shot",
                        }
                    )
                except Exception as exc:
                    traceback.print_exc()
                    response_queue.put(
                        {
                            "id": request_id,
                            "ok": False,
                            "error": str(exc),
                            "elapsed_sec": time.perf_counter() - started,
                            "reason": "full-one-shot",
                        }
                    )
                continue
            if message.get("type") != "correction":
                continue

            request_id = message.get("id")
            started = time.perf_counter()
            try:
                clip = message["audio"]
                result = get_correction_session().transcribe(
                    clip,
                    context=message.get("prompt") or "",
                    language=message.get("language"),
                    return_timestamps=True,
                    return_chunks=True,
                    forced_aligner=get_forced_aligner(),
                )
                window_start_sec = float(message.get("window_start_sec") or 0.0)
                commit_start_sec = float(message.get("commit_start_sec") or 0.0)
                commit_end_sec = float(message.get("commit_end_sec") or commit_start_sec)
                piece = corrected_text_span(
                    result.text or "",
                    result.segments or [],
                    commit_start_sec - window_start_sec,
                    commit_end_sec - window_start_sec,
                )
                response_queue.put(
                    {
                        "id": request_id,
                        "session_id": message.get("session_id"),
                        "ok": True,
                        "piece": piece,
                        "elapsed_sec": time.perf_counter() - started,
                        "reason": message.get("reason"),
                        "window_start_sec": window_start_sec,
                        "window_end_sec": float(message.get("window_end_sec") or 0.0),
                        "commit_start_sec": commit_start_sec,
                        "commit_end_sec": commit_end_sec,
                        "raw_text_chars": len(result.text or ""),
                        "piece_chars": len(piece),
                    }
                )
            except Exception as exc:
                traceback.print_exc()
                response_queue.put(
                    {
                        "id": request_id,
                        "session_id": message.get("session_id"),
                        "ok": False,
                        "error": str(exc),
                        "elapsed_sec": time.perf_counter() - started,
                        "reason": message.get("reason"),
                        "window_start_sec": float(message.get("window_start_sec") or 0.0),
                        "window_end_sec": float(message.get("window_end_sec") or 0.0),
                        "commit_start_sec": float(message.get("commit_start_sec") or 0.0),
                        "commit_end_sec": float(message.get("commit_end_sec") or 0.0),
                    }
                )
    finally:
        forced_aligner = None
        correction_session = None
        clear_mlx_cache()


def check_auth(request: Request, api_key: str) -> None:
    if request.headers.get("authorization", "") != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def require_non_empty_model(model: Optional[str]) -> str:
    value = (model or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="model must be a non-empty string")
    return value


def normalize_language(language: Optional[str]) -> Optional[str]:
    if not language or not language.strip():
        return None
    value = language.strip()
    return LANGUAGE_ALIASES.get(value.lower().replace("_", "-"), value)


def first_text_value(*values: Any) -> str:
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return ""


def asr_context_terms_from_sources(*sources: Any) -> list[str]:
    terms: list[str] = []
    for source in sources:
        terms.extend(extract_asr_context_terms(source))
    return dedupe_asr_context_terms(terms)


def extract_asr_context_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return split_asr_context_terms(value)
    if isinstance(value, (list, tuple, set)):
        terms: list[str] = []
        for item in value:
            terms.extend(extract_asr_context_terms(item))
        return terms
    if isinstance(value, dict):
        terms: list[str] = []
        for key in ASR_CONTEXT_TERM_KEYS:
            if key in value:
                terms.extend(extract_asr_context_terms(value.get(key)))
        if terms:
            return terms
        for key in ASR_CONTEXT_ITEM_KEYS:
            item = value.get(key)
            if isinstance(item, str):
                return split_asr_context_terms(item)
        return []
    return []


def split_asr_context_terms(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[\n,;；，、]+", text) if part.strip()]


def dedupe_asr_context_terms(terms: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    for term in terms:
        clean = re.sub(r"\s+", " ", term).strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        if len(result) >= ASR_CONTEXT_MAX_TERMS:
            break
        if total_chars + len(clean) > ASR_CONTEXT_MAX_CHARS:
            break
        seen.add(key)
        result.append(clean)
        total_chars += len(clean) + 1
    return result


def build_asr_context_prompt(base_prompt: str, terms: list[str]) -> tuple[str, int]:
    clean_prompt = (base_prompt or "").strip()
    clean_terms = dedupe_asr_context_terms(terms)
    if not clean_terms:
        return clean_prompt, 0
    terms_text = "\n".join(clean_terms)
    if not clean_prompt:
        return terms_text, len(clean_terms)
    return f"{clean_prompt}\n{terms_text}", len(clean_terms)


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


def print_efficiency(audio_duration: float | None, elapsed: float, text_chars: int, *, is_preview: bool, metadata: dict[str, Any] | None = None) -> None:
    kind = "preview" if is_preview else "final"
    if metadata and metadata.get("http_realtime"):
        audio_sec = float(metadata.get("http_audio_sec") or audio_duration or 0.0)
        stable_until = metadata.get("http_stable_until_sec")
        cached = metadata.get("http_preview_cached")
        strategy = metadata.get("http_final_strategy") or metadata.get("http_preview_skip_reason")
        extra = f" strategy={strategy}" if strategy else ""
        if stable_until is not None:
            extra += f" stable_until={float(stable_until):.2f}s"
        if is_preview:
            extra += f" cached={bool(cached)}"
        if audio_sec > 0 and elapsed > 0:
            speed = audio_sec / elapsed
            rtf = elapsed / audio_sec
            print(
                f"Transcription done kind={kind} http_realtime=1 audio={audio_sec:.2f}s elapsed={elapsed:.2f}s "
                f"speed={speed:.2f}x rtf={rtf:.3f} chars={text_chars}{extra}",
                flush=True,
            )
        else:
            print(f"Transcription done kind={kind} http_realtime=1 elapsed={elapsed:.2f}s chars={text_chars}{extra}", flush=True)
        return
    if audio_duration and audio_duration > 0 and elapsed > 0:
        speed = audio_duration / elapsed
        rtf = elapsed / audio_duration
        print(f"Transcription done kind={kind} audio={audio_duration:.2f}s elapsed={elapsed:.2f}s speed={speed:.2f}x rtf={rtf:.3f} chars={text_chars}", flush=True)
    else:
        print(f"Transcription done kind={kind} elapsed={elapsed:.2f}s chars={text_chars}", flush=True)


def debug_artifact_audio_path(config: Config, *, kind: str, safe_name: str, suffix: str) -> Path:
    target_dir = Path(config.debug_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    clean_name = safe_debug_filename(safe_name)
    stem = Path(clean_name).stem or kind
    clean_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    for _ in range(8):
        target = target_dir / f"{timestamp}-{kind}-{uuid.uuid4().hex[:12]}-{stem}{clean_suffix}"
        if not target.exists() and not target.with_suffix(".json").exists():
            return target
    return target_dir / f"{timestamp}-{kind}-{uuid.uuid4().hex}-{stem}{clean_suffix}"


def preserve_debug_final_audio(config: Config, audio_path: str, filename: str | None) -> Path | None:
    try:
        safe_name = safe_debug_filename(filename or "audio.wav")
        suffix = Path(safe_name).suffix or Path(audio_path).suffix or ".wav"
        target = debug_artifact_audio_path(config, kind="http", safe_name=safe_name, suffix=suffix)
        shutil.copy2(audio_path, target)
        return target
    except Exception as exc:
        print(f"WARNING: failed to preserve debug final audio: {exc}", file=sys.stderr, flush=True)
        return None


def preserve_debug_ws_audio(
    config: Config,
    audio: np.ndarray,
    *,
    client: str,
    protocol: str,
    request_model: str,
    preview_model: str,
    final_model: str,
    language: str | None,
    chunks: list[list[float]],
    final_text: str = "",
) -> Path | None:
    if audio.size <= 0:
        return None
    try:
        target = debug_artifact_audio_path(config, kind="ws", safe_name=client.replace(":", "-"), suffix=".wav")
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(str(target), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(ASR_SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        metadata = {
            "protocol": protocol,
            "request_model": request_model,
            "preview_model": preview_model,
            "final_model": final_model,
            "language": language,
            "duration": audio.size / ASR_SAMPLE_RATE,
            "text": final_text,
            "chunks": chunks,
        }
        json_path = target.with_suffix(".json")
        json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"DEBUG artifact saved kind=ws audio={str(target)!r} json={str(json_path)!r}", flush=True)
        return target
    except Exception as exc:
        print(f"WARNING: failed to preserve debug realtime audio: {exc}", file=sys.stderr, flush=True)
        return None


def write_debug_transcription_metadata(
    audio_path: Path,
    *,
    protocol: str,
    request_model: str,
    preview_model: str,
    final_model: str,
    language: str | None,
    audio_duration: float | None,
    audio_samples: int,
    text: str,
) -> None:
    try:
        metadata_path = audio_path.with_suffix(".json")
        metadata = {
            "protocol": protocol,
            "request_model": request_model,
            "preview_model": preview_model,
            "final_model": final_model,
            "language": language,
            "duration": audio_duration,
            "text": text,
            "chunks": [[0, 0, audio_samples, 0.0]] if audio_samples > 0 else [],
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"DEBUG artifact saved kind=http audio={str(audio_path)!r} json={str(metadata_path)!r}", flush=True)
    except Exception as exc:
        print(f"WARNING: failed to write debug final metadata: {exc}", file=sys.stderr, flush=True)


def log_debug_transcription_text(*, is_preview: bool, text: str) -> None:
    return None


def log_debug_realtime_text(*, protocol: str, text: str, sentence_end: bool, delta: str | None = None) -> None:
    return None


def print_startup_summary(args: argparse.Namespace, model_path: str, preview_model_path: str) -> None:
    print(
        f"""
Starting asrd

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
  Debug:             {'on' if args.debug else 'off'}
  Debug dir:         {args.debug_dir if args.debug else '-'}

VoxT Remote ASR:
  Provider:          OpenAI Transcribe
  Endpoint:          http://<host-ip>:{args.port}/v1/audio/transcriptions
  API Key:           {masked_api_key(args.api_key)}
  Model:             {args.model}
  Preview:           0.6B last-{CHUNK_PREVIEW_WINDOW_SEC:.0f}s chunk, cadence={HTTP_CHUNK_PREVIEW_UPDATE_SEC:.1f}s
  Final:             chunk cache -> bounded final-tail/catch-up; no chunk -> full one-shot

VoxT Realtime ASR:
  Provider:          Aliyun Bailian ASR
  Endpoint:          ws://<host-ip>:{args.port}/v1/realtime
  API Key:           {masked_api_key(args.api_key)}
  Model:             qwen3-asr-flash-realtime
  Realtime model:    {args.preview_model}
  Realtime mode:     fixed chunk preview, W={CHUNK_PREVIEW_WINDOW_SEC:.0f}s update={WS_CHUNK_PREVIEW_UPDATE_SEC:.0f}s
  Correction model:  {args.model}
  Audio:             PCM16 mono 16k
  Rolling correction: on, separate worker, W={ROLLING_CORRECTION_WINDOW_SEC:.0f}s stride={ROLLING_CORRECTION_STRIDE_SEC:.0f}s P={ROLLING_CORRECTION_COMMIT_POSITION_SEC:.0f}s span={ROLLING_CORRECTION_COMMIT_SPAN_SEC:.0f}s final-catchup-max={FINAL_CATCHUP_MAX_WINDOW_SEC:.0f}s
  Unload after:      {args.unload_after_sec}s idle (0 = keep loaded)
""".strip(),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
