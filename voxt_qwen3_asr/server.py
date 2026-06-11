from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from mlx_qwen3_asr.session import Session


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

DTYPES = {
    "float16": mx.float16,
    "float32": mx.float32,
    "bfloat16": mx.bfloat16,
}


@dataclass(frozen=True)
class ServerConfig:
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
    preview_max_new_tokens: int | None = None
    final_max_new_tokens: int | None = None


@dataclass
class ServerState:
    final_session: Session | None = None
    preview_session: Session | None = None
    inference_lock: asyncio.Lock | None = None
    last_used_at: float = 0.0
    unload_task: asyncio.Task | None = None


def create_app(config: ServerConfig) -> FastAPI:
    state = ServerState(inference_lock=asyncio.Lock())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Models are loaded lazily on first request and unloaded after an idle
        # period, so starting the server does not immediately consume model RAM.
        if config.unload_after_sec > 0:
            state.unload_task = asyncio.create_task(unload_idle_models_loop(state, config))
        yield
        if state.unload_task is not None:
            state.unload_task.cancel()
            try:
                await state.unload_task
            except asyncio.CancelledError:
                pass
        unload_models(state, reason="shutdown")

    app = FastAPI(title="voxt-qwen3-asr", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "model": config.model,
            "model_path": config.model_path,
            "preview_model": config.preview_model,
            "preview_model_path": config.preview_model_path,
            "models_loaded": state.final_session is not None,
            "unload_after_sec": config.unload_after_sec,
            "dtype": config.dtype,
        }

    @app.get("/v1/models")
    async def models(request: Request) -> dict:
        check_auth(request, config.api_key)
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                },
                {
                    "id": config.preview_model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

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

        suffix = Path(file.filename or "upload.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="voxt_asr_")
        tmp_path = tmp.name
        try:
            size = await write_upload_to_temp(file, tmp, config.max_file_size_mb)
            lang = normalize_language(language)
            fmt = (response_format or "json").strip().lower()
            if fmt not in {"json", "text", "verbose_json"}:
                raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")

            audio_duration = audio_duration_seconds(tmp_path)
            is_preview = is_voxt_preview_upload(file.filename)
            max_new_tokens = config.preview_max_new_tokens if is_preview else config.final_max_new_tokens
            selected_model = config.preview_model if is_preview else config.model
            print(
                "Transcription request "
                f"kind={'preview' if is_preview else 'final'} filename={file.filename!r} bytes={size} "
                f"request_model={model!r} selected_model={selected_model!r} "
                f"language={language!r}->{lang!r} prompt_chars={len(prompt or '')} format={fmt!r}",
                flush=True,
            )
            started = time.perf_counter()
            # MLX GPU streams are thread-local on this setup, so do not use
            # Session.transcribe_async() here. Serialize inference on the server
            # event-loop thread instead.
            async with state.inference_lock:
                selected_session = ensure_session_loaded(state, config, preview=is_preview)
                state.last_used_at = time.monotonic()
                result = selected_session.transcribe(
                    tmp_path,
                    language=lang,
                    context=prompt or "",
                    return_timestamps=False,
                    return_chunks=False,
                    max_new_tokens=max_new_tokens,
                )
                state.last_used_at = time.monotonic()
            elapsed = time.perf_counter() - started
            text = (result.text or "").strip()
            print_efficiency(audio_duration, elapsed, len(text), is_preview=is_preview, max_new_tokens=max_new_tokens)

            if fmt == "text":
                return PlainTextResponse(text)
            if fmt == "verbose_json":
                return JSONResponse(
                    {
                        "text": text,
                        "task": "transcribe",
                        "language": lang,
                        "duration": audio_duration,
                        "segments": [],
                    }
                )
            return JSONResponse({"text": text})
        except HTTPException:
            raise
        except Exception as exc:
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(exc), "type": type(exc).__name__}},
            )
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    return app


def ensure_session_loaded(state: ServerState, config: ServerConfig, *, preview: bool) -> Session:
    if preview:
        return ensure_preview_session_loaded(state, config)
    return ensure_final_session_loaded(state, config)


def ensure_preview_session_loaded(state: ServerState, config: ServerConfig) -> Session:
    if state.preview_session is not None:
        return state.preview_session
    if config.preview_model_path == config.model_path and state.final_session is not None:
        state.preview_session = state.final_session
        return state.preview_session

    dtype = DTYPES.get(config.dtype, mx.float16)
    print(f"Loading preview ASR model from {config.preview_model_path} dtype={config.dtype}", flush=True)
    state.preview_session = Session(config.preview_model_path, dtype=dtype)
    state.last_used_at = time.monotonic()
    print("Preview ASR model loaded", flush=True)
    return state.preview_session


def ensure_final_session_loaded(state: ServerState, config: ServerConfig) -> Session:
    if state.final_session is not None:
        return state.final_session
    if config.preview_model_path == config.model_path and state.preview_session is not None:
        state.final_session = state.preview_session
        return state.final_session

    dtype = DTYPES.get(config.dtype, mx.float16)
    print(f"Loading final ASR model from {config.model_path} dtype={config.dtype}", flush=True)
    state.final_session = Session(config.model_path, dtype=dtype)
    state.last_used_at = time.monotonic()
    print("Final ASR model loaded", flush=True)
    return state.final_session


def unload_models(state: ServerState, *, reason: str = "idle timeout") -> None:
    if state.final_session is None and state.preview_session is None:
        return
    print(f"Unloading ASR models after {reason}", flush=True)
    state.preview_session = None
    state.final_session = None
    try:
        mx.clear_cache()
    except Exception:
        pass
    try:
        mx.metal.clear_cache()
    except Exception:
        pass


async def unload_idle_models_loop(state: ServerState, config: ServerConfig) -> None:
    while True:
        await asyncio.sleep(min(max(config.unload_after_sec / 2, 5.0), 60.0))
        if state.inference_lock is None:
            continue
        async with state.inference_lock:
            if state.final_session is None:
                continue
            idle_for = time.monotonic() - state.last_used_at
            if idle_for >= config.unload_after_sec:
                unload_models(state, reason="idle timeout")


def check_auth(request: Request, api_key: str) -> None:
    if request.headers.get("authorization", "") != f"Bearer {api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def normalize_language(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    value = language.strip()
    if not value:
        return None
    return LANGUAGE_ALIASES.get(value.lower().replace("_", "-"), value)


async def write_upload_to_temp(file: UploadFile, tmp, max_file_size_mb: int) -> int:
    max_bytes = max_file_size_mb * 1024 * 1024
    size = 0
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
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


def audio_duration_seconds(path: str) -> float | None:
    wav_duration = wav_duration_seconds(path)
    if wav_duration is not None:
        return wav_duration
    return ffprobe_duration_seconds(path)


def wav_duration_seconds(path: str) -> float | None:
    try:
        import wave

        with wave.open(path, "rb") as wav:
            rate = wav.getframerate()
            frames = wav.getnframes()
            if rate > 0:
                return frames / rate
    except Exception:
        return None
    return None


def ffprobe_duration_seconds(path: str) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
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


def print_efficiency(
    audio_duration: float | None,
    elapsed: float,
    text_chars: int,
    *,
    is_preview: bool,
    max_new_tokens: int | None,
) -> None:
    kind = "preview" if is_preview else "final"
    token_cap = f" max_new_tokens={max_new_tokens}" if max_new_tokens is not None else ""
    if audio_duration and audio_duration > 0 and elapsed > 0:
        speed = audio_duration / elapsed
        rtf = elapsed / audio_duration
        print(
            f"Transcription done kind={kind} audio={audio_duration:.2f}s elapsed={elapsed:.2f}s "
            f"speed={speed:.2f}x rtf={rtf:.3f} chars={text_chars}{token_cap}",
            flush=True,
        )
        return
    print(f"Transcription done kind={kind} elapsed={elapsed:.2f}s chars={text_chars}{token_cap}", flush=True)
