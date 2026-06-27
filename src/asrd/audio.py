from __future__ import annotations

import base64
import shutil
import subprocess
import wave

import numpy as np

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_AUDIO_DIFF_THRESHOLD = 0.02


def full_audio_from_parts(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def audio_diff_ok(reference: np.ndarray, candidate: np.ndarray, threshold: float = DEFAULT_AUDIO_DIFF_THRESHOLD) -> bool:
    count = min(reference.size, candidate.size)
    if count <= 0 or reference.size != candidate.size:
        return False
    return float(np.mean(np.abs(reference[:count] - candidate[:count]))) <= threshold


def audio_slice_seconds(audio: np.ndarray, start_sec: float, end_sec: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    start = max(0, int(round(start_sec * sample_rate)))
    end = min(audio.size, int(round(end_sec * sample_rate)))
    if end <= start:
        return np.array([], dtype=np.float32)
    return audio[start:end].copy()


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


def is_voxt_preview_upload(filename: str | None) -> bool:
    return bool(filename and filename.startswith("voxt-openai-preview-"))


def audio_duration_seconds(path: str) -> float | None:
    return wav_duration_seconds(path) or ffprobe_duration_seconds(path)


def decode_audio_float32(path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        return decode_wav_float32(path, sample_rate=sample_rate)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-ar", str(sample_rate), "-ac", "1", "-f", "f32le", "pipe:1"]
    timeout = max(15.0, (audio_duration_seconds(path) or 1.0) + 10.0)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)
    if not result.stdout:
        return np.array([], dtype=np.float32)
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def decode_wav_float32(path: str, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
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
            check=False,
        )
        value = float(result.stdout.strip())
        return value if value > 0 else None
    except Exception:
        return None
