from __future__ import annotations

import json
import wave

import numpy as np
import pytest
from fastapi import HTTPException

from asrd import server


def test_parse_args_defaults_and_overrides():
    args = server.parse_args(["--port", "5101", "--debug", "--unload-after-sec", "0"])

    assert args.host == "127.0.0.1"
    assert args.port == 5101
    assert args.debug is True
    assert args.unload_after_sec == 0


def test_parse_args_defaults_without_environment_configuration():
    args = server.parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 5100
    assert args.debug is False


def test_normalize_language_aliases_and_auto():
    assert server.normalize_language(None) is None
    assert server.normalize_language("  ") is None
    assert server.normalize_language("zh") == "Chinese"
    assert server.normalize_language("en_US") == "English"
    assert server.normalize_language("Klingon") == "Klingon"


def test_context_terms_are_extracted_deduped_and_prompt_is_built():
    terms = server.asr_context_terms_from_sources(
        "VoxT, Qwen3-ASR\nVoxT",
        {"hotwords": ["MLX", {"text": "Apple Silicon"}]},
        {"name": "fallback-name-extraction"},
    )

    assert terms == ["VoxT", "Qwen3-ASR", "MLX", "Apple Silicon", "fallback-name-extraction"]

    prompt, count = server.build_asr_context_prompt("base prompt", terms)
    assert count == len(terms)
    assert prompt.splitlines() == ["base prompt", *terms]


def test_context_terms_respect_count_and_character_limits():
    many_terms = [f"term-{i}" for i in range(server.ASR_CONTEXT_MAX_TERMS + 20)]
    assert server.dedupe_asr_context_terms(many_terms) == many_terms[: server.ASR_CONTEXT_MAX_TERMS]

    long_terms = [f"{i}-" + "x" * 500 for i in range(10)]
    limited = server.dedupe_asr_context_terms(long_terms)
    assert limited == long_terms[:3]
    assert sum(len(term) + 1 for term in limited) <= server.ASR_CONTEXT_MAX_CHARS


def test_http_preview_session_cache_matches_multiple_sessions():
    state = server.State()
    audio_a = np.ones(32_000, dtype=np.float32) * 0.1
    audio_b = np.ones(32_000, dtype=np.float32) * 0.2
    session_a = server.HttpRealtimeSession(language="Chinese", prompt="a")
    session_b = server.HttpRealtimeSession(language="English", prompt="b")
    server.update_http_session_audio(session_a, audio_a)
    server.update_http_session_audio(session_b, audio_b)

    server.add_http_session(state, session_a)
    server.add_http_session(state, session_b)

    assert server.find_matching_http_session(state, audio_a, language="Chinese", prompt="a") is session_a
    assert server.find_matching_http_session(state, audio_b, language="English", prompt="b") is session_b
    assert server.find_matching_http_session(state, audio_b, language="Chinese", prompt="a") is None


def test_http_preview_session_cache_evicts_oldest():
    state = server.State()
    sessions: list[server.HttpRealtimeSession] = []
    base_seen_at = server.time.monotonic()
    for index in range(server.HTTP_PREVIEW_SESSION_CACHE_MAX + 1):
        session = server.HttpRealtimeSession(language="Chinese", prompt=str(index))
        server.update_http_session_audio(session, np.ones(160, dtype=np.float32) * index)
        session.last_seen_at = base_seen_at + index
        sessions.append(session)
        server.add_http_session(state, session)

    assert len(state.http_realtime_sessions) == server.HTTP_PREVIEW_SESSION_CACHE_MAX
    assert sessions[0] not in state.http_realtime_sessions
    assert sessions[-1] in state.http_realtime_sessions


def test_require_non_empty_model_rejects_blank():
    assert server.require_non_empty_model(" model ") == "model"

    with pytest.raises(HTTPException) as exc:
        server.require_non_empty_model(" ")
    assert exc.value.status_code == 400


def test_quantization_bits_from_name():
    assert server.quantization_bits_from_name("mlx-community/Qwen3-ASR-1.7B-8bit") == 8
    assert server.quantization_bits_from_name("Qwen3-ASR-0.6B-6bit") == 6
    assert server.quantization_bits_from_name("plain-model") is None


def test_configure_hf_cache_sets_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)

    server.configure_hf_cache(tmp_path / "cache")

    hf_home = (tmp_path / "cache" / "huggingface").resolve()
    hub_cache = hf_home / "hub"
    assert hub_cache.exists()
    assert server.os.environ["HF_HOME"] == str(hf_home)
    assert server.os.environ["HUGGINGFACE_HUB_CACHE"] == str(hub_cache)
    assert server.os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert server.os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"


def test_preserve_debug_ws_audio_writes_wav_and_metadata(tmp_path):
    config = server.Config(
        host="127.0.0.1",
        port=5100,
        api_key="password",
        model="final-model",
        model_path="/tmp/final-model",
        preview_model="preview-model",
        preview_model_path="/tmp/preview-model",
        dtype="float16",
        max_file_size_mb=10,
        unload_after_sec=0,
        debug=True,
        debug_dir=str(tmp_path),
    )
    audio = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    path = server.preserve_debug_ws_audio(
        config,
        audio,
        client="127.0.0.1:12345",
        protocol="aliyun-qwen",
        request_model="client-ws-model",
        preview_model="preview-model",
        final_model="final-model",
        language="Chinese",
        chunks=[[0, 0, 3, 0.1]],
        final_text="hello",
    )

    assert path is not None
    assert path.parent == tmp_path
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == server.ASR_SAMPLE_RATE
        assert wav.getnframes() == audio.size

    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["protocol"] == "aliyun-qwen"
    assert metadata["request_model"] == "client-ws-model"
    assert metadata["preview_model"] == "preview-model"
    assert metadata["final_model"] == "final-model"
    assert metadata["language"] == "Chinese"
    assert metadata["text"] == "hello"
    assert metadata["chunks"] == [[0, 0, 3, 0.1]]


def test_debug_http_artifact_uses_flat_same_basename_metadata(tmp_path):
    config = server.Config(
        host="127.0.0.1",
        port=5100,
        api_key="password",
        model="final-model",
        model_path="/tmp/final-model",
        preview_model="preview-model",
        preview_model_path="/tmp/preview-model",
        dtype="float16",
        max_file_size_mb=10,
        unload_after_sec=0,
        debug=True,
        debug_dir=str(tmp_path),
    )
    source = tmp_path / "source.wav"
    source.write_bytes(b"fake-audio")

    audio_path = server.preserve_debug_final_audio(config, str(source), "sample.wav")
    assert audio_path is not None
    assert audio_path.parent == tmp_path
    assert audio_path.read_bytes() == b"fake-audio"

    server.write_debug_transcription_metadata(
        audio_path,
        protocol="whisper",
        request_model="client-model",
        preview_model="preview-model",
        final_model="final-model",
        language="Chinese",
        audio_duration=1.25,
        audio_samples=20_000,
        text="final text",
    )

    metadata_path = audio_path.with_suffix(".json")
    assert metadata_path.exists()
    assert metadata_path.stem == audio_path.stem
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "protocol": "whisper",
        "request_model": "client-model",
        "preview_model": "preview-model",
        "final_model": "final-model",
        "language": "Chinese",
        "duration": 1.25,
        "text": "final text",
        "chunks": [[0, 0, 20_000, 0.0]],
    }


def test_debug_text_log_helpers_do_not_print_full_text(capsys):
    server.log_debug_transcription_text(is_preview=False, text="secret transcript")
    server.log_debug_realtime_text(protocol="aliyun-qwen", text="secret realtime", sentence_end=False, delta="secret")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
