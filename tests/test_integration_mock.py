from __future__ import annotations

import io
import json
import base64
import wave

import numpy as np
from fastapi.testclient import TestClient

from asrd import server


def make_config(tmp_path, *, debug: bool = False) -> server.Config:
    return server.Config(
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
        debug=debug,
        debug_dir=str(tmp_path / "debug"),
    )


def test_health_and_model_auth(tmp_path):
    app = server.create_app(make_config(tmp_path))

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["model"] == "final-model"
        assert health["preview_model"] == "preview-model"
        assert health["debug"] is False

        assert client.get("/v1/models").status_code == 401
        models = client.get("/v1/models", headers={"Authorization": "Bearer password"}).json()
        assert [item["id"] for item in models["data"]] == ["final-model", "preview-model"]


def test_http_transcription_uses_final_path_without_loading_model(monkeypatch, tmp_path):
    async def fake_final(state, config, full_audio, *, language, prompt):
        assert full_audio.dtype == np.float32
        assert language == "Chinese"
        assert prompt == "hello"
        return {"text": "mock final", "http_realtime": True, "http_final_strategy": "full-one-shot"}

    monkeypatch.setattr(server, "decode_audio_float32", lambda path, sample_rate: np.zeros(160, dtype=np.float32))
    monkeypatch.setattr(server, "audio_duration_seconds", lambda path: 0.01)
    monkeypatch.setattr(server, "http_realtime_final_transcription", fake_final)

    app = server.create_app(make_config(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer password"},
            data={"model": "client-model", "language": "zh", "prompt": "hello"},
            files={"file": ("sample.wav", io.BytesIO(b"not actually decoded"), "audio/wav")},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "mock final"}


def test_http_preview_filename_uses_preview_path(monkeypatch, tmp_path):
    async def fake_preview(state, config, full_audio, *, language, prompt):
        assert language is None
        return {"text": "mock preview", "http_realtime": True, "http_preview_cached": True}

    monkeypatch.setattr(server, "decode_audio_float32", lambda path, sample_rate: np.zeros(160, dtype=np.float32))
    monkeypatch.setattr(server, "audio_duration_seconds", lambda path: 0.01)
    monkeypatch.setattr(server, "http_realtime_preview_transcription", fake_preview)

    app = server.create_app(make_config(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer password"},
            data={"model": "client-model", "response_format": "text"},
            files={"file": ("voxt-openai-preview-123.wav", io.BytesIO(b"audio"), "audio/wav")},
        )

    assert response.status_code == 200
    assert response.text == "mock preview"


def test_http_preview_debug_artifact_uses_whisper_chunk_protocol(monkeypatch, tmp_path):
    async def fake_preview(state, config, full_audio, *, language, prompt):
        return {"text": "mock preview", "http_realtime": True, "http_preview_cached": True}

    monkeypatch.setattr(server, "decode_audio_float32", lambda path, sample_rate: np.zeros(160, dtype=np.float32))
    monkeypatch.setattr(server, "audio_duration_seconds", lambda path: 0.01)
    monkeypatch.setattr(server, "http_realtime_preview_transcription", fake_preview)

    app = server.create_app(make_config(tmp_path, debug=True))
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer password"},
            data={"model": "client-preview-model"},
            files={"file": ("voxt-openai-preview-123.wav", io.BytesIO(b"audio"), "audio/wav")},
        )

    assert response.status_code == 200
    wavs = list((tmp_path / "debug").glob("*.wav"))
    assert len(wavs) == 1
    metadata = json.loads(wavs[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata == {
        "protocol": "whisper-chunk",
        "request_model": "client-preview-model",
        "preview_model": "preview-model",
        "final_model": "final-model",
        "language": None,
        "duration": 0.01,
        "text": "mock preview",
        "chunks": [[0, 0, 160, 0.0]],
    }


def test_http_debug_artifact_audio_removed_if_transcription_fails(monkeypatch, tmp_path):
    def raise_decode(path, sample_rate):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(server, "decode_audio_float32", raise_decode)
    monkeypatch.setattr(server, "audio_duration_seconds", lambda path: 0.01)

    app = server.create_app(make_config(tmp_path, debug=True))
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer password"},
            data={"model": "client-model"},
            files={"file": ("sample.wav", io.BytesIO(b"audio"), "audio/wav")},
        )

    assert response.status_code == 500
    assert list((tmp_path / "debug").glob("*")) == []


def test_websocket_empty_finish_no_model_load(tmp_path):
    app = server.create_app(make_config(tmp_path))

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer password"}) as ws:
            ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "input_audio_format": "pcm",
                        "input_audio_transcription": {"language": "zh"},
                        "sample_rate": 16000,
                    },
                }
            )
            updated = ws.receive_json()
            assert updated["type"] == "session.updated"
            assert updated["session"]["model"] == "preview-model"

            ws.send_json({"type": "session.finish"})
            completed = ws.receive_json()
            finished = ws.receive_json()

    assert completed["type"] == "conversation.item.input_audio_transcription.completed"
    assert completed["text"] == ""
    assert finished["type"] == "session.finished"


def test_websocket_invalid_audio_reports_error(tmp_path):
    app = server.create_app(make_config(tmp_path))

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer password"}) as ws:
            ws.send_json({"type": "session.update", "session": {}})
            assert ws.receive_json()["type"] == "session.updated"

            ws.send_text(json.dumps({"type": "input_audio_buffer.append", "audio": "not base64"}))
            error = ws.receive_json()

    assert error["type"] == "error"
    assert error["error"]["code"] == "invalid_audio"


def test_websocket_debug_artifact_records_chunk_manifest(monkeypatch, tmp_path):
    async def fake_final(*args, **kwargs):
        return False, False

    monkeypatch.setattr(server, "ws_update_final_correction_with_deadline", fake_final)

    app = server.create_app(make_config(tmp_path, debug=True))
    raw = np.array([1, 2, 3, 4], dtype="<i2").tobytes()
    audio_b64 = base64.b64encode(raw).decode("ascii")

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer password"}) as ws:
            ws.send_json({"type": "session.update", "session": {"input_audio_transcription": {"language": "zh"}}})
            assert ws.receive_json()["type"] == "session.updated"
            ws.send_json({"type": "input_audio_buffer.append", "audio": audio_b64})
            ws.send_json({"type": "session.finish"})
            assert ws.receive_json()["type"] == "conversation.item.input_audio_transcription.completed"
            assert ws.receive_json()["type"] == "session.finished"

    wavs = list((tmp_path / "debug").glob("*.wav"))
    assert len(wavs) == 1
    with wave.open(str(wavs[0]), "rb") as wav:
        assert wav.getframerate() == server.ASR_SAMPLE_RATE
        assert wav.getnframes() == 4

    metadata = json.loads(wavs[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["protocol"] == "aliyun-qwen"
    assert metadata["request_model"] == ""
    assert metadata["preview_model"] == "preview-model"
    assert metadata["final_model"] == "final-model"
    assert metadata["language"] == "Chinese"
    assert metadata["chunks"][0][:3] == [0, 0, 4]
    assert isinstance(metadata["chunks"][0][3], float)
