from __future__ import annotations

import base64
import wave

import numpy as np

from asrd import audio


def test_pcm16_base64_decode_and_odd_byte_truncation():
    raw = np.array([-32768, 0, 32767], dtype="<i2").tobytes() + b"x"
    decoded = audio.decode_pcm16_base64(base64.b64encode(raw).decode("ascii"))

    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, np.array([-1.0, 0.0, 32767 / 32768], dtype=np.float32))


def test_decode_pcm16_base64_rejects_invalid_input():
    try:
        audio.decode_pcm16_base64("not base64!")
    except ValueError as exc:
        assert "base64-encoded PCM16" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_audio_diff_and_slicing():
    reference = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    candidate = np.array([0.0, 0.11, 0.19, 0.3], dtype=np.float32)

    assert audio.audio_diff_ok(reference, candidate, threshold=0.02)
    assert not audio.audio_diff_ok(reference, candidate[:3], threshold=0.02)

    sliced = audio.audio_slice_seconds(reference, 0.00005, 0.00020, sample_rate=10_000)
    np.testing.assert_array_equal(sliced, np.array([0.0, 0.1], dtype=np.float32))


def test_full_audio_from_parts_empty_single_and_multiple():
    empty = audio.full_audio_from_parts([])
    assert empty.dtype == np.float32
    assert empty.size == 0

    one = np.array([1.0], dtype=np.float32)
    np.testing.assert_array_equal(audio.full_audio_from_parts([one]), one)

    combined = audio.full_audio_from_parts([one, np.array([2.0], dtype=np.float32)])
    np.testing.assert_array_equal(combined, np.array([1.0, 2.0], dtype=np.float32))


def test_decode_wav_float32_mixes_channels(tmp_path):
    path = tmp_path / "stereo.wav"
    samples = np.array([[32767, -32768], [0, 32767]], dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(samples.tobytes())

    decoded = audio.decode_wav_float32(str(path), sample_rate=16_000)
    expected = np.array([(32767 / 32768 - 1.0) / 2, (0.0 + 32767 / 32768) / 2], dtype=np.float32)
    np.testing.assert_allclose(decoded, expected)
    assert audio.wav_duration_seconds(str(path)) == 2 / 16_000


def test_is_voxt_preview_upload():
    assert audio.is_voxt_preview_upload("voxt-openai-preview-abc.wav")
    assert not audio.is_voxt_preview_upload("voxt-remote-asr-abc.wav")
    assert not audio.is_voxt_preview_upload(None)
