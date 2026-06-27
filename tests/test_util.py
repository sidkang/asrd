from __future__ import annotations

import os

from asrd import util


def test_env_bool_accepts_debug_truthy(monkeypatch):
    monkeypatch.setenv("TEST_BOOL", "debug")
    assert util.env_bool("TEST_BOOL") is True

    monkeypatch.setenv("TEST_BOOL", " off ")
    assert util.env_bool("TEST_BOOL") is False

    monkeypatch.delenv("TEST_BOOL")
    assert util.env_bool("TEST_BOOL") is False


def test_safe_debug_filename_strips_paths_and_limits_length():
    value = util.safe_debug_filename("../bad dir/hello 世界?.wav")
    assert value == "hello_.wav"

    long_name = "a" * 300 + ".wav"
    assert len(util.safe_debug_filename(long_name)) == 180


def test_masked_api_key_preserves_default_only():
    assert util.masked_api_key("") == "not set"
    assert util.masked_api_key("password") == "password"
    assert util.masked_api_key("secret") == "configured"


def test_read_json_returns_empty_dict_on_invalid_or_missing(tmp_path):
    missing = tmp_path / "missing.json"
    assert util.read_json(missing) == {}

    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert util.read_json(bad) == {}

    good = tmp_path / "good.json"
    good.write_text('{"ok": true}', encoding="utf-8")
    assert util.read_json(good) == {"ok": True}
