from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path


def env_bool(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "debug"}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex}"


def safe_debug_filename(filename: str) -> str:
    base = Path(filename).name.strip() or "audio.wav"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:180] or "audio.wav"


def masked_api_key(api_key: str, default_api_key: str = "password") -> str:
    if not api_key:
        return "not set"
    if api_key == default_api_key:
        return default_api_key
    return "configured"
