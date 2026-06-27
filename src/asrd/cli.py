from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path

from . import server
from .service import launchd


def doctor() -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("macOS", sys.platform == "darwin", platform.platform()))
    checks.append(("Apple Silicon", platform.machine() in {"arm64", "aarch64"}, platform.machine()))
    checks.append(("Python >= 3.10", sys.version_info >= (3, 10), platform.python_version()))
    checks.append(("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not found"))
    checks.append(("ffprobe", shutil.which("ffprobe") is not None, shutil.which("ffprobe") or "not found"))
    checks.append(("asrd command", shutil.which("asrd") is not None, shutil.which("asrd") or "not found on PATH"))

    cache_root = Path.home() / ".cache"
    hf_home = Path(os.environ.get("HF_HOME", cache_root / "huggingface")).expanduser()
    hub_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home / "hub")).expanduser()
    checks.append(("model cache", hub_cache.exists(), str(hub_cache)))

    width = max(len(name) for name, _, _ in checks)
    failed = 0
    for name, ok, detail in checks:
        marker = "ok" if ok else "warn"
        if not ok:
            failed += 1
        print(f"{marker:>4}  {name:<{width}}  {detail}")

    print()
    print("Defaults:")
    print(f"  HTTP: http://{server.DEFAULT_HOST}:{server.DEFAULT_PORT}/v1/audio/transcriptions")
    print(f"  WS:   ws://{server.DEFAULT_HOST}:{server.DEFAULT_PORT}/v1/realtime")
    print(f"  HF endpoint: {os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')}")
    print(f"  API key: {server.masked_api_key(server.DEFAULT_API_KEY)}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "serve":
        return server.main(args[1:] if args and args[0] == "serve" else args)
    if args[0] == "service":
        return launchd.main(args[1:])
    if args[0] == "doctor":
        return doctor()
    if args[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(prog="asrd", description="Local ASR daemon powered by MLX/Qwen3-ASR.")
        sub = parser.add_subparsers(dest="command")
        sub.add_parser("serve", help="Run the ASR HTTP/WebSocket server")
        sub.add_parser("service", help="Manage the macOS launchd service")
        sub.add_parser("doctor", help="Check the local asrd runtime environment")
        parser.print_help()
        return 0
    print(f"Unknown command: {args[0]}", file=sys.stderr)
    print("Use 'asrd --help', 'asrd serve --help', 'asrd service --help', or 'asrd doctor'.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
