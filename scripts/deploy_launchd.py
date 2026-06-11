#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

LABEL = "com.sid.voxt-qwen3-asr"
PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_PLIST = PROJECT_DIR / "launchd" / f"{LABEL}.plist"
TARGET_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
CACHE_DIR = PROJECT_DIR / ".cache"
LOG_DIR = Path.home() / "Library" / "Logs"
STDOUT_LOG = LOG_DIR / f"{LABEL}.out.log"
STDERR_LOG = LOG_DIR / f"{LABEL}.err.log"


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args))
    return subprocess.run(args, text=True, check=check)


def launchd_names() -> tuple[str, str]:
    domain = f"gui/{subprocess.check_output(['id', '-u'], text=True).strip()}"
    return domain, f"{domain}/{LABEL}"


def install_files() -> None:
    if not SOURCE_PLIST.exists():
        raise SystemExit(f"missing source plist: {SOURCE_PLIST}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_PLIST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_PLIST, TARGET_PLIST)


def deploy() -> int:
    install_files()
    domain, service = launchd_names()
    run(["launchctl", "bootout", domain, str(TARGET_PLIST)], check=False)
    run(["launchctl", "bootstrap", domain, str(TARGET_PLIST)])
    run(["launchctl", "enable", service])
    run(["launchctl", "kickstart", "-k", service])
    print_summary("Installed and started", service)
    return 0


def status() -> int:
    _, service = launchd_names()
    result = run(["launchctl", "print", service], check=False)
    return result.returncode


def restart() -> int:
    install_files()
    domain, service = launchd_names()
    # Re-bootstrap so changes to ProgramArguments or log paths in the plist are applied.
    run(["launchctl", "bootout", domain, str(TARGET_PLIST)], check=False)
    run(["launchctl", "bootstrap", domain, str(TARGET_PLIST)], check=True)
    run(["launchctl", "enable", service], check=True)
    run(["launchctl", "kickstart", "-k", service])
    print_summary("Restarted", service)
    return 0


def stop() -> int:
    domain, service = launchd_names()
    run(["launchctl", "bootout", domain, str(TARGET_PLIST)], check=False)
    print(f"Stopped: {service}")
    return 0


def uninstall() -> int:
    stop()
    if TARGET_PLIST.exists():
        print("+ rm", TARGET_PLIST)
        TARGET_PLIST.unlink()
    print("Uninstalled launch agent. Logs were kept:")
    print(f"  {STDOUT_LOG}")
    print(f"  {STDERR_LOG}")
    return 0


def print_summary(action: str, service: str) -> None:
    print(f"\n{action}: {service}")
    print("Endpoint: http://127.0.0.1:5100/v1/audio/transcriptions")
    print("Logs:")
    print(f"  tail -f {STDOUT_LOG}")
    print(f"  tail -f {STDERR_LOG}")
    print("Status:")
    print(f"  python3 scripts/deploy_launchd.py status")


def main(argv: list[str] | None = None) -> int:
    if sys.platform != "darwin":
        print("launchd deployment is only supported on macOS", file=sys.stderr)
        return 1
    parser = argparse.ArgumentParser(description="Manage the voxt-qwen3-asr LaunchAgent.")
    parser.add_argument(
        "command",
        nargs="?",
        default="deploy",
        choices=["deploy", "status", "restart", "stop", "uninstall"],
    )
    args = parser.parse_args(argv)
    return globals()[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
