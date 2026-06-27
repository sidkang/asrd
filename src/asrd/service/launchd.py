from __future__ import annotations

import argparse
import html
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

LABEL = "com.sid.asrd"
LEGACY_LABEL = "com.sid.voxt-qwen3-asr"
TEMPLATE = Path(__file__).resolve().parent / "templates" / f"{LABEL}.plist"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
LOG_DIR = Path.home() / "Library" / "Logs"
TARGET_PLIST = LAUNCH_AGENTS_DIR / f"{LABEL}.plist"
STDOUT_LOG = LOG_DIR / f"{LABEL}.out.log"
STDERR_LOG = LOG_DIR / f"{LABEL}.err.log"
DEBUG_DIR = LOG_DIR / f"{LABEL}.debug"
LEGACY_PLIST = LAUNCH_AGENTS_DIR / f"{LEGACY_LABEL}.plist"


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args))
    return subprocess.run(args, text=True, check=check)


def launchd_names(label: str = LABEL) -> tuple[str, str]:
    domain = f"gui/{subprocess.check_output(['id', '-u'], text=True).strip()}"
    return domain, f"{domain}/{label}"


def command_path(explicit: str | None = None) -> str:
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    candidate = shutil.which("asrd")
    if candidate:
        return str(Path(candidate).resolve())
    argv0 = Path(sys.argv[0])
    if argv0.exists():
        return str(argv0.resolve())
    raise SystemExit("Could not find the asrd command. Run service install via `uv run asrd service install` or pass --command.")


def render_array(values: list[str]) -> str:
    return "\n".join(f"      <string>{html.escape(value)}</string>" for value in values)


def render_plist(*, program_arguments: list[str], working_directory: Path) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    return template.format(
        label=LABEL,
        program_arguments=render_array(program_arguments),
        working_directory=html.escape(str(working_directory)),
        stdout_log=html.escape(str(STDOUT_LOG)),
        stderr_log=html.escape(str(STDERR_LOG)),
    )


def install_files(args: argparse.Namespace) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    serve_args = [
        command_path(args.command),
        "serve",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--unload-after-sec",
        str(args.unload_after_sec),
    ]
    if args.api_key:
        serve_args.extend(["--api-key", args.api_key])
    if args.debug:
        serve_args.append("--debug")
    if args.debug_dir:
        serve_args.extend(["--debug-dir", args.debug_dir])
    TARGET_PLIST.write_text(render_plist(program_arguments=serve_args, working_directory=Path.cwd().resolve()), encoding="utf-8")


def stop_legacy_service() -> None:
    domain, legacy_service = launchd_names(LEGACY_LABEL)
    if LEGACY_PLIST.exists():
        run(["launchctl", "bootout", domain, str(LEGACY_PLIST)], check=False)


def install(args: argparse.Namespace) -> int:
    install_files(args)
    stop_legacy_service()
    domain, service = launchd_names()
    run(["launchctl", "bootout", domain, str(TARGET_PLIST)], check=False)
    run(["launchctl", "bootstrap", domain, str(TARGET_PLIST)])
    run(["launchctl", "enable", service])
    run(["launchctl", "kickstart", "-k", service])
    print_summary("Installed and started", service)
    return 0


def restart(args: argparse.Namespace) -> int:
    install_files(args)
    domain, service = launchd_names()
    run(["launchctl", "bootout", domain, str(TARGET_PLIST)], check=False)
    run(["launchctl", "bootstrap", domain, str(TARGET_PLIST)])
    run(["launchctl", "enable", service])
    run(["launchctl", "kickstart", "-k", service])
    print_summary("Restarted", service)
    return 0


def status(_: argparse.Namespace) -> int:
    _, service = launchd_names()
    return run(["launchctl", "print", service], check=False).returncode


def stop(_: argparse.Namespace) -> int:
    domain, service = launchd_names()
    run(["launchctl", "bootout", domain, str(TARGET_PLIST)], check=False)
    print(f"Stopped: {service}")
    return 0


def uninstall(_: argparse.Namespace) -> int:
    args = _
    stop(argparse.Namespace())
    if TARGET_PLIST.exists():
        print("+ rm", TARGET_PLIST)
        TARGET_PLIST.unlink()
    if getattr(args, "purge_logs", False):
        for path in (STDOUT_LOG, STDERR_LOG):
            if path.exists():
                print("+ rm", path)
                path.unlink()
        if DEBUG_DIR.exists():
            print("+ rm -r", DEBUG_DIR)
            shutil.rmtree(DEBUG_DIR)
        print("Uninstalled launch agent and purged logs.")
    else:
        print("Uninstalled launch agent. Logs were kept:")
        print(f"  {STDOUT_LOG}")
        print(f"  {STDERR_LOG}")
        print(f"  {DEBUG_DIR}")
    return 0


def plist_service_config() -> dict[str, str | bool | None]:
    if not TARGET_PLIST.exists():
        return {"installed": False}
    data = plistlib.loads(TARGET_PLIST.read_bytes())
    argv = list(data.get("ProgramArguments") or [])

    def value(flag: str) -> str | None:
        try:
            return str(argv[argv.index(flag) + 1])
        except (ValueError, IndexError):
            return None

    return {
        "installed": True,
        "label": str(data.get("Label") or LABEL),
        "command": str(argv[0]) if argv else None,
        "host": value("--host"),
        "port": value("--port"),
        "api_key": "configured" if value("--api-key") else "not set",
        "unload_after_sec": value("--unload-after-sec"),
        "debug": "--debug" in argv,
        "debug_dir": value("--debug-dir") or str(DEBUG_DIR),
        "working_directory": str(data.get("WorkingDirectory") or ""),
        "stdout": str(data.get("StandardOutPath") or STDOUT_LOG),
        "stderr": str(data.get("StandardErrorPath") or STDERR_LOG),
    }


def show(_: argparse.Namespace) -> int:
    config = plist_service_config()
    if not config.get("installed"):
        print(f"Not installed: {TARGET_PLIST}")
        return 1
    print(f"Label:             {config['label']}")
    print(f"Plist:             {TARGET_PLIST}")
    print(f"Command:           {config['command']}")
    print(f"Host:              {config['host']}")
    print(f"Port:              {config['port']}")
    print(f"API key:           {config['api_key']}")
    print(f"Unload after sec:  {config['unload_after_sec']}")
    print(f"Debug:             {'on' if config['debug'] else 'off'}")
    print(f"Debug dir:         {config['debug_dir']}")
    print(f"Working directory: {config['working_directory']}")
    print(f"Stdout:            {config['stdout']}")
    print(f"Stderr:            {config['stderr']}")
    return 0


def logs(args: argparse.Namespace) -> int:
    print(STDOUT_LOG)
    print(STDERR_LOG)
    if args.follow:
        run(["tail", "-f", str(STDOUT_LOG), str(STDERR_LOG)], check=False)
    return 0


def print_summary(action: str, service: str) -> None:
    print(f"\n{action}: {service}")
    print("Endpoint: http://127.0.0.1:5100/v1/audio/transcriptions")
    print("Realtime: ws://127.0.0.1:5100/v1/realtime")
    print("Logs:")
    print(f"  tail -f {STDOUT_LOG}")
    print(f"  tail -f {STDERR_LOG}")
    print("Status:")
    print("  asrd service status")


def add_install_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--command", help="Absolute asrd command path to place in the LaunchAgent.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument("--api-key", default="password")
    parser.add_argument("--unload-after-sec", type=float, default=30.0)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--debug-dir", default="")


def main(argv: list[str] | None = None) -> int:
    if sys.platform != "darwin":
        print("launchd service management is only supported on macOS", file=sys.stderr)
        return 1
    parser = argparse.ArgumentParser(prog="asrd service", description="Manage the asrd macOS LaunchAgent.")
    sub = parser.add_subparsers(dest="command_name", required=True)

    install_parser = sub.add_parser("install", help="Install and start the LaunchAgent")
    add_install_options(install_parser)
    install_parser.set_defaults(func=install)

    restart_parser = sub.add_parser("restart", help="Regenerate, bootstrap, and restart the LaunchAgent")
    add_install_options(restart_parser)
    restart_parser.set_defaults(func=restart)

    sub.add_parser("status", help="Print launchctl status").set_defaults(func=status)
    sub.add_parser("show", help="Show installed LaunchAgent configuration").set_defaults(func=show)
    sub.add_parser("stop", help="Stop the LaunchAgent").set_defaults(func=stop)
    uninstall_parser = sub.add_parser("uninstall", help="Stop and remove the LaunchAgent")
    uninstall_parser.add_argument("--purge-logs", action="store_true", help="Also remove stdout/stderr logs and debug artifacts.")
    uninstall_parser.set_defaults(func=uninstall)
    logs_parser = sub.add_parser("logs", help="Print log paths, or follow logs with --follow")
    logs_parser.add_argument("-f", "--follow", action="store_true")
    logs_parser.set_defaults(func=logs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
