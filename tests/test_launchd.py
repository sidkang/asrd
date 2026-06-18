from __future__ import annotations

import argparse
import plistlib
from pathlib import Path

from asrd.service import launchd


def test_render_plist_escapes_arguments_and_is_parseable(monkeypatch, tmp_path):
    monkeypatch.setattr(launchd, "STDOUT_LOG", tmp_path / "out & log.txt")
    monkeypatch.setattr(launchd, "STDERR_LOG", tmp_path / "err.log")

    rendered = launchd.render_plist(
        program_arguments=["/bin/asrd", "serve", "--api-key", "a&b"],
        working_directory=tmp_path / "work <dir>",
    )
    parsed = plistlib.loads(rendered.encode("utf-8"))

    assert parsed["Label"] == "com.sid.asrd"
    assert parsed["ProgramArguments"] == ["/bin/asrd", "serve", "--api-key", "a&b"]
    assert parsed["WorkingDirectory"] == str(tmp_path / "work <dir>")
    assert parsed["StandardOutPath"] == str(tmp_path / "out & log.txt")


def test_install_files_writes_expected_serve_arguments(monkeypatch, tmp_path):
    target_plist = tmp_path / "com.sid.asrd.plist"
    monkeypatch.setattr(launchd, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(launchd, "LAUNCH_AGENTS_DIR", tmp_path / "agents")
    monkeypatch.setattr(launchd, "TARGET_PLIST", target_plist)
    monkeypatch.setattr(launchd, "STDOUT_LOG", tmp_path / "logs" / "out.log")
    monkeypatch.setattr(launchd, "STDERR_LOG", tmp_path / "logs" / "err.log")

    command = tmp_path / "bin" / "asrd"
    command.parent.mkdir()
    command.touch()

    args = argparse.Namespace(
        command=str(command),
        host="127.0.0.1",
        port=5101,
        unload_after_sec=12.5,
        api_key="secret",
        debug=True,
        debug_dir="/tmp/debug",
    )

    monkeypatch.chdir(tmp_path)
    launchd.install_files(args)

    parsed = plistlib.loads(target_plist.read_bytes())
    assert parsed["ProgramArguments"] == [
        str(command.resolve()),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "5101",
        "--unload-after-sec",
        "12.5",
        "--api-key",
        "secret",
        "--debug",
        "--debug-dir",
        "/tmp/debug",
    ]
    assert parsed["WorkingDirectory"] == str(tmp_path.resolve())


def test_logs_prints_paths_without_follow(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(launchd, "STDOUT_LOG", tmp_path / "out.log")
    monkeypatch.setattr(launchd, "STDERR_LOG", tmp_path / "err.log")

    assert launchd.logs(argparse.Namespace(follow=False)) == 0
    captured = capsys.readouterr()
    assert str(tmp_path / "out.log") in captured.out
    assert str(tmp_path / "err.log") in captured.out


def test_service_install_debug_is_opt_in(monkeypatch):
    captured: list[argparse.Namespace] = []
    monkeypatch.setattr(launchd.sys, "platform", "darwin")
    monkeypatch.setattr(launchd, "install", lambda args: captured.append(args) or 0)

    assert launchd.main(["install", "--command", "/bin/asrd"]) == 0
    assert captured[0].debug is False

    assert launchd.main(["install", "--command", "/bin/asrd", "--debug"]) == 0
    assert captured[1].debug is True
