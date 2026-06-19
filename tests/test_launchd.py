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


def test_show_reads_installed_plist(monkeypatch, tmp_path, capsys):
    target_plist = tmp_path / "com.sid.asrd.plist"
    monkeypatch.setattr(launchd, "TARGET_PLIST", target_plist)
    monkeypatch.setattr(launchd, "STDOUT_LOG", tmp_path / "out.log")
    monkeypatch.setattr(launchd, "STDERR_LOG", tmp_path / "err.log")
    monkeypatch.setattr(launchd, "DEBUG_DIR", tmp_path / "debug")
    rendered = launchd.render_plist(
        program_arguments=[
            "/bin/asrd",
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "5100",
            "--unload-after-sec",
            "30.0",
            "--api-key",
            "secret",
            "--debug",
        ],
        working_directory=tmp_path,
    )
    target_plist.write_text(rendered, encoding="utf-8")

    assert launchd.show(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "Host:              0.0.0.0" in out
    assert "Port:              5100" in out
    assert "API key:           configured" in out
    assert "Debug:             on" in out


def test_uninstall_purge_logs_removes_logs_and_debug_dir(monkeypatch, tmp_path):
    target_plist = tmp_path / "com.sid.asrd.plist"
    stdout_log = tmp_path / "out.log"
    stderr_log = tmp_path / "err.log"
    debug_dir = tmp_path / "debug"
    for path in (target_plist, stdout_log, stderr_log):
        path.write_text("x", encoding="utf-8")
    debug_dir.mkdir()
    (debug_dir / "artifact.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(launchd, "TARGET_PLIST", target_plist)
    monkeypatch.setattr(launchd, "STDOUT_LOG", stdout_log)
    monkeypatch.setattr(launchd, "STDERR_LOG", stderr_log)
    monkeypatch.setattr(launchd, "DEBUG_DIR", debug_dir)
    monkeypatch.setattr(launchd, "stop", lambda args: 0)

    assert launchd.uninstall(argparse.Namespace(purge_logs=True)) == 0
    assert not target_plist.exists()
    assert not stdout_log.exists()
    assert not stderr_log.exists()
    assert not debug_dir.exists()
