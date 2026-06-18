from __future__ import annotations

from asrd import cli


def test_cli_defaults_to_serve(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.server, "main", lambda argv: calls.append(argv) or 7)

    assert cli.main([]) == 7
    assert calls == [[]]


def test_cli_strips_explicit_serve(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.server, "main", lambda argv: calls.append(argv) or 0)

    assert cli.main(["serve", "--port", "5101"]) == 0
    assert calls == [["--port", "5101"]]


def test_cli_dispatches_service(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.launchd, "main", lambda argv: calls.append(argv) or 3)

    assert cli.main(["service", "status"]) == 3
    assert calls == [["status"]]


def test_cli_unknown_command_returns_usage_error(capsys):
    assert cli.main(["bad"]) == 2
    captured = capsys.readouterr()
    assert "Unknown command: bad" in captured.err
    assert "asrd --help" in captured.err
