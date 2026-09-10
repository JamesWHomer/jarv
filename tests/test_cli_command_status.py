"""CLI failures are observable by scripts without ending the interactive UI."""

import sys

import pytest

from jarv import cli, commands, config, session_browser


@pytest.fixture(autouse=True)
def isolate_cli(monkeypatch):
    monkeypatch.setattr(cli, "_setup_nudge", lambda: None)
    monkeypatch.setattr(cli, "_print_previous_update_result", lambda: None)
    monkeypatch.setattr(cli, "_print_previous_uninstall_result", lambda: None)


@pytest.mark.parametrize("arguments", [
    ["/unknown"], ["/set"], ["/set", "command_timeout", "banana"],
    ["/history", "unexpected"], ["/setup", "unknown"],
    ["/unset", "model", "extra"], ["/setup", "model", "extra"],
    ["/undo", "banana"], ["/redo", "0"], ["/usage", "unknown"],
])
def test_invalid_command_exits_nonzero(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["jarv", *arguments])
    monkeypatch.setattr(config, "load_config", lambda: dict(config.DEFAULT_CONFIG))
    monkeypatch.setattr(config, "save_config", lambda _value: pytest.fail("invalid config was saved"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_failed_session_load_exits_nonzero(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["jarv", "/sessions", "missing"])
    monkeypatch.setattr(session_browser, "load_sessions", lambda: {"sessions": {}, "terminals": {}})
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1


def test_command_alias_preserves_update_failure_status(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["jarv", "update"])
    monkeypatch.setattr(cli, "_stdin_is_piped", lambda: False)
    monkeypatch.setattr(cli, "_maybe_command", lambda *_args: (True, "/update", []))
    monkeypatch.setattr(commands, "cmd_update", lambda: 1)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1


def test_interactive_failed_command_returns_without_exiting():
    assert cli._run_slash_command("/set", []) is True


def test_successful_set_exits_normally_and_saves(monkeypatch):
    saved = []
    monkeypatch.setattr(sys, "argv", ["jarv", "/set", "command_timeout", "42"])
    monkeypatch.setattr(config, "load_config", lambda: dict(config.DEFAULT_CONFIG))
    monkeypatch.setattr(config, "save_config", saved.append)
    assert cli.main() is None
    assert saved[0]["command_timeout"] == 42
