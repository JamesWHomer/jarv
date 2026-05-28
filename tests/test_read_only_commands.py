from pathlib import Path
from types import SimpleNamespace

from jarv import commands, settings_command, usage_command
from jarv.config import DEFAULT_CONFIG, validate_config


def test_read_only_command_display_default_is_auto():
    assert DEFAULT_CONFIG["read_only_command_display"] == "auto"
    assert validate_config(dict(DEFAULT_CONFIG))


def test_validate_config_rejects_invalid_read_only_command_display():
    config = {**DEFAULT_CONFIG, "read_only_command_display": "sideways"}

    assert not validate_config(config)


def test_settings_exposes_read_only_command_display(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    row = next(row for row in settings_command._settings_rows(config) if row["key"] == "read_only_command_display")

    assert row["section"] == "display"
    assert settings_command._settings_value_text(row, config).plain == "auto"

    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    updated, message = settings_command._settings_apply_quick(row, config)

    assert updated["read_only_command_display"] == "print"
    assert message == "saved Read-only commands: print"


def test_help_about_and_config_use_shared_renderer(monkeypatch):
    calls = []
    monkeypatch.setattr(commands, "show_read_only_command", lambda body, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(commands, "load_config", lambda: dict(DEFAULT_CONFIG))

    commands.print_help(include_setup_nudge=False)
    commands.print_about(include_setup_nudge=False)
    commands.cmd_config()

    assert [call["title"] for call in calls] == ["help", "about", "config"]
    assert calls[2]["config"]["read_only_command_display"] == "auto"


def test_usage_empty_state_uses_shared_renderer(monkeypatch):
    calls = []
    monkeypatch.setattr(usage_command, "show_read_only_command", lambda body, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        usage_command,
        "prepare_session_context",
        lambda: SimpleNamespace(history_file=Path("history.json"), session_id="session-id"),
    )
    monkeypatch.setattr(usage_command, "usage_file_for", lambda _history_file: Path("usage.json"))
    monkeypatch.setattr(usage_command, "load_usage", lambda _usage_path, _session_id: {"totals": {}})

    usage_command.cmd_usage()

    assert len(calls) == 1
    assert calls[0]["title"] == "usage"
    assert calls[0]["subtitle"] == "usage.json"
