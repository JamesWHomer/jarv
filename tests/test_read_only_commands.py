from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from jarv import commands, settings_command, usage_command
from jarv.config import DEFAULT_CONFIG, validate_config


def test_read_only_command_display_default_is_auto():
    assert DEFAULT_CONFIG["read_only_command_display"] == "auto"
    assert DEFAULT_CONFIG["print_usage_after_agent"] is False
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


def test_settings_exposes_print_usage_after_agent(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    row = next(row for row in settings_command._settings_rows(config) if row["key"] == "print_usage_after_agent")

    assert row["section"] == "display"
    assert settings_command._settings_value_text(row, config).plain == "off"

    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    updated, message = settings_command._settings_apply_quick(row, config)

    assert updated["print_usage_after_agent"] is True
    assert message == "saved Print usage after agent: on"


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


def test_usage_all_since_uses_global_renderer(monkeypatch):
    calls = []
    captured = {}
    monkeypatch.setattr(usage_command, "show_read_only_command", lambda body, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(usage_command, "global_usage_file", lambda: Path("global-usage.json"))

    def load_records(*, since=None, warn=True):
        captured["since"] = since
        captured["warn"] = warn
        return [
            {
                "created_at": "2026-05-29T01:00:00Z",
                "session_id": "session-id",
                "model": "test-model",
                "source": "root",
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "uncached_input_tokens": 10,
                "output_tokens": 5,
                "reasoning_output_tokens": 0,
                "total_tokens": 15,
            }
        ]

    monkeypatch.setattr(usage_command, "load_global_usage_records", load_records)

    usage_command.cmd_usage(["--all", "--since", "24h"])

    assert captured["since"] == timedelta(hours=24)
    assert captured["warn"] is True
    assert calls[0]["title"] == "usage"
    assert calls[0]["subtitle"] == "global-usage.json - last 24h"


def test_usage_day_alias_uses_24_hour_global_window(monkeypatch):
    captured = {}
    monkeypatch.setattr(usage_command, "_cmd_global_usage", lambda since, label: captured.update({"since": since, "label": label}))

    usage_command.cmd_usage(["day"])

    assert captured["since"] == timedelta(hours=24)
    assert captured["label"] == "last 24h"
