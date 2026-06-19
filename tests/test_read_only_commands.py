import io
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from jarv import commands, config as config_module, history, settings_command, usage_command
from jarv.config import DEFAULT_CONFIG, READ_ONLY_COMMAND_DISPLAY_CHOICES, validate_config


def _render_help_text() -> str:
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=180)
    console.print(commands._help_body())
    return output.getvalue()


def _render_read_only_text(body) -> str:
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=180)
    console.print(body)
    return output.getvalue()


def test_read_only_command_display_default_is_fullscreen():
    assert READ_ONLY_COMMAND_DISPLAY_CHOICES == ("fullscreen", "print")
    assert DEFAULT_CONFIG["read_only_command_display"] == "fullscreen"
    assert DEFAULT_CONFIG["print_usage_after_agent"] is False
    assert validate_config(dict(DEFAULT_CONFIG))


@pytest.mark.parametrize("legacy_mode", ["auto", "inline"])
def test_load_config_migrates_legacy_read_only_display_modes(monkeypatch, tmp_path, legacy_mode):
    config_file = tmp_path / "config.json"
    config_file.write_text(f'{{"read_only_command_display": "{legacy_mode}"}}', encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(history, "migrate_flat_session_files", lambda: None)

    loaded = config_module.load_config()

    assert loaded["read_only_command_display"] == "fullscreen"
    assert '"read_only_command_display": "fullscreen"' in config_file.read_text(encoding="utf-8")


def test_validate_config_rejects_invalid_read_only_command_display():
    config = {**DEFAULT_CONFIG, "read_only_command_display": "sideways"}

    assert not validate_config(config)


def test_settings_exposes_read_only_command_display(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    row = next(row for row in settings_command._settings_rows(config) if row["key"] == "read_only_command_display")

    assert row["section"] == "display"
    assert settings_command._settings_value_text(row, config).plain == "fullscreen"

    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    updated, message = settings_command._settings_apply_quick(row, config)

    assert updated["read_only_command_display"] == "print"
    assert message == "saved Read-only commands: print"


def test_settings_exposes_print_usage_after_agent(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    row = next(row for row in settings_command._settings_rows(config) if row["key"] == "print_usage_after_agent")

    assert row["section"] == "display"
    assert row["label"] == "Print usage"
    assert settings_command._settings_value_text(row, config).plain == "off"

    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    updated, message = settings_command._settings_apply_quick(row, config)

    assert updated["print_usage_after_agent"] is True
    assert message == "saved Print usage: on"


def test_settings_groups_account_and_behaviour_rows_in_requested_order():
    rows = settings_command._settings_rows(dict(DEFAULT_CONFIG))

    assert [
        row["label"] for row in rows if row["section"] == "account"
    ] == ["Provider", "API key", "Processing tier", "Base URL"]
    assert [
        row["label"] for row in rows if row["section"] == "behaviour"
    ] == ["Model", "Reasoning effort", "System prompt"]

    unsupported_rows = settings_command._settings_rows(
        {**DEFAULT_CONFIG, "provider": "groq"}
    )
    assert [
        row["label"] for row in unsupported_rows if row["section"] == "account"
    ] == ["Provider", "API key", "Base URL"]


def test_help_about_and_config_use_shared_renderer(monkeypatch):
    calls = []
    monkeypatch.setattr(commands, "show_read_only_command", lambda body, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(commands, "load_config", lambda: dict(DEFAULT_CONFIG))

    commands.print_help(include_setup_nudge=False)
    commands.print_about(include_setup_nudge=False)
    commands.cmd_config()

    assert [call["title"] for call in calls] == ["help", "about", "config"]
    assert calls[0]["fill_screen"] is True
    assert calls[2]["config"]["read_only_command_display"] == "fullscreen"


def test_help_is_compact_and_task_focused():
    help_text = _render_help_text()

    expected = [
        "jarv <prompt>",
        "command | jarv <instruction>",
        "git diff | jarv review this",
        "--provider <provider>",
        "-m, --model <model>",
        "-e, --effort <effort>",
        "--timeout <seconds>",
        "-s, --system <prompt>",
        "--new",
        "--incognito",
        "--version",
        "/sessions",
        "/setup [step]",
        "/usage [period]",
        "exit, quit, /exit, /quit",
        "/settings",
        "/config",
        "/about",
        "Common controls:",
        "Raw configuration:",
        "Full reference:",
    ]

    for item in expected:
        assert item in help_text


def test_help_uses_one_aligned_command_and_description_table():
    help_lines = _render_help_text().splitlines()
    expected_rows = {
        "jarv": "Start heads-up mode",
        "--provider <provider>": "Override the provider",
        "/new": "Start a fresh session",
        "/sessions": "List sessions",
        "/setup [step]": "Run setup or jump to a step",
        "exit, quit, /exit, /quit": "Leave heads-up mode",
    }

    description_columns = set()
    for command, description in expected_rows.items():
        line = next(line for line in help_lines if command in line and description in line)
        description_columns.add(line.index(description))

    assert len(description_columns) == 1
    assert "COMMAND / FLAG" in help_lines[0]
    assert "DESCRIPTION" in help_lines[0]
    assert "\u2500" * 20 in help_lines[1]
    assert sum(not line.strip() for line in help_lines) >= 5
    assert not any(line.lstrip().startswith("chat ") for line in help_lines)
    assert not any(line.lstrip().startswith("sessions ") for line in help_lines)


def test_read_only_bodies_do_not_repeat_panel_titles(monkeypatch):
    help_text = _render_help_text()
    about_text = _render_read_only_text(commands._about_body())
    config_bodies = []

    monkeypatch.setattr(commands, "show_read_only_command", lambda body, **_kwargs: config_bodies.append(body))
    monkeypatch.setattr(commands, "load_config", lambda: dict(DEFAULT_CONFIG))

    commands.cmd_config()
    config_text = _render_read_only_text(config_bodies[0])

    for label in ["usage", "flags", "commands", "more"]:
        assert f"{label} ─" not in help_text
    assert not about_text.lstrip().startswith("jarv\n")
    assert "settings ─" not in config_text


def test_help_omits_reference_config_and_path_sections():
    help_text = _render_help_text()
    lower_help = help_text.lower()

    assert "config keys" not in lower_help
    assert "paths" not in lower_help
    assert "sessions index" not in lower_help
    assert "session data" not in lower_help
    assert str(commands.CONFIG_FILE) not in help_text
    assert str(commands.SESSIONS_FILE) not in help_text
    assert str(commands.SESSIONS_DIR) not in help_text

    removed_config_rows = [
        "api_key",
        "max_history",
        "max_stdin_chars",
        "max_tool_output_chars",
        "command_timeout",
        "command_safety",
        "audit",
        "auditor_auto_approve",
        "auditor_model",
        "system_prompt",
        "max_subagent_depth",
        "subagent_thread_pool_max_workers",
        "check_updates",
        "read_only_command_display",
        "print_usage_after_agent",
    ]
    for row in removed_config_rows:
        assert row not in help_text


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


def test_usage_breakdown_is_reconciled_to_recorded_input_tokens():
    breakdown = {
        "system": 129,
        "tools": 483,
        "history": 4_289,
        "tool_io": 315,
        "reasoning": 0,
    }

    reconciled = usage_command._reconcile_breakdown(breakdown, 5_065)

    assert sum(reconciled.values()) == 5_065
    assert reconciled == {
        "system": 125,
        "tools": 469,
        "history": 4_165,
        "tool_io": 306,
        "reasoning": 0,
    }


def test_usage_context_line_has_no_leading_padding_and_shows_remaining(monkeypatch):
    monkeypatch.setattr(usage_command, "known_context_window", lambda _model: 1_050_000)

    line = usage_command._context_usage_renderable(
        {"model": "test-model", "input_tokens": 17_316}
    ).plain

    assert line.startswith("1.6% full")
    assert "(17,316 / 1,050,000)" in line
    assert line.endswith("1,032,684 remaining")


def test_session_usage_uses_current_and_exchange_labels(monkeypatch):
    calls = []
    usage = {
        "totals": {
            "request_count": 1,
            "input_tokens": 100,
            "cached_input_tokens": 25,
            "uncached_input_tokens": 75,
            "output_tokens": 10,
            "total_tokens": 110,
        },
        "last_request": {
            "model": "subagent-model",
            "input_tokens": 20,
            "cached_input_tokens": 0,
            "output_tokens": 5,
        },
        "last_root_request": {
            "model": "root-model",
            "provider": "openai",
            "input_tokens": 100,
            "context_breakdown": {
                "system": 10,
                "tools": 20,
                "history": 70,
                "tool_io": 0,
                "reasoning": 0,
            },
        },
    }
    monkeypatch.setattr(
        usage_command,
        "show_read_only_command",
        lambda body, **kwargs: calls.append((body, kwargs)),
    )
    monkeypatch.setattr(
        usage_command,
        "prepare_session_context",
        lambda: SimpleNamespace(history_file=Path("history.json"), session_id="session-id"),
    )
    monkeypatch.setattr(usage_command, "usage_file_for", lambda _history_file: Path("usage.json"))
    monkeypatch.setattr(usage_command, "load_usage", lambda _usage_path, _session_id: usage)
    monkeypatch.setattr(usage_command, "load_history", lambda _history_file: [{"role": "user"}])
    monkeypatch.setattr(usage_command, "known_context_window", lambda _model: 1_000)

    usage_command.cmd_usage()

    rendered = _render_read_only_text(calls[0][0])
    assert "Current model" in rendered
    assert "Current provider" in rendered
    assert "Exchanges" in rendered
    assert "1 exchange" in rendered
    assert "Last model" not in rendered
    assert "estimated allocation" in rendered


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
