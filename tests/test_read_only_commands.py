import io
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from jarv import commands, config as config_module, history, settings_command, usage, usage_command, usage_view
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
    assert calls[0]["fill_screen"] is False
    assert calls[2]["config"]["read_only_command_display"] == "fullscreen"
    assert calls[2]["fill_screen"] is True


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
        "/usage [session|day|week|month|all]",
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


def test_reference_docs_cover_every_menu_command():
    """Every menu-visible command must appear in both /help and /about.

    The autocomplete menu is already kept in sync with the registry
    (test_command_menu.py); this is the missing analogue for the reference
    docs, so a newly registered command cannot silently go undocumented.
    """
    import re

    from jarv.command_registry import COMMANDS

    help_text = _render_help_text()
    about_text = _render_read_only_text(commands._about_body())

    for name, meta in COMMANDS.items():
        if not meta.menu:
            continue
        pattern = rf"/{re.escape(name)}\b"
        assert re.search(pattern, help_text), f"/{name} is missing from /help"
        assert re.search(pattern, about_text), f"/{name} is missing from /about"


def _session_view_for_test(monkeypatch, usage_dict, *, context_window=1_000):
    monkeypatch.setattr(usage_view, "usage_file_for", lambda _history_file: Path("usage.json"))
    monkeypatch.setattr(usage_view, "load_usage", lambda _usage_path, _session_id: usage_dict)
    monkeypatch.setattr(usage_view, "known_context_window", lambda _model=None, *a, **k: context_window)
    monkeypatch.setattr(usage_command, "known_context_window", lambda _model=None, *a, **k: context_window)
    ctx = SimpleNamespace(history_file=Path("history.json"), session_id="session-id")
    return usage_view.build_usage_view("session", ctx=ctx)


def _window_records():
    return [
        {
            "created_at": "2026-06-29T01:00:00Z",
            "session_id": "s", "model": "gpt-5.4-mini", "provider": "openai", "source": "root",
            "served_service_tier": "standard",
            "input_tokens": 100, "cached_input_tokens": 0, "uncached_input_tokens": 100,
            "output_tokens": 50, "reasoning_output_tokens": 0, "total_tokens": 150,
            "provider_cost_usd": 1.5, "cost_status": "exact",
            "context_breakdown": {"system": 5, "tools": 5, "history": 90, "tool_io": 0, "reasoning": 0},
        },
        {
            "created_at": "2026-06-29T02:00:00Z",
            "session_id": "s", "model": "claude-sonnet", "provider": "openai", "source": "subagent",
            "served_service_tier": "standard",
            "input_tokens": 40, "cached_input_tokens": 0, "uncached_input_tokens": 40,
            "output_tokens": 10, "reasoning_output_tokens": 0, "total_tokens": 50,
            "provider_cost_usd": 0.4, "cost_status": "exact",
        },
    ]


def test_usage_session_body_leads_with_hero_stats(monkeypatch):
    usage_dict = {
        "totals": {
            "request_count": 3,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "provider_cost_usd": 1.5,
            "cost_exact_request_count": 3,
        },
        "models": {
            "root-model": {"total_tokens": 150, "request_count": 3, "provider_cost_usd": 1.5, "cost_exact_request_count": 3},
        },
        "providers": {"openai": {"request_count": 3}},
        "tiers": {"standard": {"request_count": 3}},
        "sources": {"root": {"request_count": 2}, "subagent": {"request_count": 1}},
        "last_root_request": {
            "model": "root-model",
            "input_tokens": 410,
            "context_breakdown": {"system": 10, "tools": 20, "history": 70, "tool_io": 0, "reasoning": 0},
        },
    }
    view = _session_view_for_test(monkeypatch, usage_dict)
    rendered = _render_read_only_text(usage_command.build_usage_body(view))

    assert "SPEND" in rendered
    assert "TOKENS" in rendered
    assert "REQUESTS" in rendered
    assert "CONTEXT" in rendered            # context column is session-only
    assert "Session" in rendered            # scope tab
    assert "root-model" in rendered          # by-model bar
    assert "openai" in rendered              # secondary facts line
    assert "2 root / 1 subagent" in rendered
    assert "estimated allocation" in rendered  # demoted context detail
    # The old flat tables are gone.
    assert "Current model" not in rendered
    assert "Token totals" not in rendered


def test_usage_window_body_shows_chart_models_and_facts(monkeypatch):
    monkeypatch.setattr(
        usage_view,
        "load_global_usage_records",
        lambda *, since=None, now=None, warn=True: _window_records(),
    )
    monkeypatch.setattr(usage_view, "global_usage_jsonl_file", lambda: Path("usage.jsonl"))
    now = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
    view = usage_view.build_usage_view("week", now=now)

    assert view.window_label == "This week"
    assert view.source_path == "usage.jsonl"
    assert view.request_count == 2

    rendered = _render_read_only_text(usage_command.build_usage_body(view))
    assert "SPEND" in rendered
    assert "Daily spend" in rendered          # >=2 days with spend -> trend chart
    assert "gpt-5.4-mini" in rendered          # top model by spend
    assert "Week" in rendered                  # active scope tab
    assert "openai" in rendered                # provider fact
    assert "1 root / 1 subagent" in rendered
    assert "CONTEXT" not in rendered           # context headroom is session-only
    assert "estimated allocation" not in rendered  # context detail is session-only too


def test_usage_empty_state_keeps_tabs(monkeypatch):
    monkeypatch.setattr(
        usage_view,
        "load_global_usage_records",
        lambda *, since=None, now=None, warn=True: [],
    )
    monkeypatch.setattr(usage_view, "global_usage_jsonl_file", lambda: Path("usage.jsonl"))
    view = usage_view.build_usage_view("week", now=datetime(2026, 6, 30, tzinfo=timezone.utc))

    assert view.is_empty
    rendered = _render_read_only_text(usage_command.build_usage_body(view))
    assert "No usage recorded for this week." in rendered
    assert "Session" in rendered  # tabs stay visible so the user can switch scope
    assert "Week" in rendered


def test_usage_screen_scope_keys_switch_and_reset_offset():
    screen = usage_command.UsageScreen(initial_scope="session")
    screen.offset = 7

    screen.on_key("RIGHT", 1)
    assert screen.scope_key == "day"
    assert screen.offset == 0

    screen.on_key("a", 1)
    assert screen.scope_key == "all"

    screen.on_key("LEFT", 1)
    assert screen.scope_key == "month"


def test_usage_screen_close_keys_stop():
    screen = usage_command.UsageScreen(initial_scope="week")
    screen._running = True
    screen.on_key("q", 1)
    assert screen._running is False


def test_usage_screen_preloads_window_views_in_one_read(monkeypatch):
    calls = {"count": 0}

    def load_records(path=None, *, since=None, now=None, warn=True):
        calls["count"] += 1
        return _window_records()

    monkeypatch.setattr(usage_view, "load_global_usage_records", load_records)
    monkeypatch.setattr(usage_view, "global_usage_jsonl_file", lambda: Path("usage.jsonl"))
    now = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
    screen = usage_command.UsageScreen(initial_scope="session", now=now)

    # Before the background preload lands, a window scope renders a loading state
    # (None) instead of blocking the loop thread on file I/O.
    screen.scope_key = "week"
    assert screen._view() is None

    # The preload reads the shared JSONL exactly once and warms every window scope.
    screen._preload_window_views()
    assert calls["count"] == 1
    assert {"day", "week", "month", "all"} <= set(screen._cache)

    # Switching periods now serves straight from the cache: no further reads.
    screen.scope_key = "month"
    view = screen._view()
    assert view is not None and view.scope_key == "month"
    assert calls["count"] == 1


def test_usage_static_fallback_prints_scoped_panel(monkeypatch):
    view = usage_view.UsageView(
        scope_key="week",
        window_label="This week",
        source_path="usage.jsonl",
        totals={"total_tokens": 150, "request_count": 2},
        cost=usage.usage_cost_summary({}),
        models=[("gpt-5.4-mini", {"total_tokens": 150})],
        providers={"openai": {}},
        tiers={"standard": {}},
        sources={"root": {"request_count": 2}},
        context=None,
        daily=[],
        request_count=2,
        is_empty=False,
        last_request=None,
        last_root=None,
    )
    monkeypatch.setattr(usage_command, "build_usage_view", lambda scope_key: view)
    monkeypatch.setattr(usage_command, "interactive_terminal", lambda: False)
    out = io.StringIO()
    monkeypatch.setattr(
        usage_command,
        "console",
        Console(file=out, force_terminal=False, color_system=None, width=200),
    )

    usage_command.cmd_usage(["week"])

    text = out.getvalue()
    assert "This week" in text
    assert "usage.jsonl" in text
    assert "SPEND" in text


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
