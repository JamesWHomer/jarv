import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from jarv import commands, config as config_module, display, history, settings_command, usage, usage_command, usage_view
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
    assert DEFAULT_CONFIG["turn_summary"] is False
    assert DEFAULT_CONFIG["colour"] is True
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
    assert json.loads(config_file.read_text(encoding="utf-8"))["read_only_command_display"] == "fullscreen"


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


def test_settings_exposes_turn_summary(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    row = next(row for row in settings_command._settings_rows(config) if row["key"] == "turn_summary")

    assert row["section"] == "turn summary"
    assert row["label"] == "Turn summary"
    assert settings_command._settings_value_text(row, config).plain == "off"

    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    updated, message = settings_command._settings_apply_quick(row, config)

    assert updated["turn_summary"] is True
    assert message == "saved Turn summary: on"


def test_settings_exposes_colour(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    row = next(row for row in settings_command._settings_rows(config) if row["key"] == "colour")

    assert row["section"] == "display"
    assert row["label"] == "Colour"
    assert settings_command._settings_value_text(row, config).plain == "on"

    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    updated, message = settings_command._settings_apply_quick(row, config)

    assert updated["colour"] is False
    assert message == "saved Colour: off"
    # Toggling in /settings takes effect on the screen you toggled it from.
    assert display.console.no_color is True

    updated, message = settings_command._settings_apply_quick(row, config)
    assert updated["colour"] is True
    assert message == "saved Colour: on"
    assert display.console.no_color == display._ENV_NO_COLOR


@pytest.mark.parametrize("saved, expected", [
    ({"monochrome": True}, False),
    ({"monochrome": False}, True),
    ({"monochrome": True, "colour": True}, True),
    ({"monochrome": False, "colour": False}, False),
])
def test_load_config_migrates_monochrome(monkeypatch, tmp_path, saved, expected):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(history, "migrate_flat_session_files", lambda: None)

    loaded = config_module.load_config()

    assert loaded["colour"] is expected
    assert "monochrome" not in loaded
    assert json.loads(config_file.read_text(encoding="utf-8")) == loaded
    assert display.console.no_color == (not expected or display._ENV_NO_COLOR)


def test_settings_groups_account_and_behaviour_rows_in_requested_order():
    rows = settings_command._settings_rows(dict(DEFAULT_CONFIG))

    assert [
        row["label"] for row in rows if row["section"] == "account"
    ] == ["Provider", "API key", "Processing tier", "Base URL"]
    assert [
        row["label"] for row in rows if row["section"] == "behaviour"
    ] == ["Model", "Reasoning effort", "System prompt", "Project context"]

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
    assert str(history.SESSIONS_FILE) not in help_text
    assert str(history.SESSIONS_DIR) not in help_text

    removed_config_rows = [
        "api_key",
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
        "subagent_timeout",
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
    # Pin a narrow width so the compact single-column layout (and its one-line
    # secondary facts) is exercised deterministically, independent of the host
    # terminal size.
    rendered = _render_read_only_text(usage_command.build_usage_body(view, width=80))

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

    rendered = _render_read_only_text(usage_command.build_usage_body(view, width=80))
    assert "SPEND" in rendered
    assert "Spend per day" in rendered        # >=2 days with spend -> trend chart
    assert "gpt-5.4-mini" in rendered          # top model by spend
    assert "Week" in rendered                  # active scope tab
    assert "openai" in rendered                # provider fact
    assert "1 root / 1 subagent" in rendered
    assert "CONTEXT" not in rendered           # context headroom is session-only
    assert "estimated allocation" not in rendered  # context detail is session-only too


def _multi_model_usage_dict():
    """A session with enough models/providers/tiers to exercise the wide layout."""
    models = {
        # Strictly decreasing spend so model ordering is stable; m01 carries a
        # distinctive request count to prove the per-model Requests column.
        f"m{i:02d}": {
            "total_tokens": 1000 - i,
            "request_count": 4321 if i == 1 else i,
            "provider_cost_usd": (9 - i) * 1.0,
            "cost_exact_request_count": 1,
        }
        for i in range(1, 9)
    }
    return {
        "totals": {
            "request_count": 30,
            "input_tokens": 6000,
            "output_tokens": 3000,
            "cached_input_tokens": 1500,
            "uncached_input_tokens": 4500,
            "reasoning_output_tokens": 500,
            "total_tokens": 9000,
            "provider_cost_usd": 36.0,
            "cost_exact_request_count": 30,
        },
        "models": models,
        "providers": {
            "openai": {"total_tokens": 7000, "provider_cost_usd": 30.0, "cost_exact_request_count": 20},
            "anthropic": {"total_tokens": 2000, "provider_cost_usd": 6.0, "cost_exact_request_count": 10},
        },
        "tiers": {
            "standard": {"provider_cost_usd": 34.0, "cost_exact_request_count": 28},
            "flex": {"provider_cost_usd": 2.0, "cost_exact_request_count": 2},
        },
        "sources": {
            "root": {"request_count": 20, "provider_cost_usd": 30.0, "cost_exact_request_count": 20},
            "subagent": {"request_count": 10, "provider_cost_usd": 6.0, "cost_exact_request_count": 10},
        },
        "last_root_request": {"model": "m01", "input_tokens": 250},
    }


def test_usage_wide_layout_surfaces_more(monkeypatch):
    view = _session_view_for_test(monkeypatch, _multi_model_usage_dict())

    narrow = _render_read_only_text(usage_command.build_usage_body(view, width=80))
    wide = _render_read_only_text(usage_command.build_usage_body(view, width=160))

    # Narrow keeps today's compact layout: only the top 6 models, a one-line
    # secondary-facts summary, no per-model request counts, and a three-stat hero.
    assert "+ 2 more" in narrow
    assert "m08" not in narrow
    assert "2 providers" in narrow
    assert "By provider" not in narrow
    assert "4,321" not in narrow
    assert "AVG / REQ" not in narrow
    assert "CACHE HIT" not in narrow

    # Wide lists every model, adds the per-model Requests column, and promotes the
    # secondary facts into provider / tier / source / token breakdown blocks.
    assert "m08" in wide
    assert "+ 2 more" not in wide
    assert "4,321" in wide                    # per-model request count column
    assert "By provider" in wide
    assert "anthropic" in wide                # per-provider spend, not just a count
    assert "By tier" in wide
    assert "Tokens" in wide and "Input" in wide and "Output" in wide

    # ...and the hero band gains derived stat columns.
    assert "AVG / REQ" in wide                # cost per request: $36.00 / 30
    assert "CACHE HIT" in wide                # cached share of input: 1500 / 6000

    # At >= _VERY_WIDE the blocks sit two-up: a single rendered line carries both a
    # left-column and a right-column block title.
    assert any("By provider" in line and "By source" in line for line in wide.splitlines())


def test_usage_model_bars_scale_with_width(monkeypatch):
    view = _session_view_for_test(monkeypatch, _multi_model_usage_dict())
    narrow = _render_read_only_text(usage_command._model_bars(view, width=70))
    wide = _render_read_only_text(usage_command._model_bars(view, width=160))
    assert max(len(line) for line in wide.splitlines()) > max(
        len(line) for line in narrow.splitlines()
    )


def test_usage_model_bars_show_share_of_spend(monkeypatch):
    usage_dict = {
        "totals": {"request_count": 2, "total_tokens": 1100, "provider_cost_usd": 10.0, "cost_exact_request_count": 2},
        "models": {
            "pricey": {"total_tokens": 100, "request_count": 1, "provider_cost_usd": 9.0, "cost_exact_request_count": 1},
            "cheap": {"total_tokens": 1000, "request_count": 1, "provider_cost_usd": 1.0, "cost_exact_request_count": 1},
        },
    }
    view = _session_view_for_test(monkeypatch, usage_dict)

    rendered = _render_read_only_text(usage_command._model_bars(view, width=80))

    assert "share of spend" in rendered       # labelled columns
    assert "tokens" in rendered and "cost" in rendered
    pricey = next(line for line in rendered.splitlines() if line.startswith("pricey"))
    assert "90%" in pricey                     # 9 of 10 dollars, despite 9% of tokens


def test_usage_tokens_block_nests_cached_under_input(monkeypatch):
    view = _session_view_for_test(monkeypatch, _multi_model_usage_dict())

    rendered = _render_read_only_text(usage_command._tokens_block(view))

    cached = next(line for line in rendered.splitlines() if "cached" in line)
    reasoning = next(line for line in rendered.splitlines() if "reasoning" in line)
    # Shares of their parent, so cached matches the hero's CACHE HIT (1500 / 6000).
    assert cached.rstrip().endswith("25%")
    assert reasoning.rstrip().endswith("17%")  # 500 / 3000


def _view_with_previous(previous_spend):
    return usage_view.UsageView(
        scope_key="week",
        window_label="This week",
        source_path="usage.jsonl",
        totals={"total_tokens": 150, "request_count": 2},
        cost=usage.usage_cost_summary({"provider_cost_usd": 3.0, "cost_exact_request_count": 2}),
        models=[],
        providers={},
        tiers={},
        sources={},
        context=None,
        trend=[],
        request_count=2,
        is_empty=False,
        last_request=None,
        last_root=None,
        previous_spend=previous_spend,
        previous_label="7d",
    )


@pytest.mark.parametrize(
    "previous,expected",
    [(2.0, "▲ 50% vs prev 7d"), (6.0, "▼ 50% vs prev 7d"), (0.1, "▲ 30× vs prev 7d"), (3.0, "= flat vs prev 7d")],
)
def test_usage_hero_compares_spend_with_previous_period(previous, expected):
    rendered = _render_read_only_text(usage_command._hero_band(_view_with_previous(previous)))
    assert expected in rendered


def test_usage_hero_omits_comparison_without_previous_spend():
    rendered = _render_read_only_text(usage_command._hero_band(_view_with_previous(None)))
    assert "vs prev" not in rendered


def _trend_view(trend, unit):
    return usage_view.UsageView(
        scope_key="day",
        window_label="Today",
        source_path="usage.jsonl",
        totals={},
        cost={},
        models=[],
        providers={},
        tiers={},
        sources={},
        context=None,
        trend=trend,
        request_count=1,
        is_empty=False,
        last_request=None,
        last_root=None,
        trend_unit=unit,
    )


@pytest.mark.parametrize("width", [60, 80, 148])
def test_usage_hourly_chart_labels_stay_on_axis(width):
    start = datetime(2026, 6, 29, 12)
    trend = [
        usage_view.TrendBucket(
            start=start.replace(day=29 + (12 + h) // 24, hour=(12 + h) % 24),
            spend_usd=float(h % 5),
            total_tokens=100,
            request_count=1,
        )
        for h in range(25)
    ]
    chart = usage_command._trend_chart(_trend_view(trend, "hour"), width=width)

    lines = _render_read_only_text(chart).splitlines()
    assert lines[0].startswith("Spend per hour")
    assert "peak $4.00" in lines[0]
    axis = next(line for line in lines if "└" in line)
    ticks = lines[lines.index(axis) + 1].rstrip()
    assert len(ticks) <= len(axis.rstrip())    # no label hangs past the axis
    assert ticks.endswith("12:00")             # the newest hour is always labelled
    assert ticks.count("12:00") == 1           # ...and the wrapped oldest hour isn't repeated


def test_usage_chart_merges_long_daily_runs_into_weeks():
    start = datetime(2026, 1, 1)
    trend = [
        usage_view.TrendBucket(start=start + timedelta(days=d), spend_usd=1.0, total_tokens=10, request_count=1)
        for d in range(70)
    ]
    rendered = _render_read_only_text(usage_command._trend_chart(_trend_view(trend, "day"), width=148))
    assert "Spend per week" in rendered
    assert "peak $7.00" in rendered


@pytest.mark.parametrize("width", [60, 80, 148])
@pytest.mark.parametrize(
    "daily_spend,daily_tokens,peak_label",
    [(2.0, 10, "$14.00"), (0.0, 900, "6,300")],
)
def test_usage_grouped_chart_keeps_axis_and_bars_aligned(width, daily_spend, daily_tokens, peak_label):
    start = datetime(2026, 1, 1)
    trend = [
        usage_view.TrendBucket(
            start=start + timedelta(days=d),
            spend_usd=daily_spend,
            total_tokens=daily_tokens,
            request_count=1,
        )
        for d in range(70)
    ]
    chart = usage_command._trend_chart(_trend_view(trend, "day"), width=width)
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=width)
    console.print(chart)
    lines = output.getvalue().splitlines()
    top = next(line for line in lines if "┤" in line)
    baseline = next(line for line in lines if "└" in line)
    axis_column = baseline.index("└")

    assert top.strip().startswith(peak_label)
    assert top.index("┤") == axis_column
    for line in lines:
        if "│" in line:
            assert line.index("│") == axis_column
            # Equal weekly totals should form straight, full-height columns.
            assert line[axis_column + 1:] == top[axis_column + 1:]


def test_usage_grouped_chart_reserves_space_for_wider_peak_label():
    start = datetime(2026, 1, 1)
    trend = [
        usage_view.TrendBucket(
            start=start + timedelta(days=d), spend_usd=2.0, total_tokens=10, request_count=1,
        )
        for d in range(497)
    ]
    chart = usage_command._trend_chart(_trend_view(trend, "day"), width=80)
    rendered = _render_read_only_text(chart)

    # 71 weekly columns fit beside the daily label, but not the wider weekly
    # peak. Regrouping keeps the full history visible within the chart width.
    assert "Spend per 2 weeks" in rendered
    assert "peak $28.00" in rendered
    assert all(len(line.rstrip()) <= 80 for line in rendered.splitlines())


def test_usage_chart_falls_back_to_tokens_without_spend():
    trend = [
        usage_view.TrendBucket(start=datetime(2026, 6, 28 + d), spend_usd=0.0, total_tokens=50_000 * (d + 1), request_count=1)
        for d in range(3)
    ]
    rendered = _render_read_only_text(usage_command._trend_chart(_trend_view(trend, "day"), width=100))
    assert "Tokens per day" in rendered
    assert "peak 150K" in rendered


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
        trend=[],
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
