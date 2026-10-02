"""Unit tests for the pure /usage view-model (jarv.usage_view)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarv import usage_view


# --------------------------------------------------------------------------- #
# parse_usage_scope / resolve_scope
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "args,expected",
    [
        (None, "session"),
        ([], "session"),
        (["session"], "session"),
        (["day"], "day"),
        (["today"], "day"),
        (["week"], "week"),
        (["month"], "month"),
        (["all"], "all"),
        (["--all"], "all"),
        (["--all", "--since", "24h"], "since:24h"),
        (["--all", "--since=7d"], "since:7d"),
    ],
)
def test_parse_usage_scope_maps_known_forms(args, expected):
    scope_key, error = usage_view.parse_usage_scope(args)
    assert error is None
    assert scope_key == expected


@pytest.mark.parametrize("args", [["bogus"], ["--all", "--since", "5x"], ["week", "extra"]])
def test_parse_usage_scope_rejects_bad_input(args):
    scope_key, error = usage_view.parse_usage_scope(args)
    assert scope_key is None
    assert error


def test_resolve_scope_canonical_and_adhoc():
    assert usage_view.resolve_scope("week").window == timedelta(days=7)
    adhoc = usage_view.resolve_scope("since:24h")
    assert adhoc.window == timedelta(hours=24)
    assert adhoc.window_label == "last 24h"
    # Unknown keys fall back to the session scope rather than raising.
    assert usage_view.resolve_scope("nonsense").key == "session"


# --------------------------------------------------------------------------- #
# build_usage_view: session
# --------------------------------------------------------------------------- #
def test_build_usage_view_session(monkeypatch):
    usage_dict = {
        "totals": {"request_count": 2, "total_tokens": 1100, "provider_cost_usd": 5.1, "cost_exact_request_count": 2},
        "models": {
            "high-tokens-low-spend": {"total_tokens": 1000, "provider_cost_usd": 0.1, "cost_exact_request_count": 1},
            "low-tokens-high-spend": {"total_tokens": 100, "provider_cost_usd": 5.0, "cost_exact_request_count": 1},
        },
        "providers": {"openai": {"request_count": 2}},
        "tiers": {"standard": {"request_count": 2}},
        "sources": {"root": {"request_count": 2}},
        "last_request": {"model": "low-tokens-high-spend"},
        "last_root_request": {"model": "root-model", "input_tokens": 250},
    }
    monkeypatch.setattr(usage_view, "usage_file_for", lambda _history_file: Path("usage.json"))
    monkeypatch.setattr(usage_view, "load_usage", lambda _usage_path, _session_id: usage_dict)
    monkeypatch.setattr(usage_view, "known_context_window", lambda _model=None, *a, **k: 1_000)

    ctx = SimpleNamespace(history_file=Path("history.json"), session_id="session-id")
    view = usage_view.build_usage_view("session", ctx=ctx)

    assert view.scope_key == "session"
    assert view.window_label == "Session"
    assert view.source_path == "usage.json"
    assert view.request_count == 2
    assert view.is_empty is False
    assert view.trend == []
    # Models are ordered by spend, not token count.
    assert [name for name, _ in view.models] == ["low-tokens-high-spend", "high-tokens-low-spend"]
    # Context headroom is derived from the last root request.
    assert view.context is not None
    assert view.context["window"] == 1_000
    assert view.context["used"] == 250
    assert view.context["remaining"] == 750
    assert view.context["percent"] == pytest.approx(25.0)


def test_build_usage_view_session_empty(monkeypatch):
    monkeypatch.setattr(usage_view, "usage_file_for", lambda _history_file: Path("usage.json"))
    monkeypatch.setattr(usage_view, "load_usage", lambda _usage_path, _session_id: {"totals": {}})
    ctx = SimpleNamespace(history_file=Path("history.json"), session_id="session-id")

    view = usage_view.build_usage_view("session", ctx=ctx)

    assert view.is_empty is True
    assert view.request_count == 0
    assert view.context is None


# --------------------------------------------------------------------------- #
# build_usage_view: windowed + daily bucketing
# --------------------------------------------------------------------------- #
def _records():
    return [
        {
            "created_at": "2026-06-28T05:00:00Z", "session_id": "s", "model": "m1",
            "provider": "openai", "source": "root", "served_service_tier": "standard",
            "input_tokens": 60, "cached_input_tokens": 0, "uncached_input_tokens": 60,
            "output_tokens": 40, "reasoning_output_tokens": 0, "total_tokens": 100,
            "provider_cost_usd": 1.0, "cost_status": "exact",
        },
        {
            "created_at": "2026-06-29T05:00:00Z", "session_id": "s", "model": "m2",
            "provider": "openai", "source": "subagent", "served_service_tier": "standard",
            "input_tokens": 120, "cached_input_tokens": 0, "uncached_input_tokens": 120,
            "output_tokens": 80, "reasoning_output_tokens": 0, "total_tokens": 200,
            "provider_cost_usd": 2.0, "cost_status": "exact",
        },
    ]


def test_build_usage_view_window(monkeypatch):
    captured = {}

    def load_records(*, since=None, now=None, warn=True):
        captured["since"] = since
        return _records()

    monkeypatch.setattr(usage_view, "load_global_usage_records", load_records)
    monkeypatch.setattr(usage_view, "global_usage_jsonl_file", lambda: Path("usage.jsonl"))
    now = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)

    view = usage_view.build_usage_view("week", now=now)

    # Twice the window: the earlier half feeds the period-over-period comparison.
    assert captured["since"] == timedelta(days=14)
    assert view.scope_key == "week"
    assert view.window_label == "This week"
    assert view.source_path == "usage.jsonl"
    assert view.request_count == 2
    assert view.totals["total_tokens"] == 300
    assert view.context is None
    assert [name for name, _ in view.models] == ["m2", "m1"]  # by spend


def test_window_daily_bucketing_is_contiguous(monkeypatch):
    monkeypatch.setattr(usage_view, "_LOCAL_TZ", timezone.utc)
    monkeypatch.setattr(
        usage_view, "load_global_usage_records",
        lambda *, since=None, now=None, warn=True: _records(),
    )
    monkeypatch.setattr(usage_view, "global_usage_jsonl_file", lambda: Path("usage.jsonl"))
    now = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)

    view = usage_view.build_usage_view("week", now=now)

    # A 7-day window spans the start date through today, inclusive (8 calendar days).
    assert view.trend_unit == "day"
    assert len(view.trend) == 8
    assert view.trend[0].start == datetime(2026, 6, 23)
    assert view.trend[-1].start == datetime(2026, 6, 30)

    by_day = {bucket.start.day: bucket for bucket in view.trend}
    assert by_day[28].spend_usd == pytest.approx(1.0)
    assert by_day[28].total_tokens == 100
    assert by_day[29].spend_usd == pytest.approx(2.0)
    assert by_day[29].request_count == 1
    # A day with no activity is still present, at zero.
    assert by_day[23].spend_usd == 0.0
    assert by_day[23].total_tokens == 0


def test_build_window_views_loads_records_once(monkeypatch):
    calls = {"count": 0}

    def load_records(path=None, *, since=None, now=None, warn=True):
        calls["count"] += 1
        # The batch loader reads the full history and filters each scope in memory.
        assert since is None
        return _records()

    monkeypatch.setattr(usage_view, "load_global_usage_records", load_records)
    monkeypatch.setattr(usage_view, "global_usage_jsonl_file", lambda: Path("usage.jsonl"))
    now = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)

    views = usage_view.build_window_views(now=now)

    # One file read warms every windowed scope (session is excluded).
    assert calls["count"] == 1
    assert set(views) == {"day", "week", "month", "all"}
    # In-memory ``since`` filtering distinguishes the scopes from one load:
    # both records (06-28, 06-29) fall inside week/month/all ...
    assert views["all"].request_count == 2
    assert views["week"].request_count == 2
    assert views["month"].request_count == 2
    # ... but the rolling 24h "day" window (cutoff 06-29T12:00) drops both.
    assert views["day"].is_empty


def test_window_view_empty(monkeypatch):
    monkeypatch.setattr(
        usage_view, "load_global_usage_records",
        lambda *, since=None, now=None, warn=True: [],
    )
    monkeypatch.setattr(usage_view, "global_usage_jsonl_file", lambda: Path("usage.jsonl"))

    view = usage_view.build_usage_view("month", now=datetime(2026, 6, 30, tzinfo=timezone.utc))

    assert view.is_empty is True
    assert view.trend == []
    assert view.models == []


def test_daily_buckets_follow_local_calendar_days(monkeypatch):
    # 2026-06-28T20:00Z is already 06-29 06:00 in UTC+10: it belongs to the 29th.
    monkeypatch.setattr(usage_view, "_LOCAL_TZ", timezone(timedelta(hours=10)))
    record = dict(_records()[0], created_at="2026-06-28T20:00:00Z")
    monkeypatch.setattr(
        usage_view, "load_global_usage_records",
        lambda *, since=None, now=None, warn=True: [record],
    )
    monkeypatch.setattr(usage_view, "global_usage_jsonl_file", lambda: Path("usage.jsonl"))

    view = usage_view.build_usage_view("week", now=datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc))

    by_day = {bucket.start.day: bucket.spend_usd for bucket in view.trend}
    assert by_day[29] == pytest.approx(1.0)
    assert by_day[28] == 0.0
    assert view.trend[-1].start == datetime(2026, 6, 30)  # 22:00 local, still the 30th


def test_day_scope_buckets_hourly(monkeypatch):
    monkeypatch.setattr(usage_view, "_LOCAL_TZ", timezone.utc)
    records = [
        dict(_records()[0], created_at="2026-06-30T09:15:00Z"),
        dict(_records()[1], created_at="2026-06-30T09:45:00Z"),
    ]
    monkeypatch.setattr(
        usage_view, "load_global_usage_records",
        lambda *, since=None, now=None, warn=True: records,
    )
    monkeypatch.setattr(usage_view, "global_usage_jsonl_file", lambda: Path("usage.jsonl"))

    view = usage_view.build_usage_view("day", now=datetime(2026, 6, 30, 12, 30, tzinfo=timezone.utc))

    assert view.trend_unit == "hour"
    # The rolling 24h window starts mid-hour, so both its partial end hours appear.
    assert len(view.trend) == 25
    assert view.trend[0].start == datetime(2026, 6, 29, 12)
    assert view.trend[-1].start == datetime(2026, 6, 30, 12)
    by_hour = {bucket.start: bucket for bucket in view.trend}
    assert by_hour[datetime(2026, 6, 30, 9)].spend_usd == pytest.approx(3.0)
    assert by_hour[datetime(2026, 6, 30, 9)].request_count == 2


def test_window_views_carry_previous_period_spend(monkeypatch):
    records = [
        dict(_records()[0], created_at="2026-06-20T05:00:00Z"),  # previous 7d window
        dict(_records()[1], created_at="2026-06-29T05:00:00Z"),  # current 7d window
    ]
    monkeypatch.setattr(
        usage_view, "load_global_usage_records",
        lambda path=None, *, since=None, now=None, warn=True: records,
    )
    monkeypatch.setattr(usage_view, "global_usage_jsonl_file", lambda: Path("usage.jsonl"))

    views = usage_view.build_window_views(now=datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc))

    assert views["week"].previous_spend == pytest.approx(1.0)
    assert views["week"].previous_label == "7d"
    assert views["week"].request_count == 1
    assert views["month"].previous_label == "30d"
    assert views["month"].previous_spend is None  # nothing recorded before the window
    assert views["day"].previous_label == "24h"
    assert views["all"].previous_spend is None and views["all"].previous_label == ""
