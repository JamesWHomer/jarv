"""Pure data -> view-model for the ``/usage`` screen.

No Rich, no I/O of its own beyond the shared :mod:`jarv.usage` data layer, so the
whole thing is unit-testable at fixed inputs. Both the interactive
:class:`jarv.usage_command.UsageScreen` and the static fallback render from the
:class:`UsageView` this module produces, so session and system-wide usage share a
single code path instead of the two ~80-line functions they used to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from .history import parse_timestamp, prepare_session_context, utc_now
from .usage import (
    aggregate_usage_records,
    global_usage_jsonl_file,
    known_context_window,
    load_global_usage_records,
    load_usage,
    usage_cost_summary,
    usage_file_for,
)


@dataclass(frozen=True)
class Scope:
    """One selectable usage window: its keys, labels, and time span."""

    key: str
    tab_label: str
    window_label: str
    window: timedelta | None


# The five canonical scopes the interactive screen cycles through. ``day`` uses a
# rolling 24h window (calendar-day bucketed for the trend chart); ``all`` has no
# cutoff and reads the full retained history.
SCOPES: tuple[Scope, ...] = (
    Scope("session", "Session", "Session", None),
    Scope("day", "Today", "Today", timedelta(days=1)),
    Scope("week", "Week", "This week", timedelta(days=7)),
    Scope("month", "Month", "This month", timedelta(days=30)),
    Scope("all", "All", "All time", None),
)

SCOPE_KEYS: tuple[str, ...] = tuple(scope.key for scope in SCOPES)
_SCOPE_BY_KEY: dict[str, Scope] = {scope.key: scope for scope in SCOPES}

_USAGE_ERROR = "Usage: jarv /usage [session|day|week|month|all]"
_SINCE_ERROR = "Usage: jarv /usage --all --since 24h|7d|30d"


@dataclass(frozen=True)
class DayBucket:
    """Aggregated spend/tokens for a single calendar day (UTC)."""

    day: date
    spend_usd: float
    total_tokens: int
    request_count: int


@dataclass(frozen=True)
class UsageView:
    """Everything the renderers need for one scope, derived and ready to draw."""

    scope_key: str
    window_label: str
    source_path: str
    totals: dict
    cost: dict
    models: list[tuple[str, dict]]
    providers: dict
    tiers: dict
    sources: dict
    context: dict | None
    daily: list[DayBucket]
    request_count: int
    is_empty: bool
    last_request: dict | None
    last_root: dict | None


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _parse_since_value(value: str) -> timedelta | None:
    raw = value.strip().lower()
    if len(raw) < 2:
        return None
    unit = raw[-1]
    try:
        amount = int(raw[:-1])
    except ValueError:
        return None
    if amount <= 0:
        return None
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    return None


def _since_scope(raw: str) -> tuple[str | None, str | None]:
    if _parse_since_value(raw) is None:
        return None, _SINCE_ERROR
    return f"since:{raw.strip().lower()}", None


def parse_usage_scope(args: list[str] | None) -> tuple[str | None, str | None]:
    """Map ``/usage`` arguments to a scope key. Returns ``(scope_key, error)``.

    ``[] -> session``; ``session|day|today|week|month|all`` map to their scope;
    and the back-compat ``--all [--since 24h|7d|30d]`` form maps to ``all`` or an
    ad-hoc ``since:<raw>`` window. Replaces the old 4-tuple parser.
    """
    args = [str(arg) for arg in (args or [])]
    if not args:
        return "session", None

    first = args[0].lower()
    if first == "--all":
        if len(args) == 1:
            return "all", None
        if len(args) == 3 and args[1] == "--since":
            return _since_scope(args[2])
        if len(args) == 2 and args[1].startswith("--since="):
            return _since_scope(args[1].split("=", 1)[1])
        return None, _SINCE_ERROR

    aliases = {
        "session": "session",
        "day": "day",
        "today": "day",
        "week": "week",
        "month": "month",
        "all": "all",
    }
    if len(args) == 1 and first in aliases:
        return aliases[first], None
    return None, _USAGE_ERROR


def resolve_scope(scope_key: str) -> Scope:
    """Return the :class:`Scope` for a key, synthesizing ad-hoc ``since:`` windows."""
    scope = _SCOPE_BY_KEY.get(scope_key)
    if scope is not None:
        return scope
    if isinstance(scope_key, str) and scope_key.startswith("since:"):
        raw = scope_key.split(":", 1)[1]
        window = _parse_since_value(raw)
        if window is not None:
            label = f"last {raw}"
            return Scope(scope_key, label, label, window)
    return _SCOPE_BY_KEY["session"]


# --------------------------------------------------------------------------- #
# View construction
# --------------------------------------------------------------------------- #
def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _dict_or_none(value: object) -> dict | None:
    return value if isinstance(value, dict) else None


def _sorted_models(models: dict) -> list[tuple[str, dict]]:
    """Models ordered by spend, then total tokens (both descending)."""
    items = [(str(name), bucket) for name, bucket in models.items() if isinstance(bucket, dict)]

    def sort_key(item: tuple[str, dict]) -> tuple[float, int]:
        _name, bucket = item
        spend = float(usage_cost_summary(bucket).get("total_usd") or 0.0)
        return (spend, int(bucket.get("total_tokens") or 0))

    return sorted(items, key=sort_key, reverse=True)


def _session_context(last_root: dict | None) -> dict | None:
    if not isinstance(last_root, dict):
        return None
    model = str(last_root.get("model") or "")
    window = known_context_window(model)
    if not window:
        return None
    used = int(last_root.get("input_tokens") or 0)
    return {
        "model": model,
        "window": int(window),
        "used": used,
        "remaining": max(int(window) - used, 0),
        "percent": (used / int(window)) * 100 if window else 0.0,
    }


def _bucket_daily(records: list[dict], scope: Scope, now: datetime) -> list[DayBucket]:
    """Group records into a contiguous run of calendar days (UTC) for the trend."""
    by_day: dict[date, list[dict]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        timestamp = parse_timestamp(str(record.get("created_at") or ""))
        if timestamp is None:
            continue
        day = timestamp.astimezone(timezone.utc).date()
        by_day.setdefault(day, []).append(record)

    if not by_day:
        return []
    today = now.astimezone(timezone.utc).date()
    if scope.window is not None:
        start = (now - scope.window).astimezone(timezone.utc).date()
    else:
        start = min(by_day)
    start = min(start, today)

    out: list[DayBucket] = []
    cursor = start
    one_day = timedelta(days=1)
    while cursor <= today:
        day_records = by_day.get(cursor)
        if day_records:
            totals = _dict(aggregate_usage_records(day_records).get("totals"))
            out.append(
                DayBucket(
                    day=cursor,
                    spend_usd=float(usage_cost_summary(totals).get("total_usd") or 0.0),
                    total_tokens=int(totals.get("total_tokens") or 0),
                    request_count=int(totals.get("request_count") or 0),
                )
            )
        else:
            out.append(DayBucket(day=cursor, spend_usd=0.0, total_tokens=0, request_count=0))
        cursor += one_day
    return out


def _session_view(scope: Scope, ctx) -> UsageView:
    ctx = ctx or prepare_session_context()
    usage_path = usage_file_for(ctx.history_file)
    usage = load_usage(usage_path, ctx.session_id)
    totals = _dict(usage.get("totals"))
    last_root = _dict_or_none(usage.get("last_root_request"))
    request_count = int(totals.get("request_count") or 0)
    return UsageView(
        scope_key=scope.key,
        window_label=scope.window_label,
        source_path=str(usage_path),
        totals=totals,
        cost=usage_cost_summary(totals),
        models=_sorted_models(_dict(usage.get("models"))),
        providers=_dict(usage.get("providers")),
        tiers=_dict(usage.get("tiers")),
        sources=_dict(usage.get("sources")),
        context=_session_context(last_root),
        daily=[],
        request_count=request_count,
        is_empty=request_count <= 0,
        last_request=_dict_or_none(usage.get("last_request")),
        last_root=last_root,
    )


def _window_view_from_records(scope: Scope, records: list[dict], now: datetime) -> UsageView:
    """Derive a windowed :class:`UsageView` from records already filtered to ``scope``."""
    usage = aggregate_usage_records(records)
    totals = _dict(usage.get("totals"))
    request_count = int(totals.get("request_count") or 0)
    return UsageView(
        scope_key=scope.key,
        window_label=scope.window_label,
        source_path=str(global_usage_jsonl_file()),
        totals=totals,
        cost=usage_cost_summary(totals),
        models=_sorted_models(_dict(usage.get("models"))),
        providers=_dict(usage.get("providers")),
        tiers=_dict(usage.get("tiers")),
        sources=_dict(usage.get("sources")),
        context=None,
        daily=_bucket_daily(records, scope, now),
        request_count=request_count,
        is_empty=request_count <= 0,
        last_request=_dict_or_none(usage.get("last_request")),
        last_root=_dict_or_none(usage.get("last_root_request")),
    )


def _window_view(scope: Scope, now: datetime) -> UsageView:
    records = load_global_usage_records(since=scope.window, now=now, warn=True)
    return _window_view_from_records(scope, records, now)


def _filter_records(records: list[dict], since: timedelta | None, now: datetime) -> list[dict]:
    """In-memory equivalent of the ``since`` cutoff in ``load_global_usage_records``."""
    if since is None:
        return list(records)
    cutoff = now - since
    out: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        created_at = parse_timestamp(str(record.get("created_at") or ""))
        if created_at is not None and created_at >= cutoff:
            out.append(record)
    return out


def build_window_views(now: datetime | None = None, *, warn: bool = False) -> dict[str, UsageView]:
    """Build every windowed scope (day/week/month/all) from a single records load.

    The shared global JSONL is read and parsed exactly once; each scope is then
    derived by an in-memory ``since`` filter rather than its own file read. This is
    what lets :class:`~jarv.usage_command.UsageScreen` warm its whole scope cache in
    one background pass, so switching time periods costs no I/O on the loop thread.
    The cheap per-session scope is excluded (it reads a small file on demand).
    """
    now = now or utc_now()
    records = load_global_usage_records(now=now, warn=warn)
    views: dict[str, UsageView] = {}
    for scope in SCOPES:
        if scope.key == "session":
            continue
        windowed = _filter_records(records, scope.window, now)
        views[scope.key] = _window_view_from_records(scope, windowed, now)
    return views


def build_usage_view(scope_key: str, *, ctx=None, now: datetime | None = None) -> UsageView:
    """Build the :class:`UsageView` for a scope from the shared usage data layer."""
    now = now or utc_now()
    scope = resolve_scope(scope_key)
    if scope.key == "session":
        return _session_view(scope, ctx)
    return _window_view(scope, now)
