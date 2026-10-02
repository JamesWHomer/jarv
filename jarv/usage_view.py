"""Pure data -> view-model for the ``/usage`` screen.

No Rich, no I/O of its own beyond the shared :mod:`jarv.usage` data layer, so the
whole thing is unit-testable at fixed inputs. Both the interactive
:class:`jarv.usage_command.UsageScreen` and the static fallback render from the
:class:`UsageView` this module produces, so session and system-wide usage share a
single code path instead of the two ~80-line functions they used to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, tzinfo

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


# Trend buckets follow the user's wall clock, not UTC: a "day" is a local
# calendar day. ``None`` means the system local zone; tests pin it.
_LOCAL_TZ: tzinfo | None = None

# Windows up to this long chart hourly; longer ones chart daily.
_HOURLY_MAX_WINDOW = timedelta(days=2)


@dataclass(frozen=True)
class TrendBucket:
    """Aggregated spend/tokens for one local hour or calendar day."""

    start: datetime  # naive local wall-clock time
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
    trend: list[TrendBucket]
    request_count: int
    is_empty: bool
    last_request: dict | None
    last_root: dict | None
    # ``"hour"`` or ``"day"``: the granularity of ``trend`` (empty for session).
    trend_unit: str = "day"
    # Spend over the equally long window just before this one (``None`` when the
    # scope has no window or that window recorded nothing), and its short label.
    previous_spend: float | None = None
    previous_label: str = ""


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


def _trend_unit(scope: Scope) -> str:
    if scope.window is not None and scope.window <= _HOURLY_MAX_WINDOW:
        return "hour"
    return "day"


def _bucket_trend(records: list[dict], scope: Scope, now: datetime) -> list[TrendBucket]:
    """Group records into a contiguous run of local hours or days for the trend."""
    hourly = _trend_unit(scope) == "hour"

    def bucket_start(moment: datetime) -> datetime:
        local = moment.astimezone(_LOCAL_TZ).replace(tzinfo=None)
        if hourly:
            return local.replace(minute=0, second=0, microsecond=0)
        return datetime.combine(local.date(), time())

    by_bucket: dict[datetime, list[dict]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        timestamp = parse_timestamp(str(record.get("created_at") or ""))
        if timestamp is None:
            continue
        by_bucket.setdefault(bucket_start(timestamp), []).append(record)

    if not by_bucket:
        return []
    last = bucket_start(now)
    start = bucket_start(now - scope.window) if scope.window is not None else min(by_bucket)
    start = min(start, last)

    out: list[TrendBucket] = []
    step = timedelta(hours=1) if hourly else timedelta(days=1)
    cursor = start
    while cursor <= last:
        bucket_records = by_bucket.get(cursor)
        if bucket_records:
            totals = _dict(aggregate_usage_records(bucket_records).get("totals"))
            out.append(
                TrendBucket(
                    start=cursor,
                    spend_usd=float(usage_cost_summary(totals).get("total_usd") or 0.0),
                    total_tokens=int(totals.get("total_tokens") or 0),
                    request_count=int(totals.get("request_count") or 0),
                )
            )
        else:
            out.append(TrendBucket(start=cursor, spend_usd=0.0, total_tokens=0, request_count=0))
        cursor += step
    return out


def _window_short_label(window: timedelta) -> str:
    """``24h`` / ``7d`` / ``30d``: how the hero names the comparison window."""
    hours = int(window.total_seconds() // 3600)
    if hours > 48 and hours % 24 == 0:
        return f"{hours // 24}d"
    return f"{hours}h"


def _previous_spend(records: list[dict] | None) -> float | None:
    if not records:
        return None
    totals = _dict(aggregate_usage_records(records).get("totals"))
    return float(usage_cost_summary(totals).get("total_usd") or 0.0)


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
        trend=[],
        request_count=request_count,
        is_empty=request_count <= 0,
        last_request=_dict_or_none(usage.get("last_request")),
        last_root=last_root,
    )


def _window_view_from_records(
    scope: Scope,
    records: list[dict],
    now: datetime,
    previous: list[dict] | None = None,
) -> UsageView:
    """Derive a windowed :class:`UsageView` from records already filtered to ``scope``.

    ``previous`` holds the records from the equally long window just before it,
    for the hero's period-over-period spend comparison.
    """
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
        trend=_bucket_trend(records, scope, now),
        request_count=request_count,
        is_empty=request_count <= 0,
        last_request=_dict_or_none(usage.get("last_request")),
        last_root=_dict_or_none(usage.get("last_root_request")),
        trend_unit=_trend_unit(scope),
        previous_spend=_previous_spend(previous),
        previous_label=_window_short_label(scope.window) if scope.window is not None else "",
    )


def _split_windows(
    records: list[dict], window: timedelta | None, now: datetime
) -> tuple[list[dict], list[dict] | None]:
    """Split records into ``(current window, the equally long window before it)``."""
    if window is None:
        return list(records), None
    return (
        _filter_records(records, window, now),
        _filter_records(records, window * 2, now, until=now - window),
    )


def _window_view(scope: Scope, now: datetime) -> UsageView:
    since = scope.window * 2 if scope.window is not None else None
    records = load_global_usage_records(since=since, now=now, warn=True)
    current, previous = _split_windows(records, scope.window, now)
    return _window_view_from_records(scope, current, now, previous)


def _filter_records(
    records: list[dict],
    since: timedelta | None,
    now: datetime,
    *,
    until: datetime | None = None,
) -> list[dict]:
    """In-memory equivalent of the ``since`` cutoff in ``load_global_usage_records``.

    ``until`` (exclusive) additionally drops records at or after that moment.
    """
    if since is None and until is None:
        return list(records)
    cutoff = now - since if since is not None else None
    out: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        created_at = parse_timestamp(str(record.get("created_at") or ""))
        if created_at is None:
            continue
        if cutoff is not None and created_at < cutoff:
            continue
        if until is not None and created_at >= until:
            continue
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
        current, previous = _split_windows(records, scope.window, now)
        views[scope.key] = _window_view_from_records(scope, current, now, previous)
    return views


def build_usage_view(scope_key: str, *, ctx=None, now: datetime | None = None) -> UsageView:
    """Build the :class:`UsageView` for a scope from the shared usage data layer."""
    now = now or utc_now()
    scope = resolve_scope(scope_key)
    if scope.key == "session":
        return _session_view(scope, ctx)
    return _window_view(scope, now)
