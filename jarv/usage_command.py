"""Token usage command rendering."""

from datetime import timedelta

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .display import section_rule
from .history import load_history, prepare_session_context
from .read_only_display import show_read_only_command
from .usage import (
    aggregate_usage_records,
    format_cost,
    format_int,
    global_usage_file,
    known_context_window,
    load_global_usage_records,
    load_usage,
    usage_cost_summary,
    usage_file_for,
)


_BREAKDOWN_KEYS = ("system", "tools", "history", "tool_io", "reasoning")
_BREAKDOWN_LABELS = {
    "system": "System",
    "tools": "Tools",
    "history": "History",
    "tool_io": "Tool I/O",
    "reasoning": "Reasoning",
}
_BREAKDOWN_COLORS = {
    "system": "white",
    "tools": "yellow",
    "history": "cyan",
    "tool_io": "magenta",
    "reasoning": "green",
}


_BAR_FILL_CHARS = " ▏▎▍▌▋▊▉█"


def _smooth_bar(percent: float | None, width: int = 36, color: str = "cyan") -> Text:
    """Render a smooth horizontal bar with sub-cell precision."""
    bar = Text()
    if percent is None:
        bar.append("─" * width, style="bright_black")
        return bar
    pct = max(0.0, min(percent, 100.0)) / 100
    total_eighths = pct * width * 8
    full = int(total_eighths // 8)
    remainder = int(total_eighths - full * 8)
    if full > width:
        full = width
        remainder = 0
    bar.append("█" * full, style=color)
    if full < width:
        if remainder:
            bar.append(_BAR_FILL_CHARS[remainder], style=color)
            empty = width - full - 1
        else:
            empty = width - full
        if empty > 0:
            bar.append("─" * empty, style="bright_black")
    return bar


def _fill_color(percent: float | None) -> str:
    if percent is None:
        return "bright_black"
    if percent >= 90:
        return "bright_red"
    if percent >= 70:
        return "yellow"
    if percent >= 40:
        return "cyan"
    return "green"


def _breakdown_bar(breakdown: dict, width: int = 48) -> Text:
    total = sum(int(breakdown.get(k, 0)) for k in _BREAKDOWN_KEYS)
    if total == 0:
        return Text("─" * width, style="bright_black")
    bar = Text()
    used = 0
    non_zero = [k for k in _BREAKDOWN_KEYS if int(breakdown.get(k, 0)) > 0]
    for i, key in enumerate(non_zero):
        count = int(breakdown.get(key, 0))
        is_last = i == len(non_zero) - 1
        if is_last:
            chars = width - used
        else:
            chars = max(1, round((count / total) * width))
            chars = min(chars, width - used - (len(non_zero) - i - 1))
        if chars > 0:
            bar.append("█" * chars, style=_BREAKDOWN_COLORS[key])
            used += chars
    if used < width:
        bar.append("─" * (width - used), style="bright_black")
    return bar


def _reconcile_breakdown(breakdown: dict, target_total: int) -> dict[str, int]:
    """Scale estimated categories so their integer sum matches recorded input."""
    estimates = {key: max(int(breakdown.get(key, 0)), 0) for key in _BREAKDOWN_KEYS}
    estimated_total = sum(estimates.values())
    if estimated_total <= 0 or target_total <= 0:
        return estimates

    scaled = {
        key: (estimates[key] * target_total) / estimated_total
        for key in _BREAKDOWN_KEYS
    }
    reconciled = {key: int(value) for key, value in scaled.items()}
    remainder = target_total - sum(reconciled.values())
    ranked = sorted(
        _BREAKDOWN_KEYS,
        key=lambda key: scaled[key] - reconciled[key],
        reverse=True,
    )
    for key in ranked[:remainder]:
        reconciled[key] += 1
    return reconciled


def _breakdown_section(breakdown: dict, *, input_tokens: int) -> Group:
    reconciled = _reconcile_breakdown(breakdown, input_tokens)
    total = sum(reconciled.values())
    bar = _breakdown_bar(reconciled)

    bd_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    bd_table.add_column(no_wrap=True, width=1)
    bd_table.add_column(no_wrap=True)
    bd_table.add_column(justify="right", no_wrap=True)
    bd_table.add_column(justify="right", style="dim", no_wrap=True, width=5)

    for key in _BREAKDOWN_KEYS:
        count = reconciled[key]
        pct = f"{round(count / total * 100)}%" if total > 0 else "—"
        bd_table.add_row(
            Text("●", style=_BREAKDOWN_COLORS[key]),
            Text(_BREAKDOWN_LABELS[key]),
            Text(format_int(count), style="bold"),
            Text(pct, style="dim"),
        )
    return Group(bar, Text(""), bd_table)


def _plural(value: int, singular: str, plural: str | None = None) -> str:
    word = singular if value == 1 else (plural or f"{singular}s")
    return f"{value:,} {word}"


def _context_usage_renderable(last_root: dict | None) -> Text:
    if not isinstance(last_root, dict):
        return Text("Unknown until a root request is recorded", style="dim")
    model = str(last_root.get("model") or "")
    context_window = known_context_window(model)
    input_tokens = int(last_root.get("input_tokens") or 0)
    if context_window is None:
        return Text("Unknown for this model", style="dim")
    percent = (input_tokens / context_window) * 100
    remaining = max(context_window - input_tokens, 0)
    color = _fill_color(percent)
    line = Text()
    line.append(f"{percent:.1f}% full", style=f"bold {color}")
    line.append("  ")
    line.append_text(_smooth_bar(percent, width=32, color=color))
    line.append("  ")
    line.append(f"({format_int(input_tokens)} / {format_int(context_window)})", style="dim")
    line.append(f" · {format_int(remaining)} remaining", style="dim")
    return line


def _cost_text(bucket: dict) -> Text:
    summary = usage_cost_summary(bucket)
    known_requests = summary["exact_requests"] + summary["estimated_requests"]
    text = Text()
    if known_requests or summary["has_tracked_cost"]:
        text.append(format_cost(summary["total_usd"]), style="bold green")
        if summary["exact_requests"] and summary["estimated_requests"]:
            text.append(" mixed", style="dim")
        elif summary["exact_requests"]:
            text.append(" exact", style="dim")
        elif summary["estimated_requests"]:
            text.append(" estimated", style="dim")
        else:
            text.append(" partial", style="dim")
    else:
        text.append("Unknown", style="yellow")
    if summary["unknown_requests"]:
        text.append(
            f" + {_plural(summary['unknown_requests'], 'unknown request')}",
            style="yellow",
        )
    if summary["contract_requests"]:
        text.append(
            f" + {_plural(summary['contract_requests'], 'contract-priced request')}",
            style="yellow",
        )
    return text


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


def _parse_global_usage_args(args: list[str] | None) -> tuple[bool, timedelta | None, str, str | None]:
    args = list(args or [])
    aliases = {
        "day": ("last 24h", timedelta(hours=24)),
        "week": ("last 7d", timedelta(days=7)),
        "month": ("last 30d", timedelta(days=30)),
    }
    if not args:
        return False, None, "", None
    if len(args) == 1 and args[0].lower() in aliases:
        label, since = aliases[args[0].lower()]
        return True, since, label, None
    if args[0] != "--all":
        return False, None, "", "Usage: jarv /usage [--all [--since 24h|7d|30d] | day|week|month]"
    if len(args) == 1:
        return True, None, "all time", None
    if len(args) == 3 and args[1] == "--since":
        since = _parse_since_value(args[2])
        if since is None:
            return False, None, "", "Usage: jarv /usage --all --since 24h|7d|30d"
        return True, since, f"last {args[2].lower()}", None
    if len(args) == 2 and args[1].startswith("--since="):
        raw = args[1].split("=", 1)[1]
        since = _parse_since_value(raw)
        if since is None:
            return False, None, "", "Usage: jarv /usage --all --since 24h|7d|30d"
        return True, since, f"last {raw.lower()}", None
    return False, None, "", "Usage: jarv /usage [--all [--since 24h|7d|30d] | day|week|month]"


def _token_totals_table(usage: dict, *, exchanges: int | None = None) -> Table:
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
    request_count = int(totals.get("request_count") or 0)
    reasoning_tokens = int(totals.get("reasoning_output_tokens") or 0)

    token_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    token_table.add_column("Field", style="dim", no_wrap=True)
    token_table.add_column("Value", no_wrap=False)
    if exchanges is not None:
        token_table.add_row("Exchanges", Text(_plural(exchanges, "exchange")))
    token_table.add_row("Requests", Text(_plural(request_count, "request")))
    token_table.add_row("Input tokens", Text(format_int(totals.get("input_tokens"))))
    token_table.add_row("Cached input", Text(format_int(totals.get("cached_input_tokens")), style="cyan"))
    token_table.add_row("New input", Text(format_int(totals.get("uncached_input_tokens"))))
    token_table.add_row("Output tokens", Text(format_int(totals.get("output_tokens"))))
    if reasoning_tokens:
        token_table.add_row("Reasoning output", Text(format_int(reasoning_tokens), style="green"))
    token_table.add_row("Total tokens", Text(format_int(totals.get("total_tokens")), style="bold"))
    token_table.add_row("Tracked cost", _cost_text(totals))
    return token_table


def _breakdown_table(buckets: dict, *, kind: str) -> Table:
    table = Table(box=None, show_header=True, padding=(0, 2), pad_edge=False, header_style="dim")
    table.add_column(kind, no_wrap=True)
    table.add_column("Requests", justify="right", no_wrap=True)
    table.add_column("Tokens", justify="right", no_wrap=True)
    table.add_column("Cost", justify="right", no_wrap=True)

    rows = sorted(
        ((name, bucket) for name, bucket in buckets.items() if isinstance(bucket, dict)),
        key=lambda item: int(item[1].get("total_tokens") or 0),
        reverse=True,
    )
    for name, bucket in rows:
        cells = [
            Text(str(name), style="bold magenta" if kind == "Model" else "bold cyan"),
            Text(format_int(bucket.get("request_count"))),
            Text(format_int(bucket.get("total_tokens")), style="bold"),
        ]
        cells.append(_cost_text(bucket))
        table.add_row(*cells)
    return table


def _cmd_global_usage(since: timedelta | None, window_label: str) -> None:
    usage_path = global_usage_file()
    records = load_global_usage_records(since=since, warn=True)
    usage = aggregate_usage_records(records)
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
    request_count = int(totals.get("request_count") or 0)
    subtitle = f"{usage_path} - {window_label}"
    if request_count <= 0:
        show_read_only_command(
            Text(f"No system-wide token usage recorded for {window_label}.", style="dim"),
            title="usage",
            subtitle=subtitle,
        )
        return

    last_request = usage.get("last_request") if isinstance(usage.get("last_request"), dict) else None
    overview = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    overview.add_column("Field", style="dim", no_wrap=True)
    overview.add_column("Value", no_wrap=False)
    overview.add_row("Scope", Text("System-wide", style="bold cyan"))
    overview.add_row("Window", Text(window_label))
    if last_request:
        last_model = str(last_request.get("model") or "unknown")
        overview.add_row("Last model", Text(last_model, style="bold magenta"))
        overview.add_row(
            "Last provider",
            Text(str(last_request.get("provider") or "unknown"), style="bold cyan"),
        )
        requested_tier = str(last_request.get("requested_service_tier") or "standard")
        served_tier = str(last_request.get("served_service_tier") or "")
        tier_label = (
            f"{requested_tier} -> {served_tier}"
            if served_tier and served_tier != requested_tier
            else served_tier or requested_tier
        )
        overview.add_row("Last tier", Text(tier_label, style="bold"))

    panel_parts: list = [
        section_rule("system-wide overview"),
        Text(""),
        overview,
        Text(""),
        section_rule("token totals"),
        Text(""),
        _token_totals_table(usage),
    ]
    sources = usage.get("sources") if isinstance(usage.get("sources"), dict) else {}
    if sources:
        panel_parts += [
            Text(""),
            section_rule("by source"),
            Text(""),
            _breakdown_table(sources, kind="Source"),
        ]
    providers = usage.get("providers") if isinstance(usage.get("providers"), dict) else {}
    if providers:
        panel_parts += [
            Text(""),
            section_rule("by provider"),
            Text(""),
            _breakdown_table(providers, kind="Provider"),
        ]
    tiers = usage.get("tiers") if isinstance(usage.get("tiers"), dict) else {}
    if tiers:
        panel_parts += [
            Text(""),
            section_rule("by processing tier"),
            Text(""),
            _breakdown_table(tiers, kind="Tier"),
        ]
    models = usage.get("models") if isinstance(usage.get("models"), dict) else {}
    if models:
        panel_parts += [
            Text(""),
            section_rule("by model"),
            Text(""),
            _breakdown_table(models, kind="Model"),
        ]

    show_read_only_command(Group(*panel_parts), title="usage", subtitle=subtitle)


def cmd_usage(args: list[str] | None = None) -> None:
    is_global, since, window_label, error = _parse_global_usage_args(args)
    if error is not None:
        show_read_only_command(Text(error, style="yellow"), title="usage")
        return
    if is_global:
        _cmd_global_usage(since, window_label)
        return

    ctx = prepare_session_context()
    usage_path = usage_file_for(ctx.history_file)
    usage = load_usage(usage_path, ctx.session_id)
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
    request_count = int(totals.get("request_count") or 0)
    if request_count <= 0:
        show_read_only_command(
            Text("No token usage recorded for this session yet.", style="dim"),
            title="usage",
            subtitle=str(usage_path),
        )
        return

    history = load_history(ctx.history_file)
    exchanges = sum(1 for item in history if isinstance(item, dict) and item.get("role") == "user")
    last_request = usage.get("last_request") if isinstance(usage.get("last_request"), dict) else None
    last_root = usage.get("last_root_request") if isinstance(usage.get("last_root_request"), dict) else None
    root_model = str((last_root or {}).get("model") or "unknown")

    context_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    context_table.add_column("Field", style="dim", no_wrap=True)
    context_table.add_column("Value", no_wrap=False)
    context_table.add_row("Current model", Text(root_model, style="bold magenta"))
    if last_root:
        context_table.add_row(
            "Current provider",
            Text(str(last_root.get("provider") or "unknown"), style="bold cyan"),
        )
        requested_tier = str(last_root.get("requested_service_tier") or "standard")
        served_tier = str(last_root.get("served_service_tier") or "")
        tier_label = (
            f"{requested_tier} -> {served_tier}"
            if served_tier and served_tier != requested_tier
            else served_tier or requested_tier
        )
        context_table.add_row("Processing tier", Text(tier_label, style="bold"))
    context_table.add_row("Context usage", _context_usage_renderable(last_root))

    token_table = _token_totals_table(usage, exchanges=exchanges)
    if last_request is not None:
        last_line = Text()
        last_line.append(format_int(last_request.get("input_tokens")), style="bold")
        last_line.append(" in ", style="dim")
        last_line.append("(", style="dim")
        last_line.append(format_int(last_request.get("cached_input_tokens")), style="cyan")
        last_line.append(" cached", style="dim")
        last_line.append(") · ", style="dim")
        last_line.append(format_int(last_request.get("output_tokens")), style="bold")
        last_line.append(" out", style="dim")
        token_table.add_row("Last request", last_line)

    breakdown = (last_root or {}).get("context_breakdown")
    panel_parts: list = [
        section_rule("session overview"),
        Text(""),
        context_table,
    ]
    if isinstance(breakdown, dict) and any(breakdown.get(k, 0) for k in _BREAKDOWN_KEYS):
        panel_parts += [
            Text(""),
            section_rule("context breakdown [dim](estimated allocation)[/dim]"),
            Text(""),
            _breakdown_section(
                breakdown,
                input_tokens=int((last_root or {}).get("input_tokens") or 0),
            ),
        ]
    panel_parts += [
        Text(""),
        section_rule("token totals"),
        Text(""),
        token_table,
    ]

    show_read_only_command(Group(*panel_parts), title="usage", subtitle=str(usage_path))


