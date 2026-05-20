"""Token usage command rendering."""

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .display import console, jarv_panel, section_rule
from .history import load_history, prepare_session_context
from .usage import (
    estimate_token_cost_usd,
    format_cost,
    format_int,
    known_context_window,
    load_usage,
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


def _breakdown_section(breakdown: dict) -> Group:
    total = sum(int(breakdown.get(k, 0)) for k in _BREAKDOWN_KEYS)
    bar = _breakdown_bar(breakdown)

    bd_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    bd_table.add_column(no_wrap=True, width=1)
    bd_table.add_column(no_wrap=True)
    bd_table.add_column(justify="right", no_wrap=True)
    bd_table.add_column(justify="right", style="dim", no_wrap=True, width=5)

    for key in _BREAKDOWN_KEYS:
        count = int(breakdown.get(key, 0))
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
    color = _fill_color(percent)
    line = Text()
    line.append(f"{percent:5.1f}% full", style=f"bold {color}")
    line.append("  ")
    line.append_text(_smooth_bar(percent, width=32, color=color))
    line.append("  ")
    line.append(f"({format_int(input_tokens)} / {format_int(context_window)})", style="dim")
    return line


def _estimated_total_cost(usage: dict) -> float | None:
    models = usage.get("models") if isinstance(usage.get("models"), dict) else {}
    total = 0.0
    saw_model = False
    for model, bucket in models.items():
        if not isinstance(bucket, dict):
            continue
        if int(bucket.get("request_count") or 0) <= 0:
            continue
        saw_model = True
        estimate = estimate_token_cost_usd(bucket, str(model))
        if estimate is None:
            return None
        total += estimate
    if saw_model:
        return total
    return None


def cmd_usage() -> None:
    ctx = prepare_session_context()
    usage_path = usage_file_for(ctx.history_file)
    usage = load_usage(usage_path, ctx.session_id)
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
    request_count = int(totals.get("request_count") or 0)
    if request_count <= 0:
        console.print("[dim]No token usage recorded for this session yet.[/dim]")
        return

    history = load_history(ctx.history_file)
    exchanges = sum(1 for item in history if isinstance(item, dict) and item.get("role") == "user")
    last_request = usage.get("last_request") if isinstance(usage.get("last_request"), dict) else None
    last_root = usage.get("last_root_request") if isinstance(usage.get("last_root_request"), dict) else None
    model = str((last_request or {}).get("model") or "unknown")
    estimated_cost = _estimated_total_cost(usage)

    root_model = str((last_root or {}).get("model") or "unknown")

    context_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    context_table.add_column("Field", style="dim", no_wrap=True)
    context_table.add_column("Value", no_wrap=False)
    context_table.add_row("Latest root model", Text(root_model, style="bold magenta"))
    context_table.add_row("Context usage", _context_usage_renderable(last_root))

    reasoning_tokens = int(totals.get("reasoning_output_tokens") or 0)
    token_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    token_table.add_column("Field", style="dim", no_wrap=True)
    token_table.add_column("Value", no_wrap=False)
    token_table.add_row("Last model", Text(model, style="bold magenta"))
    token_table.add_row("Messages", Text(_plural(exchanges, "exchange")))
    token_table.add_row("Requests", Text(_plural(request_count, "request")))
    token_table.add_row("Input tokens", Text(format_int(totals.get("input_tokens"))))
    token_table.add_row("Cached input", Text(format_int(totals.get("cached_input_tokens")), style="cyan"))
    token_table.add_row("New input", Text(format_int(totals.get("uncached_input_tokens"))))
    token_table.add_row("Output tokens", Text(format_int(totals.get("output_tokens"))))
    if reasoning_tokens:
        token_table.add_row("Reasoning output", Text(format_int(reasoning_tokens), style="green"))
    token_table.add_row("Total tokens", Text(format_int(totals.get("total_tokens")), style="bold"))
    token_table.add_row("Estimated cost", Text(format_cost(estimated_cost), style="bold green"))
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
            section_rule("context breakdown [dim](estimated)[/dim]"),
            Text(""),
            _breakdown_section(breakdown),
        ]
    panel_parts += [
        Text(""),
        section_rule("token totals"),
        Text(""),
        token_table,
    ]

    console.print(jarv_panel(Group(*panel_parts), title="usage", subtitle=str(usage_path)))


