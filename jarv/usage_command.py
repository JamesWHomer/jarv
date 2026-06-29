"""Token usage command: one interactive screen with a live scope switcher.

The session and system-wide code paths are unified behind a single
:class:`~jarv.usage_view.UsageView` (built in :mod:`jarv.usage_view`) and a single
body renderer (:func:`build_usage_body`) used by both the interactive
:class:`UsageScreen` and the static fallback. The screen leads with hero stats
(spend, tokens, requests, context headroom), a daily-spend trend, and a by-model
bar chart; ``←/→`` (or ``1-5`` / ``s t w m a``) cycles the scope live.
"""

from __future__ import annotations

import threading

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .display import (
    console,
    jarv_panel,
    rendered_text_lines,
    section_rule,
    terminal_size,
)
from .read_only_display import interactive_terminal, read_only_display_mode
from .tui_app import AltScreenApp
from .tui_frame import panel_width, wrap_frame
from .tui_layout import append_bottom_footer
from .tui_overlay import (
    apply_scroll_keys,
    body_content_rows,
    clamp_scroll_offset,
    scroll_position_hint,
)
from .usage import (
    format_cost,
    format_int,
    format_tokens_compact,
    known_context_window,
    usage_cost_summary,
)
from .usage_view import (
    SCOPE_KEYS,
    SCOPES,
    UsageView,
    build_usage_view,
    build_window_views,
    parse_usage_scope,
    resolve_scope,
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
# Vertical block glyphs for the daily-spend sparkline (index 0 is a blank day).
_SPARK_CHARS = " ▁▂▃▄▅▆▇█"
_WEEKDAY_INITIALS = "MTWTFSS"
_DAILY_MAX_BARS = 30


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


def _compact_cost(bucket: dict) -> Text:
    """A tight per-row cost: just the dollar figure (provenance lives in the hero)."""
    summary = usage_cost_summary(bucket)
    if summary["has_tracked_cost"] or summary["exact_requests"] or summary["estimated_requests"]:
        return Text(format_cost(summary["total_usd"]), style="bold green")
    if summary["contract_requests"]:
        return Text("contract", style="yellow")
    return Text("—", style="dim")


# --------------------------------------------------------------------------- #
# Pure render helpers (UsageView -> renderable)
# --------------------------------------------------------------------------- #
def _scope_tabs(active_key: str) -> Text:
    """Segmented control across the five scopes; the active tab is bracketed cyan."""
    line = Text(no_wrap=True, overflow="crop")
    if active_key not in SCOPE_KEYS:
        # Ad-hoc --all --since window (back-compat CLI shortcut, static only).
        line.append(f"‹ {resolve_scope(active_key).tab_label} ›", style="bold cyan")
        line.append("   ")
    for index, scope in enumerate(SCOPES):
        if index:
            line.append("   ")
        if scope.key == active_key:
            line.append(f"‹ {scope.tab_label} ›", style="bold cyan")
        else:
            line.append(scope.tab_label, style="dim")
    return line


def _cost_tag(cost: dict) -> Text:
    """A tiny provenance tag for the hero spend value."""
    if cost.get("exact_requests") and cost.get("estimated_requests"):
        return Text("~mixed", style="dim")
    if cost.get("estimated_requests"):
        return Text("~estimated", style="dim")
    if cost.get("exact_requests"):
        return Text("exact", style="dim")
    if cost.get("unknown_requests"):
        return Text("unknown", style="yellow")
    if cost.get("contract_requests"):
        return Text("contract", style="yellow")
    return Text("")


def _hero_context(context: dict) -> Group:
    percent = float(context.get("percent") or 0.0)
    window = int(context.get("window") or 0)
    color = _fill_color(percent)
    value = Text(f"{percent:.0f}% · {format_tokens_compact(window)}", style=f"bold {color}")
    return Group(value, _smooth_bar(percent, width=14, color=color))


def _hero_band(view: UsageView) -> Table:
    """SPEND / TOKENS / REQUESTS (+ CONTEXT in session scope) as big hero stats."""
    cost = view.cost
    has_cost = bool(
        cost.get("has_tracked_cost")
        or cost.get("exact_requests")
        or cost.get("estimated_requests")
    )
    table = Table(box=None, show_header=True, header_style="dim", padding=(0, 3), pad_edge=False)
    table.add_column("SPEND", no_wrap=True)
    table.add_column("TOKENS", no_wrap=True)
    table.add_column("REQUESTS", no_wrap=True)

    spend_value = Text(
        format_cost(cost.get("total_usd")) if has_cost else "—",
        style="bold green" if has_cost else "dim",
    )
    cells = [
        Group(spend_value, _cost_tag(cost)),
        Group(Text(format_tokens_compact(view.totals.get("total_tokens")), style="bold"), Text("")),
        Group(Text(format_int(view.request_count), style="bold"), Text("")),
    ]
    if view.context is not None:
        table.add_column("CONTEXT", no_wrap=True)
        cells.append(_hero_context(view.context))
    table.add_row(*cells)
    return table


def _daily_chart(view: UsageView) -> Group | None:
    """A vertical sparkline of daily spend, shown for windows with >=2 days of data."""
    days = view.daily
    if len(days) < 2:
        return None
    max_spend = max((day.spend_usd for day in days), default=0.0)
    if max_spend <= 0:
        return None

    shown = days[-_DAILY_MAX_BARS:]
    # Spaced bars (with weekday ticks) read best for a short window; a long one
    # packs the glyphs contiguously so the line never overflows the panel.
    spaced = len(shown) <= 14
    gap = " " if spaced else ""
    bars = Text("  ", no_wrap=True, overflow="crop")
    for day in shown:
        if day.spend_usd <= 0:
            glyph = " "
        else:
            step = int(round((day.spend_usd / max_spend) * (len(_SPARK_CHARS) - 1)))
            glyph = _SPARK_CHARS[max(1, min(len(_SPARK_CHARS) - 1, step))]
        bars.append(glyph + gap, style="cyan")

    spend_range = f"{format_cost(0)} – {format_cost(max_spend)}"
    parts: list = [Text("Daily spend", style="bold")]
    if spaced:
        bars.append("   ")
        bars.append(spend_range, style="dim")
        parts.append(bars)
        ticks = Text("  ", no_wrap=True, overflow="crop")
        for day in shown:
            ticks.append(_WEEKDAY_INITIALS[day.day.weekday()] + " ", style="dim")
        parts.append(ticks)
    else:
        parts.append(bars)
        date_range = f"{shown[0].day.strftime('%b %d')} – {shown[-1].day.strftime('%b %d')}"
        parts.append(Text(f"  {date_range}   ·   {spend_range}", style="dim"))
    return Group(*parts)


def _model_bars(view: UsageView) -> Group:
    """Per-model token-share bars with compact tokens and cost; top 6 then '+k more'."""
    total_tokens = int(view.totals.get("total_tokens") or 0)
    top = view.models[:6]
    table = Table(box=None, show_header=False, padding=(0, 1), pad_edge=False)
    table.add_column("Model", no_wrap=True, overflow="ellipsis", width=20)
    table.add_column("Bar", no_wrap=True)
    table.add_column("Tokens", justify="right", no_wrap=True)
    table.add_column("Cost", justify="right", no_wrap=True)
    for name, bucket in top:
        tokens = int(bucket.get("total_tokens") or 0)
        share = (tokens / total_tokens * 100) if total_tokens else 0.0
        table.add_row(
            Text(name, style="bold magenta"),
            _smooth_bar(share, width=14, color="cyan"),
            Text(format_tokens_compact(tokens)),
            _compact_cost(bucket),
        )
    extra = len(view.models) - len(top)
    if extra > 0:
        return Group(table, Text(f"  + {extra} more", style="dim"))
    return Group(table)


def _secondary_facts(view: UsageView) -> Text | None:
    """One dim line: providers · tiers · root/subagent request split."""
    segments: list[str] = []
    providers = [str(name) for name in view.providers if name and name != "unknown"]
    if len(providers) == 1:
        segments.append(providers[0])
    elif len(providers) > 1:
        segments.append(f"{len(providers)} providers")
    tiers = [str(name) for name in view.tiers if name]
    if len(tiers) == 1:
        segments.append(tiers[0])
    elif len(tiers) > 1:
        segments.append(f"{len(tiers)} tiers")
    source_bits: list[str] = []
    for label in ("root", "subagent"):
        bucket = view.sources.get(label)
        count = int(bucket.get("request_count") or 0) if isinstance(bucket, dict) else 0
        if count:
            source_bits.append(f"{count} {label}")
    if source_bits:
        segments.append(" / ".join(source_bits))
    if not segments:
        return None
    return Text(" · ".join(segments), style="dim")


def _context_detail(view: UsageView) -> Group | None:
    """The demoted session context breakdown (estimated allocation), below the hero."""
    if view.scope_key != "session":
        return None
    last_root = view.last_root
    if not isinstance(last_root, dict):
        return None
    breakdown = last_root.get("context_breakdown")
    if not (isinstance(breakdown, dict) and any(int(breakdown.get(k, 0) or 0) for k in _BREAKDOWN_KEYS)):
        return None
    return Group(
        section_rule("context breakdown [dim](estimated allocation)[/dim]"),
        Text(""),
        _context_usage_renderable(last_root),
        Text(""),
        _breakdown_section(breakdown, input_tokens=int(last_root.get("input_tokens") or 0)),
    )


def _empty_state(view: UsageView) -> Text:
    if view.scope_key == "session":
        return Text("No token usage recorded for this session yet.", style="dim")
    if view.scope_key == "all":
        return Text("No system-wide usage recorded yet.", style="dim")
    return Text(f"No usage recorded for {view.window_label.lower()}.", style="dim")


def _usage_body_sections(view: UsageView) -> list:
    """Everything below the tab row, as a flat list of renderables."""
    if view.is_empty:
        return [_empty_state(view)]
    parts: list = [_hero_band(view)]
    chart = _daily_chart(view)
    if chart is not None:
        parts += [Text(""), chart]
    if view.models:
        parts += [Text(""), Text("By model", style="bold"), _model_bars(view)]
    facts = _secondary_facts(view)
    if facts is not None:
        parts += [Text(""), facts]
    detail = _context_detail(view)
    if detail is not None:
        parts += [Text(""), detail]
    return parts


def build_usage_body(view: UsageView) -> Group:
    """Single source of truth: tabs + hero + chart + model bars + facts + detail."""
    return Group(_scope_tabs(view.scope_key), Text(""), *_usage_body_sections(view))


# --------------------------------------------------------------------------- #
# Interactive screen
# --------------------------------------------------------------------------- #
_FOOTER_HINT = "←/→ scope   ↑/↓ scroll   q close"


class UsageScreen(AltScreenApp):
    """The interactive /usage view: fixed tab row + scrollable body + fixed footer.

    Scopes switch live (``←/→`` / ``1-5`` / ``s t w m a``); each built
    :class:`UsageView` is cached per scope so switching back is instant. The four
    windowed scopes (day/week/month/all) all read the same global JSONL, so on
    open a background thread reads it *once* and pre-builds every window view (see
    :meth:`_preload_window_views`). The first paint shows the cheap session scope
    immediately while that warms, so switching periods later costs no loop-thread
    I/O. A window scope visited before its preload lands renders a brief loading
    placeholder rather than blocking the loop.
    """

    use_mouse_capture = True
    use_bracketed_paste = False
    translate_mouse_wheel = True
    text_mode = False
    batch_text = False
    clear_on_resize = False
    first_paint_label = "usage"

    _DIGIT_SCOPES = {"1": "session", "2": "day", "3": "week", "4": "month", "5": "all"}
    _LETTER_SCOPES = {"s": "session", "t": "day", "w": "week", "m": "month", "a": "all"}

    def __init__(self, *, initial_scope: str = "session", now=None):
        super().__init__(console=console)
        self.scope_key = initial_scope
        self.offset = 0
        self._now = now
        self._cache: dict[str, UsageView] = {}
        self._cache_lock = threading.Lock()
        self._preload_thread: threading.Thread | None = None
        self._preload_done = threading.Event()

    def on_start(self) -> None:
        self._start_preload()

    def _start_preload(self) -> None:
        """Warm the windowed-scope cache off the loop thread (started once on open)."""
        if self._preload_thread is not None:
            return
        thread = threading.Thread(
            target=self._preload_window_views,
            name="usage-preload",
            daemon=True,
        )
        self._preload_thread = thread
        thread.start()

    def _preload_window_views(self) -> None:
        """Read the global JSONL once and cache all window views, then repaint."""
        try:
            views = build_window_views(now=self._now)
            with self._cache_lock:
                for key, view in views.items():
                    self._cache.setdefault(key, view)
        except Exception:
            pass
        finally:
            self._preload_done.set()
            self.invalidate()

    def _view(self) -> UsageView | None:
        """Current scope's view, or ``None`` while a window scope is still loading.

        The session scope is cheap, so it is built inline on demand. The windowed
        scopes are warmed in the background; until that lands they render a loading
        state (``None``) instead of blocking the loop on file I/O. Once the preload
        has finished, an inline build is the safety net for the rare case it didn't
        cache the scope (e.g. it raised).
        """
        with self._cache_lock:
            cached = self._cache.get(self.scope_key)
        if cached is not None:
            return cached
        if self.scope_key == "session" or self._preload_done.is_set():
            view = build_usage_view(self.scope_key, now=self._now)
            with self._cache_lock:
                self._cache.setdefault(self.scope_key, view)
                return self._cache[self.scope_key]
        return None

    def _geometry(self) -> tuple[int, int, int, int, bool]:
        _term_w, term_h = terminal_size(console=console)
        width = panel_width(_term_w)
        inner_width = max(1, width - 4)
        body_rows, show_footer = body_content_rows(term_h)
        body_rows = max(1, body_rows - 2)  # reserve the fixed tab row + spacer
        return term_h, width, inner_width, body_rows, show_footer

    def _body_lines(self, view: UsageView, inner_width: int) -> list[Text]:
        return rendered_text_lines(Group(*_usage_body_sections(view)), inner_width)

    def _loading_frame(self, term_h: int, width: int):
        """Placeholder shown while a window scope's data is still loading."""
        body = Group(
            _scope_tabs(self.scope_key),
            Text(""),
            Text("Loading usage…", style="dim"),
        )
        return wrap_frame(
            jarv_panel(
                body,
                "usage",
                subtitle=resolve_scope(self.scope_key).window_label,
                padding=(0, 1),
                width=width,
                height=term_h,
            )
        )

    def render(self):
        term_h, width, inner_width, body_rows, show_footer = self._geometry()
        view = self._view()
        if view is None:
            return self._loading_frame(term_h, width)
        lines = self._body_lines(view, inner_width)
        total = len(lines)
        self.offset = clamp_scroll_offset(self.offset, total, body_rows)
        start = self.offset
        end = min(total, start + body_rows)

        parts: list = [_scope_tabs(self.scope_key), Text("")]
        parts.extend(lines[start:end])
        if show_footer:
            position = scroll_position_hint(start, end, total)
            append_bottom_footer(
                parts,
                term_h,
                Text(
                    f"{_FOOTER_HINT}   ·   {position}",
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                ),
            )
        subtitle = f"{view.window_label} · {view.source_path}"
        return wrap_frame(
            jarv_panel(
                Group(*parts),
                "usage",
                subtitle=subtitle,
                padding=(0, 1),
                width=width,
                height=term_h,
            )
        )

    def on_interrupt(self) -> None:
        self.stop()

    def _set_scope(self, key: str) -> None:
        if key != self.scope_key:
            self.scope_key = key
            self.offset = 0

    def _switch_scope(self, delta: int, repeat: int) -> None:
        try:
            index = SCOPE_KEYS.index(self.scope_key)
        except ValueError:
            index = SCOPE_KEYS.index("all")
        index = max(0, min(len(SCOPE_KEYS) - 1, index + delta * max(1, repeat)))
        self._set_scope(SCOPE_KEYS[index])

    def _scroll(self, key: str, repeat: int) -> None:
        view = self._view()
        if view is None:
            return
        _term_h, _width, inner_width, body_rows, _show_footer = self._geometry()
        total = len(self._body_lines(view, inner_width))
        self.offset = apply_scroll_keys(
            key, repeat, offset=self.offset, total=total, body_rows=body_rows
        )

    def on_key(self, key: str, repeat: int) -> None:
        if key in ("q", "Q", "ESC", "ENTER"):
            self.stop()
            return
        if key in ("LEFT", "RIGHT"):
            self._switch_scope(-1 if key == "LEFT" else 1, repeat)
            return
        target = self._DIGIT_SCOPES.get(key) or self._LETTER_SCOPES.get(key)
        if target is not None:
            self._set_scope(target)
            return
        if key in ("UP", "DOWN", "PAGEUP", "PAGEDOWN", "HOME", "END"):
            self._scroll(key, repeat)


def cmd_usage(args: list[str] | None = None) -> None:
    scope_key, error = parse_usage_scope(args)
    if error is not None:
        console.print(jarv_panel(Text(error, style="yellow"), "usage"))
        return

    if scope_key in SCOPE_KEYS and interactive_terminal() and read_only_display_mode() != "print":
        UsageScreen(initial_scope=scope_key).run()
        return

    view = build_usage_view(scope_key)
    hint = Text("Scopes: session · day · week · month · all", style="dim italic")
    body = Group(build_usage_body(view), Text(""), hint)
    subtitle = f"{view.window_label} · {view.source_path}"
    console.print(jarv_panel(body, "usage", subtitle=subtitle))
