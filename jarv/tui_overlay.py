"""Fullscreen scrollable overlay helpers shared by jarv TUI views."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .command_input import _read_key_with_repeats, mouse_capture
from .display import console, jarv_panel, refresh_on_resize, terminal_size
from .tui_layout import append_bottom_footer


DEFAULT_CLOSE_KEYS = frozenset({"ESC", "ENTER", "q", "Q"})


@dataclass
class ScrollOverlayState:
    offset: int = 0
    extra: dict = field(default_factory=dict)


def body_content_rows(term_h: int, *, footer_rows: int = 2) -> tuple[int, bool]:
    """Return scrollable body row count and whether a footer is shown."""
    show_footer = term_h >= 6
    rows = max(1, term_h - 2 - (footer_rows if show_footer else 0))
    return rows, show_footer


def clamp_scroll_offset(offset: int, total: int, body_rows: int) -> int:
    max_off = max(0, total - body_rows)
    return max(0, min(offset, max_off))


def scroll_position_hint(start: int, end: int, total: int) -> str:
    if total:
        return f"{start + 1}-{end} of {total}"
    return "0"


def apply_scroll_keys(
    key: str,
    repeat_count: int,
    *,
    offset: int,
    total: int,
    body_rows: int,
) -> int:
    """Update scroll offset for standard navigation keys."""
    page = max(1, body_rows - 1)
    max_off = max(0, total - body_rows)
    if key == "UP":
        return max(0, offset - repeat_count)
    if key == "DOWN":
        return min(max_off, offset + repeat_count)
    if key == "PAGEUP":
        return max(0, offset - (page * repeat_count))
    if key == "PAGEDOWN":
        return min(max_off, offset + (page * repeat_count))
    if key == "HOME":
        return 0
    if key == "END":
        return max_off
    return offset


def run_scroll_live(
    render_panel: Callable[[], Panel],
    on_key: Callable[[str, int, ScrollOverlayState], bool],
    *,
    state: ScrollOverlayState | None = None,
    close_keys: frozenset[str] = DEFAULT_CLOSE_KEYS,
    console_ref: Any | None = None,
    live_cls: type | None = None,
    refresh_on_resize_fn: Callable[..., Any] | None = None,
    mouse_capture_fn: Callable[..., Any] | None = None,
    read_key_fn: Callable[[], tuple[str, int]] | None = None,
) -> None:
    """Run a fullscreen alternate-screen overlay with resize-safe refresh."""
    console_ref = console_ref or console
    live_cls = live_cls or Live
    refresh_on_resize_fn = refresh_on_resize_fn or refresh_on_resize
    mouse_capture_fn = mouse_capture_fn or mouse_capture
    read_key_fn = read_key_fn or _read_key_with_repeats

    scroll_state = state or ScrollOverlayState()
    live = live_cls(
        get_renderable=render_panel,
        console=console_ref,
        screen=True,
        auto_refresh=False,
        transient=False,
        vertical_overflow="crop",
    )

    def _on_resize() -> None:
        with live._lock:
            try:
                console_ref.clear()
            except Exception:
                pass
            live.refresh()

    with live, refresh_on_resize_fn(live, on_change=_on_resize), mouse_capture_fn():
        while True:
            live.refresh()
            try:
                key, repeat_count = read_key_fn()
            except KeyboardInterrupt:
                break
            if key in close_keys or on_key(key, repeat_count, scroll_state):
                break


def scroll_overlay(
    *,
    title: str,
    subtitle: str | None,
    lines_for_width: Callable[[int], list[Text]],
    footer_hint: str,
    max_width: int | None = None,
    fill_screen: bool = False,
    compact_chrome_rows: int = 4,
    close_keys: frozenset[str] = DEFAULT_CLOSE_KEYS,
    on_key: Callable[[str, int, ScrollOverlayState], bool] | None = None,
    console_ref: Any | None = None,
    live_cls: type | None = None,
    terminal_size_fn: Callable[..., tuple[int, int]] | None = None,
    refresh_on_resize_fn: Callable[..., Any] | None = None,
    mouse_capture_fn: Callable[..., Any] | None = None,
    read_key_fn: Callable[[], tuple[str, int]] | None = None,
) -> None:
    """Render a scrollable read-only overlay with shared chrome and key handling."""
    console_ref = console_ref or console
    terminal_size_fn = terminal_size_fn or terminal_size
    state = ScrollOverlayState()
    visual_cache: dict[int, list[Text]] = {}

    def _lines(width: int) -> list[Text]:
        width = max(1, width)
        cached = visual_cache.get(width)
        if cached is None:
            cached = lines_for_width(width) or [Text("  (empty)", style="dim")]
            visual_cache[width] = cached
        return cached

    def _is_compact(term_w: int, term_h: int) -> bool:
        panel_width = min(term_w, max_width) if max_width else term_w
        inner_width = max(1, panel_width - 4)
        return len(_lines(inner_width)) + compact_chrome_rows <= term_h

    def _render() -> Panel:
        term_w, term_h = terminal_size_fn(console=console_ref)
        panel_width = max(1, min(term_w, max_width) if max_width else term_w)
        inner_width = max(1, panel_width - 4)
        lines = _lines(inner_width)
        total = len(lines)

        if not fill_screen and _is_compact(term_w, term_h):
            state.offset = 0
            parts: list[Text] = list(lines)
            parts.append(Text(""))
            parts.append(Text(footer_hint, style="dim italic", no_wrap=True, overflow="crop"))
            return jarv_panel(
                Group(*parts),
                title,
                subtitle,
                padding=(0, 1),
                width=panel_width,
            )

        body_rows, show_footer = body_content_rows(term_h)
        state.offset = clamp_scroll_offset(state.offset, total, body_rows)
        start = state.offset
        end = min(total, start + body_rows)
        parts = list(lines[start:end])
        if show_footer:
            position = scroll_position_hint(start, end, total)
            append_bottom_footer(
                parts,
                term_h,
                Text(
                    f"Up/Down scroll   PgUp/PgDn   Home/End   {footer_hint}   .   {position}",
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                ),
            )

        return jarv_panel(
            Group(*parts),
            title,
            subtitle,
            padding=(0, 1),
            width=panel_width,
            height=max(3, term_h),
        )

    def _handle_key(key: str, repeat_count: int, scroll_state: ScrollOverlayState) -> bool:
        if on_key is not None and on_key(key, repeat_count, scroll_state):
            return True
        term_w, term_h = terminal_size_fn(console=console_ref)
        if not fill_screen and _is_compact(term_w, term_h):
            return False
        panel_width = min(term_w, max_width) if max_width else term_w
        total = len(_lines(max(1, panel_width - 4)))
        body_rows, _ = body_content_rows(term_h)
        scroll_state.offset = apply_scroll_keys(
            key,
            repeat_count,
            offset=scroll_state.offset,
            total=total,
            body_rows=body_rows,
        )
        return False

    run_scroll_live(
        _render,
        _handle_key,
        state=state,
        close_keys=close_keys,
        console_ref=console_ref,
        live_cls=live_cls,
        refresh_on_resize_fn=refresh_on_resize_fn,
        mouse_capture_fn=mouse_capture_fn,
        read_key_fn=read_key_fn,
    )
