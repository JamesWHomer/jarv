"""Temporary display helpers for read-only slash command output."""

from __future__ import annotations

import json
import sys

from rich import box
from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .command_input import _read_key_with_repeats, mouse_capture
from .config import (
    CONFIG_FILE,
    DEFAULT_CONFIG,
    LEGACY_READ_ONLY_COMMAND_DISPLAY_CHOICES,
    READ_ONLY_COMMAND_DISPLAY_CHOICES,
    is_setup_complete,
)
from .display import (
    PANEL_BORDER_STYLE,
    TITLE_STYLE,
    console,
    refresh_on_resize,
    rendered_text_lines,
    terminal_size,
)


def _config_display_mode(config: dict | None = None) -> str:
    value = None
    if config is not None:
        value = config.get("read_only_command_display")
    elif CONFIG_FILE.exists():
        try:
            loaded = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                value = loaded.get("read_only_command_display")
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            value = None

    if value in LEGACY_READ_ONLY_COMMAND_DISPLAY_CHOICES:
        return "fullscreen"
    if value in READ_ONLY_COMMAND_DISPLAY_CHOICES:
        return str(value)
    return str(DEFAULT_CONFIG["read_only_command_display"])


def _interactive_terminal() -> bool:
    isatty = getattr(sys.stdin, "isatty", None)
    return callable(isatty) and isatty() and console.is_terminal


def _command_panel(
    body: RenderableType,
    *,
    title: str,
    subtitle: str | None = None,
    width: int | None = None,
    height: int | None = None,
    padding: tuple[int, int] = (1, 2),
) -> Panel:
    return Panel(
        body,
        title=f"[{TITLE_STYLE}]jarv \u25b8 {title}[/{TITLE_STYLE}]",
        title_align="left",
        subtitle=f"[dim]{subtitle}[/dim]" if subtitle else None,
        subtitle_align="right",
        border_style=PANEL_BORDER_STYLE,
        box=box.ROUNDED,
        padding=padding,
        width=width,
        height=height,
    )


def _setup_nudge() -> Text | None:
    if is_setup_complete():
        return None
    return Text("Tip: run jarv /setup to configure your API key and get started.", style="dim")


def _with_optional_setup_nudge(body: RenderableType, *, include_setup_nudge: bool) -> RenderableType:
    if not include_setup_nudge:
        return body
    nudge = _setup_nudge()
    if nudge is None:
        return body
    return Group(nudge, Text(""), body)


def _is_close_key(key: str) -> bool:
    return key in ("ESC", "ENTER", "q", "Q")


# Number of non-content rows a compact overlay panel reserves: 2 panel borders
# plus a blank spacer and the "q/Esc/Enter close" hint line.
_COMPACT_CHROME_ROWS = 4


def _show_overlay(
    body: RenderableType,
    *,
    title: str,
    subtitle: str | None,
    max_width: int | None,
    close_hint: str,
) -> None:
    """Render a read-only command view in the alternate screen buffer.

    The alternate screen is the only reliable surface for an interactive,
    resizable panel: Rich repaints it from ``home`` every frame (see
    ``Live.process_renderables`` -> ``Control.home()`` when ``_alt_screen`` is
    set), so terminal resize/zoom can never leave stacked or torn frames the way
    inline rendering in the normal buffer does (relative cursor moves desync
    after the emulator reflows scrollback, and ghosts that scroll out of the
    viewport can't be erased at all).

    When the content fits, the panel is sized to its content. Otherwise it
    fills the screen and becomes scrollable.
    """
    offset = 0
    visual_cache: dict[int, list[Text]] = {}

    def _lines(width: int) -> list[Text]:
        width = max(1, width)
        cached = visual_cache.get(width)
        if cached is None:
            cached = rendered_text_lines(body, width) or [Text("  (empty)", style="dim")]
            visual_cache[width] = cached
        return cached

    def _body_rows(term_h: int) -> int:
        show_footer = term_h >= 6
        return max(1, term_h - 2 - (2 if show_footer else 0))

    def _is_compact(term_w: int, term_h: int) -> bool:
        panel_width = min(term_w, max_width) if max_width else term_w
        inner_width = max(1, panel_width - 4)
        return len(_lines(inner_width)) + _COMPACT_CHROME_ROWS <= term_h

    def _render() -> Panel:
        nonlocal offset
        term_w, term_h = terminal_size(console=console)
        panel_width = max(1, min(term_w, max_width) if max_width else term_w)
        inner_width = max(1, panel_width - 4)
        lines = _lines(inner_width)
        total = len(lines)

        if _is_compact(term_w, term_h):
            offset = 0
            parts: list[Text] = list(lines)
            parts.append(Text(""))
            parts.append(Text(close_hint, style="dim italic", no_wrap=True, overflow="crop"))
            return _command_panel(
                Group(*parts),
                title=title,
                subtitle=subtitle,
                width=panel_width,
                padding=(0, 1),
            )

        show_footer = term_h >= 6
        body_rows = _body_rows(term_h)
        max_off = max(0, total - body_rows)
        offset = max(0, min(offset, max_off))
        start = offset
        end = min(total, start + body_rows)

        parts = list(lines[start:end])
        if show_footer:
            target_rows_before_footer = max(0, term_h - 2 - 2)
            while len(parts) < target_rows_before_footer:
                parts.append(Text(""))
            position = f"{start + 1}-{end} of {total}" if total else "0"
            parts.append(Text(""))
            parts.append(
                Text(
                    f"Up/Down scroll   PgUp/PgDn   Home/End   {close_hint}   .   {position}",
                    style="dim italic",
                    no_wrap=True,
                    overflow="crop",
                )
            )

        return _command_panel(
            Group(*parts),
            title=title,
            subtitle=subtitle,
            width=panel_width,
            height=max(3, term_h),
            padding=(0, 1),
        )

    live = Live(
        get_renderable=_render,
        console=console,
        screen=True,
        auto_refresh=False,
        transient=False,
        vertical_overflow="crop",
    )

    def _on_resize() -> None:
        # Clear the alternate screen before repainting so a mode switch
        # (compact <-> scrollable) on resize can never leave residual rows.
        with live._lock:
            try:
                console.clear()
            except Exception:
                pass
            live.refresh()

    with live, refresh_on_resize(live, on_change=_on_resize), mouse_capture():
        while True:
            live.refresh()
            try:
                key, repeat_count = _read_key_with_repeats()
            except KeyboardInterrupt:
                break
            term_w, term_h = terminal_size(console=console)
            panel_width = min(term_w, max_width) if max_width else term_w
            total = len(_lines(max(1, panel_width - 4)))
            body_rows = _body_rows(term_h)
            page = max(1, body_rows - 1)
            max_off = max(0, total - body_rows)
            if _is_close_key(key):
                break
            if key == "UP":
                offset = max(0, offset - repeat_count)
            elif key == "DOWN":
                offset = min(max_off, offset + repeat_count)
            elif key == "PAGEUP":
                offset = max(0, offset - (page * repeat_count))
            elif key == "PAGEDOWN":
                offset = min(max_off, offset + (page * repeat_count))
            elif key == "HOME":
                offset = 0
            elif key == "END":
                offset = max_off


def show_read_only_command(
    body: RenderableType,
    *,
    title: str,
    subtitle: str | None = None,
    config: dict | None = None,
    mode: str | None = None,
    include_setup_nudge: bool = True,
    max_width: int | None = None,
    close_hint: str = "q/Esc/Enter close",
) -> None:
    """Display read-only command output permanently or in a temporary view."""
    body = _with_optional_setup_nudge(body, include_setup_nudge=include_setup_nudge)
    selected_mode = mode if mode in READ_ONLY_COMMAND_DISPLAY_CHOICES else _config_display_mode(config)

    if selected_mode == "print" or not _interactive_terminal():
        console.print(_command_panel(body, title=title, subtitle=subtitle, width=max_width))
        return

    # Fullscreen uses the alternate screen buffer. Short content keeps a
    # compact panel; longer content fills the screen and becomes scrollable.
    _show_overlay(
        body,
        title=title,
        subtitle=subtitle,
        max_width=max_width,
        close_hint=close_hint,
    )
