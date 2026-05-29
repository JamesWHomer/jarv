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
from .config import CONFIG_FILE, DEFAULT_CONFIG, READ_ONLY_COMMAND_DISPLAY_CHOICES, is_setup_complete
from .display import (
    PANEL_BORDER_STYLE,
    TITLE_STYLE,
    console,
    refresh_on_resize,
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


def _transient_body(body: RenderableType) -> RenderableType:
    return Group(body, Text(""), Text("q/Esc/Enter close", style="dim italic", no_wrap=True, overflow="crop"))


def _rendered_text_lines(renderable: RenderableType, width: int) -> list[Text]:
    options = console.options.update(width=max(1, width))
    rendered = console.render_lines(renderable, options, pad=False)
    lines: list[Text] = []
    for rendered_line in rendered:
        line = Text(no_wrap=True, overflow="crop")
        for segment in rendered_line:
            if segment.text:
                line.append(segment.text, style=segment.style)
        lines.append(line)
    return lines


def _rendered_line_count(renderable: RenderableType, width: int) -> int:
    return len(_rendered_text_lines(renderable, width))


def _auto_mode(body: RenderableType, *, title: str, subtitle: str | None) -> str:
    term_w, term_h = terminal_size(console=console)
    width = max(1, term_w)
    panel = _command_panel(_transient_body(body), title=title, subtitle=subtitle, width=width, padding=(0, 1))
    return "inline" if _rendered_line_count(panel, width) <= max(3, term_h) else "fullscreen"


def _is_close_key(key: str) -> bool:
    return key in ("ESC", "ENTER", "q", "Q")


def _show_inline(body: RenderableType, *, title: str, subtitle: str | None) -> None:
    def _render() -> Panel:
        term_w, _ = terminal_size(console=console)
        return _command_panel(
            _transient_body(body),
            title=title,
            subtitle=subtitle,
            width=max(1, term_w),
            padding=(0, 1),
        )

    with Live(
        get_renderable=_render,
        console=console,
        screen=False,
        auto_refresh=False,
        transient=True,
        vertical_overflow="crop",
    ) as live, refresh_on_resize(live):
        while True:
            live.refresh()
            try:
                key, _repeat_count = _read_key_with_repeats()
            except KeyboardInterrupt:
                break
            if _is_close_key(key):
                break


def _show_fullscreen(body: RenderableType, *, title: str, subtitle: str | None) -> None:
    offset = 0
    visual_cache: dict[int, list[Text]] = {}

    def _lines(width: int) -> list[Text]:
        width = max(1, width)
        cached = visual_cache.get(width)
        if cached is None:
            cached = _rendered_text_lines(body, width) or [Text("  (empty)", style="dim")]
            visual_cache[width] = cached
        return cached

    def _body_rows() -> int:
        _, term_h = terminal_size(console=console)
        show_footer = term_h >= 6
        return max(1, term_h - 2 - (2 if show_footer else 0))

    def _render() -> Panel:
        nonlocal offset
        term_w, term_h = terminal_size(console=console)
        panel_width = max(1, term_w)
        show_footer = term_h >= 6
        body_rows = _body_rows()
        inner_width = max(1, panel_width - 4)
        lines = _lines(inner_width)
        total = len(lines)
        max_off = max(0, total - body_rows)
        offset = max(0, min(offset, max_off))
        start = offset
        end = min(total, start + body_rows)

        parts: list[Text] = []
        parts.extend(lines[start:end])

        if show_footer:
            target_rows_before_footer = max(0, term_h - 2 - 2)
            while len(parts) < target_rows_before_footer:
                parts.append(Text(""))
            position = f"{start + 1}-{end} of {total}" if total else "0"
            parts.append(Text(""))
            parts.append(
                Text(
                    f"Up/Down scroll   PgUp/PgDn   Home/End   q/Esc/Enter close   .   {position}",
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

    with Live(
        get_renderable=_render,
        console=console,
        screen=True,
        auto_refresh=False,
        transient=False,
        vertical_overflow="crop",
    ) as live, refresh_on_resize(live), mouse_capture():
        while True:
            live.refresh()
            try:
                key, repeat_count = _read_key_with_repeats()
            except KeyboardInterrupt:
                break
            term_w, _ = terminal_size(console=console)
            total = len(_lines(max(1, term_w - 4)))
            page = max(1, _body_rows() - 1)
            max_off = max(0, total - _body_rows())
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
) -> None:
    """Display read-only command output permanently or in a temporary view."""
    body = _with_optional_setup_nudge(body, include_setup_nudge=include_setup_nudge)
    selected_mode = mode if mode in READ_ONLY_COMMAND_DISPLAY_CHOICES else _config_display_mode(config)

    if selected_mode == "print" or not _interactive_terminal():
        console.print(_command_panel(body, title=title, subtitle=subtitle))
        return

    if selected_mode == "auto":
        selected_mode = _auto_mode(body, title=title, subtitle=subtitle)

    if selected_mode == "fullscreen":
        _show_fullscreen(body, title=title, subtitle=subtitle)
    else:
        _show_inline(body, title=title, subtitle=subtitle)
