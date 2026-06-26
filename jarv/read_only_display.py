"""Temporary display helpers for read-only slash command output."""

from __future__ import annotations

import json
import sys

from rich.console import Group, RenderableType
from rich.live import Live
from rich.text import Text

from .command_input import _key_available, _read_key_with_repeats
from .config import (
    CONFIG_FILE,
    DEFAULT_CONFIG,
    LEGACY_READ_ONLY_COMMAND_DISPLAY_CHOICES,
    READ_ONLY_COMMAND_DISPLAY_CHOICES,
    is_setup_complete,
)
from .display import console, jarv_panel, rendered_text_lines, terminal_size
from .tui_overlay import scroll_overlay


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
    fill_screen: bool = False,
) -> None:
    """Display read-only command output permanently or in a temporary view."""
    body = _with_optional_setup_nudge(body, include_setup_nudge=include_setup_nudge)
    selected_mode = mode if mode in READ_ONLY_COMMAND_DISPLAY_CHOICES else _config_display_mode(config)

    if selected_mode == "print" or not _interactive_terminal():
        console.print(jarv_panel(body, title, subtitle=subtitle, width=max_width))
        return

    def lines_for_width(width: int) -> list[Text]:
        return rendered_text_lines(body, width)

    scroll_overlay(
        title=title,
        subtitle=subtitle,
        lines_for_width=lines_for_width,
        footer_hint=close_hint,
        max_width=max_width,
        fill_screen=fill_screen,
        console_ref=console,
        live_cls=Live,
        terminal_size_fn=terminal_size,
        read_key_fn=_read_key_with_repeats,
        key_available_fn=_key_available,
    )
