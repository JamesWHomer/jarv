"""Tests for ScrollOverlayApp -- the read-only scroll overlay on the shared loop.

This replaced the old threaded run_scroll_live driver (Rich Live + a
jarv-resize-refresh daemon grabbing live._lock + a SIGWINCH handler). These
tests drive the real single-threaded loop with scripted input.
"""

import io
from contextlib import contextmanager
from unittest import mock

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import jarv.tui_app as tui_app
from jarv.tui_overlay import (
    ScrollOverlayState,
    apply_scroll_keys,
    body_content_rows,
    run_scroll_live,
)


@contextmanager
def _null(*_args, **_kwargs):
    yield


def _drive(keys, *, on_key=None, close_keys=frozenset({"ESC"})):
    """Run a ScrollOverlayApp with scripted keys; return (state, render_offsets)."""
    state = ScrollOverlayState()
    lines = [Text(f"line {i}") for i in range(40)]
    rendered: list[int] = []

    def render_panel():
        rendered.append(state.offset)
        body = "\n".join(line.plain for line in lines[state.offset : state.offset + 5])
        return Panel(Text(body), height=8, width=40)

    def _scroll(key, repeat, st):
        if on_key is not None:
            return on_key(key, repeat, st)
        body_rows, _ = body_content_rows(10)
        st.offset = apply_scroll_keys(
            key, repeat, offset=st.offset, total=len(lines), body_rows=body_rows
        )
        return False

    queue = list(keys)
    console = Console(file=io.StringIO(), force_terminal=True, color_system=None, width=40, height=10)

    with (
        mock.patch.object(tui_app, "raw_input_mode", _null),
        mock.patch.object(tui_app, "mouse_capture", _null),
        mock.patch.object(tui_app, "bracketed_paste", _null),
        mock.patch.object(tui_app, "windows_vt_input", _null),
        mock.patch.object(tui_app, "disable_mouse_capture", lambda *a, **k: None),
    ):
        run_scroll_live(
            render_panel,
            _scroll,
            state=state,
            close_keys=close_keys,
            console_ref=console,
            read_key_fn=lambda: queue.pop(0) if queue else ("ESC", 1),
            key_available_fn=lambda: True,
            terminal_size_fn=lambda *, console=None: (40, 10),
        )
    return state, rendered


def test_scroll_overlay_closes_on_close_key():
    state, rendered = _drive([("ESC", 1)])
    assert state.offset == 0
    assert rendered  # at least the initial paint happened


def test_scroll_overlay_scrolls_then_closes():
    state, rendered = _drive([("DOWN", 1), ("DOWN", 1), ("ESC", 1)])
    assert state.offset == 2  # two DOWNs advanced the offset
    assert rendered[-1] == 2  # last paint reflects the scrolled offset


def test_scroll_overlay_closes_on_keyboard_interrupt():
    def _raise():
        raise KeyboardInterrupt

    state = ScrollOverlayState()

    def render_panel():
        return Panel(Text("x"), height=8, width=40)

    console = Console(file=io.StringIO(), force_terminal=True, color_system=None, width=40, height=10)
    with (
        mock.patch.object(tui_app, "raw_input_mode", _null),
        mock.patch.object(tui_app, "mouse_capture", _null),
        mock.patch.object(tui_app, "bracketed_paste", _null),
        mock.patch.object(tui_app, "windows_vt_input", _null),
        mock.patch.object(tui_app, "disable_mouse_capture", lambda *a, **k: None),
    ):
        # Ctrl-C during the read closes the overlay via on_interrupt -> stop().
        run_scroll_live(
            render_panel,
            lambda *a: False,
            state=state,
            console_ref=console,
            read_key_fn=_raise,
            key_available_fn=lambda: True,
            terminal_size_fn=lambda *, console=None: (40, 10),
        )
    # Returns (does not hang or raise) -- the loop exited cleanly.


def test_scroll_overlay_custom_on_key_can_close():
    # on_key returning True closes even for a non-close key.
    state, rendered = _drive(
        [("x", 1)], on_key=lambda key, repeat, st: True, close_keys=frozenset()
    )
    assert rendered  # painted at least once before closing
