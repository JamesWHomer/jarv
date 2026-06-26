"""Tests for the shared stale-edge frame wrap and its capability gate."""

import io

from rich.console import Console
from rich.panel import Panel

from jarv.tui_frame import EraseTrailingColumns, wrap_frame


def _render(renderable, *, width: int = 40) -> str:
    buffer = io.StringIO()
    Console(file=buffer, force_terminal=True, color_system=None, width=width).print(renderable)
    return buffer.getvalue()


def test_wrap_frame_wraps_with_erase_trailing_columns():
    framed = wrap_frame(Panel("hi"))
    assert isinstance(framed, EraseTrailingColumns)


def test_wrap_frame_emits_erase_eol_on_a_terminal():
    out = _render(wrap_frame(Panel("hi")))
    assert "\x1b[0K" in out  # erase-to-end-of-line before each row


def test_erase_eol_gate_suppresses_control(monkeypatch):
    monkeypatch.setenv("JARV_NO_ERASE_EOL", "1")
    out = _render(wrap_frame(Panel("hi")))
    assert "\x1b[0K" not in out


def test_erase_trailing_columns_is_noop_off_terminal():
    buffer = io.StringIO()
    # Not a terminal -> no control codes, content still renders.
    Console(file=buffer, force_terminal=False, width=40).print(wrap_frame(Panel("hi")))
    out = buffer.getvalue()
    assert "\x1b[0K" not in out
    assert "hi" in out
