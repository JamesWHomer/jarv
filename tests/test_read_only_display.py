import io

import pytest
from conftest import FakeLive, neutralize_tui_modes
from rich.console import Console
from rich.text import Text

from jarv import read_only_display


class TtyStdin:
    def isatty(self):
        return True


class NonTtyStdin:
    def isatty(self):
        return False


def _install_display_harness(monkeypatch, *, width=80, height=24, key="ENTER", force_terminal=True):
    FakeLive.instances = []
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=force_terminal, width=width, color_system=None)
    monkeypatch.setattr(read_only_display, "console", test_console)
    monkeypatch.setattr(read_only_display.sys, "stdin", TtyStdin())
    monkeypatch.setattr(read_only_display, "terminal_size", lambda *, console: (width, height))
    monkeypatch.setattr(read_only_display, "Live", FakeLive)

    def read_key_with_repeats():
        if key == "KeyboardInterrupt":
            raise KeyboardInterrupt
        return key, 1

    # The shared loop polls key_available before reading; one close key is enough
    # because on_key stops the loop on the first read.
    monkeypatch.setattr(read_only_display, "_read_key_with_repeats", read_key_with_repeats)
    monkeypatch.setattr(read_only_display, "_key_available", lambda: True)

    # Neutralise the loop's terminal-mode managers (no real terminal under test).
    neutralize_tui_modes(monkeypatch)
    return output


def test_fullscreen_uses_compact_overlay_for_short_output(monkeypatch):
    _install_display_harness(monkeypatch, height=20)

    read_only_display.show_read_only_command(
        Text("short"),
        title="test",
        config={"read_only_command_display": "fullscreen"},
        include_setup_nudge=False,
    )

    # Interactive views always render in the alternate screen buffer.
    assert FakeLive.instances[-1].kwargs["screen"] is True
    # The frame is wrapped in EraseTrailingColumns for the stale-edge fix.
    assert FakeLive.instances[-1].renderable.renderable.height is None


def test_fill_screen_uses_full_height_for_short_output(monkeypatch):
    _install_display_harness(monkeypatch, height=20)

    read_only_display.show_read_only_command(
        Text("short"),
        title="test",
        config={"read_only_command_display": "fullscreen"},
        include_setup_nudge=False,
        fill_screen=True,
    )

    assert FakeLive.instances[-1].kwargs["screen"] is True
    assert FakeLive.instances[-1].renderable.renderable.height == 20


def test_fullscreen_uses_scrollable_view_for_long_output(monkeypatch):
    _install_display_harness(monkeypatch, height=8)
    body = Text("\n".join(f"line {i}" for i in range(40)))

    read_only_display.show_read_only_command(
        body,
        title="test",
        config={"read_only_command_display": "fullscreen"},
        include_setup_nudge=False,
    )

    assert FakeLive.instances[-1].kwargs["screen"] is True


def test_print_mode_bypasses_live(monkeypatch):
    output = _install_display_harness(monkeypatch)

    read_only_display.show_read_only_command(
        Text("printed"),
        title="test",
        config={"read_only_command_display": "print"},
        include_setup_nudge=False,
    )

    assert FakeLive.instances == []
    assert "printed" in output.getvalue()


def test_print_mode_respects_max_width(monkeypatch):
    output = _install_display_harness(monkeypatch, width=120)

    read_only_display.show_read_only_command(
        Text("narrow"),
        title="test",
        config={"read_only_command_display": "print"},
        include_setup_nudge=False,
        max_width=40,
    )

    assert max(len(line) for line in output.getvalue().splitlines()) == 40


def test_non_tty_prints_even_when_fullscreen_requested(monkeypatch):
    output = _install_display_harness(monkeypatch, force_terminal=True)
    monkeypatch.setattr(read_only_display.sys, "stdin", NonTtyStdin())

    read_only_display.show_read_only_command(
        Text("non tty"),
        title="test",
        config={"read_only_command_display": "fullscreen"},
        include_setup_nudge=False,
    )

    assert FakeLive.instances == []
    assert "non tty" in output.getvalue()


def test_fullscreen_view_uses_max_width_and_custom_close_hint(monkeypatch):
    _install_display_harness(monkeypatch, width=120, height=20)

    read_only_display.show_read_only_command(
        Text("close me"),
        title="test",
        config={"read_only_command_display": "fullscreen"},
        include_setup_nudge=False,
        max_width=60,
        close_hint="q / Esc / Enter  Close",
    )

    # The frame is wrapped in EraseTrailingColumns; the panel is one level in.
    panel = FakeLive.instances[-1].renderable.renderable
    assert panel.width == 60
    assert "q / Esc / Enter  Close" in panel.renderable.renderables[-1].plain


@pytest.mark.parametrize("key", ["q", "ESC", "ENTER", "KeyboardInterrupt"])
def test_fullscreen_view_closes_on_expected_keys(monkeypatch, key):
    _install_display_harness(monkeypatch, key=key)

    read_only_display.show_read_only_command(
        Text("close me"),
        title="test",
        config={"read_only_command_display": "fullscreen"},
        include_setup_nudge=False,
    )

    assert FakeLive.instances[-1].kwargs["screen"] is True
    assert FakeLive.instances[-1].refresh_count == 1
