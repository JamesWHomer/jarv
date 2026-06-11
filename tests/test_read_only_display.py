import io
from contextlib import contextmanager

import pytest
from rich.console import Console
from rich.text import Text

from jarv import read_only_display


class TtyStdin:
    def isatty(self):
        return True


class NonTtyStdin:
    def isatty(self):
        return False


@contextmanager
def noop_context(*_args, **_kwargs):
    yield


class FakeLive:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.refresh_count = 0
        FakeLive.instances.append(self)

    def __enter__(self):
        self._render_once()
        return self

    def __exit__(self, *_exc):
        return False

    def _render_once(self):
        get_renderable = self.kwargs.get("get_renderable")
        if get_renderable is not None:
            self.renderable = get_renderable()
        elif self.args:
            self.renderable = self.args[0]
        else:
            self.renderable = None

    def refresh(self):
        self.refresh_count += 1
        self._render_once()


def _install_display_harness(monkeypatch, *, width=80, height=24, key="ENTER", force_terminal=True):
    FakeLive.instances = []
    output = io.StringIO()
    test_console = Console(file=output, force_terminal=force_terminal, width=width, color_system=None)
    monkeypatch.setattr(read_only_display, "console", test_console)
    monkeypatch.setattr(read_only_display.sys, "stdin", TtyStdin())
    monkeypatch.setattr(read_only_display, "terminal_size", lambda *, console: (width, height))
    monkeypatch.setattr(read_only_display, "Live", FakeLive)
    monkeypatch.setattr(read_only_display, "refresh_on_resize", noop_context)
    monkeypatch.setattr(read_only_display, "mouse_capture", noop_context)

    def read_key_with_repeats():
        if key == "KeyboardInterrupt":
            raise KeyboardInterrupt
        return key, 1

    monkeypatch.setattr(read_only_display, "_read_key_with_repeats", read_key_with_repeats)
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

    panel = FakeLive.instances[-1].renderable
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
