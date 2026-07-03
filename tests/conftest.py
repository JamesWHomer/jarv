"""Shared test helpers for the TUI suites.

Plain module (importable via ``from conftest import ...`` — pytest puts this
directory on ``sys.path``, and ``scripts/headsup_tool_server.py`` adds it
explicitly), so the helpers work both under pytest and in the live harness.
"""

from __future__ import annotations

import io
import time
from contextlib import contextmanager
from unittest import mock

try:
    import pytest
except ImportError:  # headsup_tool_server imports the harness without pytest
    pytest = None

from rich.console import Console


def wait_for(predicate, timeout=1.0, interval=0.01):
    """Poll ``predicate`` until truthy or ``timeout``; return its final value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def make_console(width=100, height=None, *, force_terminal=False):
    """Return ``(console, output)`` — a plain-text Console over a StringIO."""
    output = io.StringIO()
    console = Console(
        file=output,
        force_terminal=force_terminal,
        color_system=None,
        width=width,
        height=height,
    )
    return console, output


class FakeLive:
    """Fake ``Live`` usable as a ``live_factory`` or a patched ``Live`` class.

    Records every painted frame in ``frames`` (latest also in ``renderable``)
    and counts ``refresh`` calls; the class-level ``instances`` registry lets
    tests that patch a module's ``Live`` symbol reach the created instance.
    Nothing is rendered on ``__enter__`` — the loop's initial paint drives the
    first frame, keeping ``refresh_count``/``frames`` counts exact.
    """

    instances: list["FakeLive"] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        get_renderable = kwargs.get("get_renderable")
        if get_renderable is None and args and callable(args[0]):
            get_renderable = args[0]
        self._get_renderable = get_renderable
        self.console = kwargs.get("console") or (args[1] if len(args) > 1 else None)
        self.frames: list = []
        self.renderable = None
        self.refresh_count = 0
        self.entered = False
        self.exited = False
        type(self).instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def refresh(self):
        self.refresh_count += 1
        if self._get_renderable is not None:
            self.renderable = self._get_renderable()
            self.frames.append(self.renderable)

    def start(self, refresh=False):
        pass

    def stop(self):
        pass


class SnapshotLive(FakeLive):
    """``FakeLive`` that also renders each frame to plain text ``snapshots``.

    Paints once on ``__enter__`` (the session-browser tests read the first
    frame before any key arrives).
    """

    def __init__(self, *args, snapshot_width=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_width = snapshot_width
        self.snapshots: list[str] = []

    def __enter__(self):
        super().__enter__()
        self.refresh()
        return self

    def refresh(self):
        super().refresh()
        if self.renderable is not None:
            console, output = make_console(width=self.snapshot_width)
            console.print(self.renderable)
            self.snapshots.append(output.getvalue())


@contextmanager
def _null_context(*_args, **_kwargs):
    yield


_NEUTRAL_TUI_ATTRS = {
    "raw_input_mode": _null_context,
    "mouse_capture": _null_context,
    "bracketed_paste": _null_context,
    "windows_vt_input": _null_context,
    "disable_mouse_capture": lambda *a, **k: None,
}


@contextmanager
def neutral_terminal_modes():
    """Neutralise ``tui_app``'s terminal-mode managers (no real terminal)."""
    import jarv.tui_app as tui_app

    with mock.patch.multiple(tui_app, **_NEUTRAL_TUI_ATTRS):
        yield


def neutralize_tui_modes(monkeypatch):
    """Monkeypatch flavour of :func:`neutral_terminal_modes`."""
    import jarv.tui_app as tui_app

    for name, replacement in _NEUTRAL_TUI_ATTRS.items():
        monkeypatch.setattr(tui_app, name, replacement)


if pytest is not None:

    @pytest.fixture
    def neutral_tui_terminal():
        with neutral_terminal_modes():
            yield
