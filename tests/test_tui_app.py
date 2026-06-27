"""Tests for the single-threaded alternate-screen application loop.

These exercise the loop without a real terminal by injecting a fake ``Live``, a
scripted keyboard source, and a controllable terminal-size function. The goal is
behaviour coverage (keys dispatched, resize handled, worker events painted, clean
shutdown) so the loop can be refactored without these tests pinning internals.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from rich.text import Text

from jarv import tui_app
from jarv.tui_app import AltScreenApp, AppEvent


class FakeConsole:
    """Stand-in console that records ``clear()`` calls."""

    def __init__(self):
        self.clears = 0

    def clear(self):
        self.clears += 1

    def set_alt_screen(self, enable=True):
        return True


class FakeLive:
    """Records each painted frame by invoking the app's ``get_renderable``."""

    def __init__(self, get_renderable, console):
        self._get_renderable = get_renderable
        self.console = console
        self.frames: list[Text] = []
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def refresh(self):
        self.frames.append(self._get_renderable())


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


class ScriptedApp(AltScreenApp):
    """Test app driven by a fixed list of keys, then idle until stopped."""

    def __init__(self, keys=None, **kwargs):
        self._keys = list(keys or [])
        self.seen_keys: list[tuple[str, int]] = []
        self.resizes: list[tuple[int, int]] = []
        self.app_events: list[AppEvent] = []
        self.ticks = 0
        self.started = False
        self.stopped = False
        super().__init__(**kwargs)

    def render(self):
        return Text(f"keys={len(self.seen_keys)} events={len(self.app_events)}")

    # Scripted input source -----------------------------------------------
    def _key_source_available(self) -> bool:
        return bool(self._keys)

    def _key_source_read(self) -> tuple[str, int]:
        return self._keys.pop(0), 1

    # Hooks ----------------------------------------------------------------
    def on_start(self):
        self.started = True

    def on_stop(self):
        self.stopped = True

    def on_key(self, key, repeat):
        self.seen_keys.append((key, repeat))
        if key == "STOP":
            self.stop("finished")

    def on_resize(self, size):
        self.resizes.append(size)

    def on_app_event(self, event):
        self.app_events.append(event)

    def on_tick(self):
        self.ticks += 1


def _make_app(keys=None, *, size=(80, 24), console=None, **kwargs):
    console = console or FakeConsole()
    holder: dict[str, FakeLive] = {}

    def live_factory(get_renderable, _console):
        live = FakeLive(get_renderable, console)
        holder["live"] = live
        return live

    app = ScriptedApp(
        keys=keys,
        console=console,
        poll_interval=0.005,
        live_factory=live_factory,
        terminal_size_fn=lambda *, console=None: size,
        **kwargs,
    )
    app._key_available_fn = app._key_source_available
    app._read_key_fn = app._key_source_read
    return app, holder


def test_run_enters_screen_and_dispatches_keys():
    app, holder = _make_app(keys=["a", "b", "STOP"])
    result = app.run()

    assert result == "finished"
    assert app.started and app.stopped
    assert app.seen_keys == [("a", 1), ("b", 1), ("STOP", 1)]
    live = holder["live"]
    assert live.entered and live.exited
    assert live.frames, "expected at least one painted frame"


def test_first_paint_happens_before_idle():
    # No keys: the loop should still paint once, then idle until stopped.
    app, holder = _make_app(keys=[])

    def stop_soon():
        time.sleep(0.02)
        app.stop("done")

    t = threading.Thread(target=stop_soon)
    t.start()
    result = app.run()
    t.join(timeout=1.0)

    assert result == "done"
    assert holder["live"].frames, "loop must paint the initial frame"


def test_worker_thread_event_is_painted():
    # Run the loop in a thread; the test thread plays the background producer
    # and observes painted frames, so the assertion never races shutdown.
    app, holder = _make_app(keys=[])
    loop_thread = threading.Thread(target=app.run)
    loop_thread.start()
    try:
        assert _wait_until(lambda: "live" in holder and holder["live"].frames)
        app.post_app_event("stream_delta", "hello")
        app.invalidate()
        assert _wait_until(
            lambda: any("events=1" in f.plain for f in holder["live"].frames)
        )
    finally:
        app.stop("ok")
        loop_thread.join(timeout=1.0)

    assert not loop_thread.is_alive()
    assert app.result == "ok"
    assert [e.kind for e in app.app_events] == ["stream_delta"]
    assert app.app_events[0].payload == "hello"


def test_resize_detected_and_clears_console():
    console = FakeConsole()
    # Size is stable, then grows; clamps so later polls keep the new size.
    sizes = [(80, 24), (80, 24), (100, 30), (100, 30)]
    calls = {"n": 0}

    def sizer(*, console=None):
        idx = min(calls["n"], len(sizes) - 1)
        calls["n"] += 1
        return sizes[idx]

    app, _ = _make_app(keys=[], console=console)
    app._terminal_size_fn = sizer

    base_on_resize = app.on_resize

    def on_resize(size):
        base_on_resize(size)
        if size == (100, 30):
            app.stop("resized")

    app.on_resize = on_resize
    assert app.run() == "resized"
    assert (100, 30) in app.resizes
    assert console.clears >= 1


def test_resize_without_change_is_ignored():
    app, _ = _make_app(keys=["STOP"], size=(80, 24))
    app.run()
    assert app.resizes == []


def test_no_key_sentinel_does_not_block_resize_detection():
    # The reader can hand back the resize sentinel (no actionable key) while the
    # terminal size changes; the loop must still detect the resize and repaint
    # rather than treating the sentinel as a blocking read.
    console = FakeConsole()
    sizes = [(80, 24), (100, 30), (100, 30)]
    calls = {"n": 0}

    def sizer(*, console=None):
        idx = min(calls["n"], len(sizes) - 1)
        calls["n"] += 1
        return sizes[idx]

    app, holder = _make_app(keys=[], console=console)
    app._terminal_size_fn = sizer
    # Input always reports "available" but only ever yields the no-key sentinel.
    app._key_available_fn = lambda: True
    app._read_key_fn = lambda: ("RESIZE", 1)

    base_on_resize = app.on_resize

    def on_resize(size):
        base_on_resize(size)
        if size == (100, 30):
            app.stop("resized")

    app.on_resize = on_resize
    assert app.run() == "resized"
    assert (100, 30) in app.resizes
    assert holder["live"].frames


class _StubLive:
    def __init__(self):
        self.stopped = False
        self.started = False

    def stop(self):
        self.stopped = True

    def start(self, refresh=False):
        self.started = True


def test_suspended_forces_repaint_without_phantom_resize_clear():
    console = FakeConsole()
    app, _ = _make_app(keys=[], console=console)
    app._last_size = (80, 24)
    live = _StubLive()
    app.live = live

    with app.suspended():
        pass

    # The nested view owned the terminal: our Live was stopped and restarted.
    assert live.stopped and live.started
    # A repaint is forced on resume (so the nested view is covered)...
    assert app._dirty is True
    # ...but the baseline is re-synced to the real current size rather than None,
    # so the next poll does NOT see a phantom resize and wipe the screen after we
    # have already repainted -- that bogus clear was a one-frame flicker on exit.
    assert app._last_size == (80, 24)
    assert app._check_resize() is False
    # Only the single intentional clear on the way *in* happened; resume adds none.
    assert console.clears == 1


def test_suspended_drops_vt_input_for_nested_view(monkeypatch):
    # A VT-input app (heads-up) must hand the console to the nested view as plain
    # key records: otherwise arrows/wheel arrive as ``ESC [ A/B`` and the leading
    # ESC closes the nested view while ``[A``/``[B`` leaks into our input box.
    events: list[str] = []

    @contextmanager
    def fake_vt_suspend():
        events.append("vt_off")
        try:
            yield
        finally:
            events.append("vt_on")

    monkeypatch.setattr(tui_app, "windows_vt_input_suspended", fake_vt_suspend)

    app, _ = _make_app(keys=[])
    app.use_vt_input = True
    app.live = _StubLive()

    with app.suspended():
        events.append("nested")

    assert events == ["vt_off", "nested", "vt_on"]


def test_suspended_leaves_vt_input_untouched_without_vt_input(monkeypatch):
    # Views that never enable VT input must not poke the console mode at all.
    events: list[str] = []

    @contextmanager
    def fake_vt_suspend():
        events.append("vt_off")
        try:
            yield
        finally:
            events.append("vt_on")

    monkeypatch.setattr(tui_app, "windows_vt_input_suspended", fake_vt_suspend)

    app, _ = _make_app(keys=[])
    app.use_vt_input = False
    app.live = _StubLive()

    with app.suspended():
        events.append("nested")

    assert events == ["nested"]


def test_preserve_alt_screen_holds_both_directions():
    calls = []

    class Console(FakeConsole):
        def set_alt_screen(self, enable=True):
            calls.append(enable)
            return True

    app, _ = _make_app(keys=[], console=Console())

    with app._preserve_alt_screen():
        # Neither toggle reaches the console while the alt screen is held steady,
        # so no redundant \x1b[?1049h/l control codes are written. Both still
        # report the screen as enabled so a nested Live homes the cursor.
        assert app.console.set_alt_screen(False) is True
        assert app.console.set_alt_screen(True) is True
    assert calls == []

    # After the block, the real method handles both directions again.
    app.console.set_alt_screen(False)
    app.console.set_alt_screen(True)
    assert calls == [False, True]


def test_app_event_marks_dirty():
    app, _ = _make_app(keys=[])
    app._dirty = False
    app._dispatch(AppEvent("status", 1))
    assert app._dirty is True
    assert app.app_events[0].kind == "status"


def test_invalidate_sets_dirty_and_wakes():
    app, _ = _make_app(keys=[])
    app._dirty = False
    app.invalidate()
    assert app._dirty is True
    # The wake sentinel is enqueued so an idle wait returns promptly.
    assert not app._queue.empty()


def test_on_start_can_stop_immediately():
    class ImmediateStop(ScriptedApp):
        def on_start(self):
            super().on_start()
            self.stop("early")

    console = FakeConsole()

    def live_factory(get_renderable, _c):
        return FakeLive(get_renderable, console)

    app = ImmediateStop(
        console=console,
        poll_interval=0.005,
        live_factory=live_factory,
        terminal_size_fn=lambda *, console=None: (80, 24),
    )
    app._key_available_fn = app._key_source_available
    app._read_key_fn = app._key_source_read

    assert app.run() == "early"
    assert app.stopped


def test_keyboard_interrupt_routes_to_on_interrupt():
    app, _ = _make_app(keys=[])
    interrupted = {"hit": False}

    def raising_read():
        raise KeyboardInterrupt

    app._key_available_fn = lambda: True
    app._read_key_fn = raising_read

    def on_interrupt():
        interrupted["hit"] = True
        app.stop("interrupted")

    app.on_interrupt = on_interrupt
    assert app.run() == "interrupted"
    assert interrupted["hit"]


def test_tick_invalidation_repaints_until_stopped():
    app, holder = _make_app(keys=[])
    target_ticks = 3

    def tick():
        app.ticks += 1
        if app.ticks <= target_ticks:
            app.invalidate()
        else:
            app.stop("ticked")

    app.on_tick = tick
    result = app.run()

    assert result == "ticked"
    assert app.ticks >= target_ticks
    # Each invalidating tick produced a fresh paint beyond the initial one.
    assert len(holder["live"].frames) >= target_ticks
