"""Single-threaded alternate-screen application loop.

This is the shared foundation for jarv's interactive full-screen views (heads-up,
the session browser, the settings menu, ...). It replaces the historical model in
which several daemon threads -- a ticker, an idle-animation timer, an esc
listener, the agent worker -- all mutated shared state and raced to repaint a
single Rich ``Live`` behind a tangle of locks. That model is the source of the
recurring stale-right-edge / dropped-refresh / flicker bugs: ``refresh()`` even
dropped repaints silently when its lock was contended.

The model here is a classic event loop:

* **One thread renders.** The thread that calls :meth:`AltScreenApp.run` owns the
  ``Live`` display and is the *only* code that ever paints. There are no render
  locks because there is only one renderer.
* **Everything else is a producer.** Keyboard input, terminal resize, periodic
  animation ticks, and updates from worker threads (streaming deltas, tool cards,
  status changes) never touch the screen. They mutate app state and call
  :meth:`AltScreenApp.invalidate` (or :meth:`post`); the loop coalesces the work
  and paints exactly one frame per iteration when something changed.

Input stays on the loop thread and is *polled* (``_key_available`` then a
non-blocking read) rather than read on a background thread. This matters because
``command_input._read_key`` blocks indefinitely on every platform (Windows
``ReadConsoleInputW``/``getwch``; POSIX ``select`` until a key), so a background
reader could never be stopped cleanly and would fight the next view for stdin.
While idle the loop blocks on its event queue with a short timeout, so background
events wake it immediately and CPU stays near zero between keystrokes.

Subclasses implement :meth:`render` and override the ``on_*`` hooks they need.
The pure layout/scroll helpers in :mod:`jarv.tui_frame` and :mod:`jarv.tui_overlay`
compose with this loop; this module owns only the loop, input, and lifecycle.
"""

from __future__ import annotations

import queue
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Callable

from rich.console import RenderableType
from rich.live import Live

from .command_input import (
    _key_available,
    _read_key_with_repeats,
    bracketed_paste,
    disable_mouse_capture,
    mouse_capture,
)
from .display import console as default_console
from .display import mark_first_paint, refresh_on_resize, terminal_size


@dataclass(frozen=True)
class AppEvent:
    """An application-defined event posted from a worker thread.

    ``kind`` names the event (e.g. ``"stream_delta"``); ``payload`` is arbitrary.
    This is the only payload that travels through the loop's queue: keypresses,
    resizes, and animation ticks are detected on the loop thread itself, so they
    never need to be enqueued.
    """

    kind: str
    payload: Any = None


# Internal sentinel: wakes the idle wait so a pending repaint happens promptly
# without carrying any state of its own.
_WAKE = object()

_DEFAULT_POLL_INTERVAL = 0.04


class AltScreenApp:
    """Base class for a single-threaded, event-driven full-screen view.

    Override :meth:`render` (required) plus any of the ``on_*`` hooks. Drive the
    view with :meth:`run`, which enters the alternate screen, pumps events until
    :meth:`stop` is called, and returns :attr:`result`.

    Worker threads communicate only through the thread-safe producer API
    (:meth:`post`, :meth:`post_app_event`, :meth:`invalidate`, :meth:`stop`).
    """

    #: Override to enable/disable behaviours without touching ``__init__``.
    text_mode: bool = True
    batch_text: bool = True
    translate_mouse_wheel: bool = False
    use_mouse_capture: bool = False
    use_bracketed_paste: bool = True
    clear_on_resize: bool = True
    first_paint_label: str = "alt-screen"

    def __init__(
        self,
        *,
        console: Any | None = None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        repeatable_keys: frozenset[str] | None = None,
        # Injection points -- defaulted to the real implementations, replaced in tests.
        live_factory: Callable[..., Any] | None = None,
        read_key_fn: Callable[[], tuple[str, int]] | None = None,
        key_available_fn: Callable[[], bool] | None = None,
        terminal_size_fn: Callable[..., tuple[int, int]] | None = None,
        refresh_on_resize_fn: Callable[..., Any] | None = None,
    ):
        self.console = console or default_console
        self.poll_interval = poll_interval
        self._repeatable_keys = repeatable_keys
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._dirty = True
        self.live: Any | None = None
        self.result: Any = None
        self._last_size: tuple[int, int] | None = None
        self._loop_thread: threading.Thread | None = None

        self._live_factory = live_factory or self._default_live_factory
        self._read_key_fn = read_key_fn or self._default_read_key
        self._key_available_fn = key_available_fn or _key_available
        self._terminal_size_fn = terminal_size_fn or terminal_size
        self._refresh_on_resize_fn = refresh_on_resize_fn or refresh_on_resize

    # ------------------------------------------------------------------ #
    # Subclass API (called on the loop thread)
    # ------------------------------------------------------------------ #
    def render(self) -> RenderableType:
        """Return the renderable for the current frame. Required override."""
        raise NotImplementedError

    def on_start(self) -> None:
        """Called once after the screen is entered, before the first paint."""

    def on_stop(self) -> None:
        """Called once after the loop exits, before the screen is restored."""

    def on_key(self, key: str, repeat: int) -> None:
        """Handle a keypress. Default: no-op. Call :meth:`stop` to exit."""

    def on_resize(self, size: tuple[int, int]) -> None:
        """Handle a terminal resize. Default: no-op (a repaint is automatic)."""

    def on_tick(self) -> None:
        """Advance time-based state (spinners, animations). Default: no-op.

        Call :meth:`invalidate` from here when the visible state changed so the
        loop repaints; otherwise idle frames are skipped.
        """

    def on_app_event(self, event: Any) -> None:
        """Handle an event posted via :meth:`post`/:meth:`post_app_event`.

        Usually an :class:`AppEvent`, but any object passed to :meth:`post` is
        delivered here. Default: no-op.
        """

    def on_interrupt(self) -> None:
        """Handle Ctrl-C raised while reading input. Default: :meth:`stop`."""
        self.stop()

    # ------------------------------------------------------------------ #
    # Producer API (safe to call from any thread)
    # ------------------------------------------------------------------ #
    def invalidate(self) -> None:
        """Mark the view as needing a repaint and wake the loop promptly.

        Called from a worker thread, this enqueues a wake sentinel so the idle
        loop repaints immediately. Called from the loop thread itself (e.g. an
        ``on_tick`` animation frame), it only sets the dirty flag -- enqueuing a
        wake there would short-circuit the idle wait and busy-spin the loop, so
        the frame interval is left to pace the repaint.
        """
        self._dirty = True
        if threading.current_thread() is not self._loop_thread:
            try:
                self._queue.put_nowait(_WAKE)
            except queue.Full:  # pragma: no cover - unbounded queue
                pass

    def post(self, event: Any) -> None:
        """Enqueue an event object for the loop to dispatch."""
        self._queue.put(event)

    def post_app_event(self, kind: str, payload: Any = None) -> None:
        """Enqueue an :class:`AppEvent` for :meth:`on_app_event`."""
        self._queue.put(AppEvent(kind, payload))

    def stop(self, result: Any = None) -> None:
        """Ask the loop to exit. ``result`` (if given) becomes :attr:`result`."""
        if result is not None:
            self.result = result
        self._running = False
        try:
            self._queue.put_nowait(_WAKE)
        except queue.Full:  # pragma: no cover - unbounded queue
            pass

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #
    def run(self) -> Any:
        """Enter the alternate screen and pump events until stopped."""
        self._running = True
        self._dirty = True
        self._loop_thread = threading.current_thread()
        self._last_size = self._terminal_size_fn(console=self.console)

        disable_mouse_capture()
        with self._screen_context():
            self.on_start()
            if self._running:
                # Paint the initial frame before blocking on input, so the view
                # appears immediately (and first-paint latency is measured here).
                self._paint()
                self._dirty = False
            try:
                while self._running:
                    progressed = self._pump()
                    self.on_tick()
                    if self._dirty and self._running:
                        self._paint()
                        self._dirty = False
                    if self._running and not progressed:
                        self._idle_wait()
            finally:
                self._running = False
                self.on_stop()
                self.live = None
        return self.result

    def _pump(self) -> bool:
        """Process queued events, a resize, and any buffered input.

        Returns True if anything was handled, so the caller can skip the idle
        wait while the view is actively busy.
        """
        progressed = self._drain_queue()
        if self._check_resize():
            progressed = True
        if self._service_input():
            progressed = True
        return progressed

    def _drain_queue(self) -> bool:
        handled = False
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            handled = True
            self._dispatch(event)
        return handled

    def _idle_wait(self) -> None:
        """Block until a producer wakes us or the poll interval elapses."""
        try:
            event = self._queue.get(timeout=self.poll_interval)
        except queue.Empty:
            return
        self._dispatch(event)

    def _dispatch(self, event: Any) -> None:
        if event is _WAKE:
            return
        self.on_app_event(event)
        self._dirty = True

    def _check_resize(self) -> bool:
        size = self._terminal_size_fn(console=self.console)
        if size == self._last_size:
            return False
        self._last_size = size
        if self.clear_on_resize:
            try:
                self.console.clear()
            except Exception:
                pass
        self.on_resize(size)
        self._dirty = True
        return True

    def _service_input(self) -> bool:
        handled = False
        while self._running and self._key_available_fn():
            try:
                key, repeat = self._read_key_fn()
            except KeyboardInterrupt:
                self.on_interrupt()
                return True
            self.on_key(key, repeat)
            self._dirty = True
            handled = True
        return handled

    def _paint(self) -> None:
        live = self.live
        if live is None:
            return
        live.refresh()
        mark_first_paint(self.first_paint_label)

    def paint_now(self) -> None:
        """Force one synchronous repaint. Only call on the loop thread.

        Used by nested modal reads (a confirmation prompt, captured command
        output) that block the main loop and need the screen updated in place
        before they return.
        """
        self._paint()
        self._dirty = False

    # ------------------------------------------------------------------ #
    # Screen / input plumbing (overridable via injection)
    # ------------------------------------------------------------------ #
    def _screen_context(self):
        """Context manager that owns the Live display and terminal modes."""

        @contextmanager
        def _ctx():
            live = self._live_factory(self.render, self.console)
            with ExitStack() as stack:
                stack.enter_context(live)
                self.live = live
                if self.use_mouse_capture:
                    stack.enter_context(mouse_capture())
                if self.use_bracketed_paste:
                    stack.enter_context(bracketed_paste())
                stack.enter_context(
                    self._refresh_on_resize_fn(live, on_change=self._on_resize_signal)
                )
                yield live

        return _ctx()

    def _on_resize_signal(self) -> None:
        """Resize watcher callback: just nudge the loop; it polls the size."""
        self.invalidate()

    def _default_live_factory(self, get_renderable, console):
        return Live(
            get_renderable=get_renderable,
            console=console,
            screen=True,
            auto_refresh=False,
            transient=False,
            vertical_overflow="crop",
        )

    def _default_read_key(self) -> tuple[str, int]:
        kwargs: dict[str, Any] = {
            "text_mode": self.text_mode,
            "batch_text": self.batch_text,
            "translate_mouse_wheel": self.translate_mouse_wheel,
        }
        if self._repeatable_keys is not None:
            kwargs["repeatable"] = self._repeatable_keys
        return _read_key_with_repeats(**kwargs)
