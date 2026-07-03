"""Headless driver for the alternate-screen heads-up UI.

Drives a real :class:`jarv.headsup.HeadsupApp` event loop with scripted input
and captures every rendered frame, so rendering/handling changes can be asserted
end-to-end without a real terminal.

It is built on the seams ``HeadsupApp`` already resolves at call time -- the
module symbols ``jarv.headsup.Live`` / ``_read_key_with_repeats`` /
``_key_available`` / ``terminal_size`` (see the "resolve patchable module
symbols at call time" note in ``HeadsupApp``) plus the per-instance
``_terminal_size_fn`` injection point on ``AltScreenApp``. No stale patch list:
the terminal-mode context managers in ``jarv.tui_app`` are neutralised so the
loop never touches the real terminal, whether run under pytest or standalone.

Public API:
    with HeadsupHarness(width=80, height=24, run_agent=fake) as h:
        h.feed_text("hello")
        h.feed_key("enter")
        h.wait_idle()
        assert "hello" in h.plain_frame
"""

from __future__ import annotations

import collections
import contextlib
import io
import re
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

from conftest import neutral_terminal_modes
from rich.console import Console

from jarv.command_input import TextInput
from jarv.headsup import HeadsupApp

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

#: Sentinel fed into the scripted reader to simulate a Ctrl-C interrupt: when the
#: loop pops it, the patched reader raises KeyboardInterrupt (just like a real
#: read), so on_interrupt fires on the loop thread.
_INTERRUPT = object()


def strip_ansi(text: str) -> str:
    """Return ``text`` with CSI escape sequences removed."""
    return _ANSI_RE.sub("", text)


# Friendly step key names -> the canonical key tokens the loop dispatches.
_KEY_ALIASES = {
    "enter": "ENTER",
    "return": "ENTER",
    "esc": "ESC",
    "escape": "ESC",
    "tab": "TAB",
    "backspace": "BACKSPACE",
    "del": "DELETE",
    "delete": "DELETE",
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    "pageup": "PAGEUP",
    "pagedown": "PAGEDOWN",
    "home": "HOME",
    "end": "END",
}


def _normalize_key(value: str) -> str:
    return _KEY_ALIASES.get(value.lower(), value.upper())


class _SizeHolder:
    """Mutable terminal size shared by the loop, render(), and frame capture."""

    def __init__(self, width: int, height: int):
        self.size = (max(1, width), max(1, height))

    def __call__(self, *, console: Any | None = None) -> tuple[int, int]:
        return self.size


class ScriptedInput:
    """Thread-safe queue of ``(key, repeat)`` pairs fed to the polled reader.

    The loop always calls :meth:`available` before :meth:`read`, and only the
    loop thread reads while only producers append, so no key is ever popped from
    an empty deque.
    """

    def __init__(self) -> None:
        self._items: collections.deque[tuple[Any, int]] = collections.deque()
        self._lock = threading.Lock()

    def feed(self, key: Any, repeat: int = 1) -> None:
        with self._lock:
            self._items.append((key, repeat))

    def available(self) -> bool:
        with self._lock:
            return bool(self._items)

    def read(self) -> tuple[Any, int]:
        with self._lock:
            return self._items.popleft()


class CapturingLive:
    """Stand-in for ``rich.live.Live`` that records each painted frame.

    Mimics just enough of the Live surface the loop touches: context-manager
    entry/exit, ``start``/``stop``/``refresh``. On every refresh it renders the
    current frame to a throwaway terminal console at the live terminal size,
    preserving control codes (e.g. the ``\\x1b[0K`` stale-edge erase) for
    fidelity.
    """

    def __init__(self, holder: _SizeHolder, *, get_renderable=None, console=None, **_ignored):
        self._holder = holder
        self._get_renderable = get_renderable
        self._lock = threading.Lock()
        self.frames: list[str] = []
        self.latest: str = ""
        self.refresh_count = 0

    def __enter__(self) -> "CapturingLive":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self, refresh: bool = False) -> None:
        if refresh:
            self.refresh()

    def stop(self) -> None:
        pass

    def refresh(self) -> None:
        if self._get_renderable is None:
            return
        renderable = self._get_renderable()
        width, height = self._holder.size
        buffer = io.StringIO()
        recorder = Console(
            file=buffer,
            force_terminal=True,
            color_system="truecolor",
            width=width,
            height=height,
        )
        recorder.print(renderable)
        frame = buffer.getvalue()
        with self._lock:
            self.latest = frame
            self.frames.append(frame)
            self.refresh_count += 1


def _noop(*_args, **_kwargs) -> None:
    return None


class HeadsupHarness:
    """Context manager that runs a real HeadsupApp loop headlessly."""

    def __init__(
        self,
        *,
        width: int = 80,
        height: int = 24,
        config: dict | None = None,
        args: Any | None = None,
        run_agent: Callable[..., Any] | None = None,
        sync_history: bool = False,
    ):
        self._holder = _SizeHolder(width, height)
        self._scripted = ScriptedInput()
        self._config = config or {"provider": "openai", "model": "test-model"}
        self._args = args
        self._run_agent = run_agent
        self._sync_history = sync_history
        self._stack: contextlib.ExitStack | None = None
        self._thread: threading.Thread | None = None
        self._created: list[CapturingLive] = []
        self.app: HeadsupApp | None = None
        self.live: CapturingLive | None = None

    # -- lifecycle ----------------------------------------------------- #
    def __enter__(self) -> "HeadsupHarness":
        ready = threading.Event()
        ready.set()
        module = SimpleNamespace()
        if self._run_agent is not None:
            module.run_agent = self._run_agent
        render_console = Console(
            file=io.StringIO(), force_terminal=False, color_system=None, width=self._holder.size[0]
        )
        app = HeadsupApp(
            self._config,
            client=object(),
            args=self._args,
            agent_loader=({"module": module}, ready),
            handle_slash=lambda command, rest, config, client, args, hint: (config, client),
            maybe_command=lambda _first, _rest: None,
            render_console=render_console,
        )
        # Drive resize detection and first-size off the shared holder. This is
        # bound per-instance because the base captured the real terminal_size at
        # construction time, so patching the module symbol alone would not steer
        # _check_resize.
        app._terminal_size_fn = self._holder
        if not self._sync_history:
            app._initial_history_synced = True
        self.app = app

        def _live_factory(**kwargs):
            live = CapturingLive(self._holder, **kwargs)
            self._created.append(live)
            return live

        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch("jarv.headsup.Live", _live_factory))
        stack.enter_context(
            mock.patch("jarv.headsup._read_key_with_repeats", self._read_scripted)
        )
        stack.enter_context(mock.patch("jarv.headsup._key_available", self._scripted.available))
        stack.enter_context(mock.patch("jarv.headsup.terminal_size", self._holder))
        stack.enter_context(mock.patch("jarv.headsup.disable_mouse_capture", _noop))
        stack.enter_context(neutral_terminal_modes())
        self._stack = stack

        self._thread = threading.Thread(target=app.run, name="headsup-harness", daemon=True)
        self._thread.start()
        # Wait for the first paint so callers see a frame immediately.
        self._wait(lambda: bool(self._created) and self._created[0].refresh_count >= 1, timeout=2.0)
        if self._created:
            self.live = self._created[0]
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def stop(self) -> None:
        app = self.app
        if app is not None:
            app.stop()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=6.0)
            self._thread = None
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    # -- input --------------------------------------------------------- #
    def feed_text(self, text: str) -> None:
        self._scripted.feed(TextInput(text), 1)
        self._wake()

    def feed_key(self, name: str, repeat: int = 1) -> None:
        self._scripted.feed(_normalize_key(name), repeat)
        self._wake()

    def feed_interrupt(self) -> None:
        self._scripted.feed(_INTERRUPT, 1)
        self._wake()

    def feed(self, steps: list[dict]) -> None:
        """Feed ``{"type": "text"|"key"|"interrupt", "value": ...}`` steps."""
        for step in steps:
            kind = step.get("type")
            value = step.get("value", "")
            if kind == "text":
                self._scripted.feed(TextInput(str(value)), 1)
            elif kind == "key":
                self._scripted.feed(_normalize_key(str(value)), int(step.get("repeat", 1)))
            elif kind == "interrupt":
                self._scripted.feed(_INTERRUPT, 1)
            else:
                raise ValueError(f"unknown step type: {kind!r}")
        self._wake()

    def resize(self, width: int, height: int) -> None:
        self._holder.size = (max(1, width), max(1, height))
        self._wake()

    # -- observation --------------------------------------------------- #
    @property
    def frame(self) -> str:
        return self.live.latest if self.live is not None else ""

    @property
    def plain_frame(self) -> str:
        return strip_ansi(self.frame)

    @property
    def transcript(self) -> str:
        if self.app is None:
            return ""
        width = self._holder.size[0]
        return "\n".join(line.plain for line in self.app._transcript_lines(width))

    @property
    def prompt_buffer(self) -> str:
        if self.app is None:
            return ""
        return str(getattr(self.app, "editor", {}).get("buffer", ""))

    @property
    def answer_prompt(self) -> str | None:
        request = getattr(self.app, "_answer_request", None)
        return request.get("label") if request else None

    def state(self) -> dict:
        app = self.app
        return {
            "frame": self.frame,
            "plain": self.plain_frame,
            "transcript": self.transcript,
            "prompt_buffer": self.prompt_buffer,
            "answer_prompt": self.answer_prompt,
            "size": list(self._holder.size),
            "agent_busy": bool(getattr(app, "_agent_busy", False)),
            "answer_pending": self.answer_prompt is not None,
            "running": bool(getattr(app, "_running", False)),
            "refreshes": self.live.refresh_count if self.live is not None else 0,
        }

    def wait_idle(self, timeout: float = 3.0) -> bool:
        """Block until scripted input is drained and no agent turn is running."""
        return self._wait(self._is_idle, timeout=timeout)

    # -- internals ----------------------------------------------------- #
    def _read_scripted(self, **_kwargs) -> tuple[Any, int]:
        key, repeat = self._scripted.read()
        if key is _INTERRUPT:
            raise KeyboardInterrupt
        return key, repeat

    def _is_idle(self) -> bool:
        app = self.app
        if app is None:
            return True
        if self._scripted.available():
            return False
        if not app._queue.empty():
            return False
        return not bool(getattr(app, "_agent_busy", False))

    def _wake(self) -> None:
        app = self.app
        if app is not None:
            # Invalidate from a non-loop thread enqueues a wake sentinel so the
            # idle loop services the new input without waiting out poll_interval.
            app.invalidate()

    def _wait(self, predicate: Callable[[], bool], *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.005)
        return predicate()
