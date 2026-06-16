"""Cooperative cancellation primitives for one agent turn."""

from __future__ import annotations

import ctypes
import os
import signal
import threading
from collections.abc import Callable
from contextlib import contextmanager


class TurnCancelled(Exception):
    """Raised when the current agent turn has been cancelled."""


class CancellationToken:
    """Thread-safe cancellation state with resource cleanup callbacks."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._next_callback_id = 0

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        callbacks: list[Callable[[], None]]
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = list(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

    def throw_if_cancelled(self) -> None:
        if self.cancelled:
            raise TurnCancelled

    def register(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register cleanup and return a function that unregisters it."""
        with self._lock:
            if self._event.is_set():
                run_now = True
                callback_id = -1
            else:
                run_now = False
                callback_id = self._next_callback_id
                self._next_callback_id += 1
                self._callbacks[callback_id] = callback

        if run_now:
            try:
                callback()
            except Exception:
                pass

        def unregister() -> None:
            if callback_id < 0:
                return
            with self._lock:
                self._callbacks.pop(callback_id, None)

        return unregister


@contextmanager
def cancel_token_on_sigint(token: CancellationToken):
    """Cancel a token immediately when Ctrl+C is received.

    Python raises KeyboardInterrupt on the main thread, which can be delayed on
    Windows while subprocess pipe I/O is blocked. A Windows console control
    handler runs promptly on Ctrl+C and lets registered token callbacks kill
    active subprocesses before KeyboardInterrupt is delivered.
    """

    previous_sigint = None
    signal_installed = False
    windows_handler = None

    if threading.current_thread() is threading.main_thread():
        previous_sigint = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum, frame):
            token.cancel()
            if callable(previous_sigint):
                return previous_sigint(signum, frame)
            if previous_sigint == signal.SIG_DFL:
                raise KeyboardInterrupt
            return None

        signal.signal(signal.SIGINT, _handle_sigint)
        signal_installed = True

    if os.name == "nt":
        kernel32 = getattr(ctypes, "windll", None)
        kernel32 = getattr(kernel32, "kernel32", None)
        if kernel32 is not None:
            handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

            @handler_type
            def _handle_console_event(ctrl_type):
                if ctrl_type in (0, 1):
                    token.cancel()
                return False

            try:
                if kernel32.SetConsoleCtrlHandler(_handle_console_event, True):
                    windows_handler = _handle_console_event
            except Exception:
                windows_handler = None

    try:
        yield
    finally:
        if windows_handler is not None:
            try:
                ctypes.windll.kernel32.SetConsoleCtrlHandler(windows_handler, False)
            except Exception:
                pass
        if signal_installed:
            signal.signal(signal.SIGINT, previous_sigint)
