"""Cooperative cancellation primitives for one agent turn."""

from __future__ import annotations

import threading
from collections.abc import Callable


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
