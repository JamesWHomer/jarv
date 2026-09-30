"""Limits and unattended-input policy shared by a run and its children."""

from __future__ import annotations

import threading
import time

from .cancellation import CancellationToken, TurnCancelled


class RunStopped(TurnCancelled):
    def __init__(self, message: str, status: str = "input_required"):
        super().__init__(message)
        self.status = status


class RunControl:
    def __init__(self, token: CancellationToken, *, max_turns=None, timeout=None):
        self.token = token
        self.max_turns = max_turns
        self.deadline = time.monotonic() + timeout if timeout else None
        self.turns = 0
        self.error: str | None = None
        self.status: str | None = None
        self._lock = threading.Lock()
        self._timer = threading.Timer(timeout, self.stop, args=(
            "Run timeout exceeded.", "limit",
        )) if timeout else None
        if self._timer:
            self._timer.daemon = True
            self._timer.start()

    def stop(self, message: str, status: str = "input_required") -> None:
        with self._lock:
            if self.error is None:
                self.error, self.status = message, status
        self.token.cancel()

    def check(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.stop("Run timeout exceeded.", "limit")
        if self.error:
            raise RunStopped(self.error, self.status or "error")
        self.token.throw_if_cancelled()

    def begin_turn(self) -> None:
        self.check()
        with self._lock:
            exceeded = self.max_turns is not None and self.turns >= self.max_turns
            if not exceeded:
                self.turns += 1
        if exceeded:
            self.stop("Maximum agent turns reached.", "limit")
            self.check()

    def close(self) -> None:
        if self._timer:
            self._timer.cancel()


def require_user_input(config: dict | None, message: str) -> None:
    """Abort explicitly unattended runs instead of guessing an approval/answer."""
    if not (config or {}).get("_non_interactive"):
        return
    control = config.get("_run_control")
    if control is not None:
        control.stop(message)
    raise RunStopped(message)


def emit_event(config: dict, event_type: str, **data) -> None:
    sink = config.get("_event_sink")
    if sink is not None:
        sink(event_type, **data)
