"""Shared, cancellable request limits for keyless web search."""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from time import monotonic

from .cancellation import CancellationToken


class SearchDeadlineExceeded(Exception):
    """The total search time, including queued requests and retries, elapsed."""


class SearchCooldownError(Exception):
    """Search is paused after an upstream challenge or rate limit."""

    def __init__(self, reason: str, remaining: float) -> None:
        self.reason = reason
        self.remaining = max(0.0, remaining)
        super().__init__(
            f"DuckDuckGo search is paused for {self.remaining:.1f} more seconds: {reason}"
        )


class SearchBudget:
    """Bound an entire search and cancel its active HTTP resources at the deadline.

    Pass ``token`` to HTTP calls so their registered cleanup runs at the deadline.
    Call ``check`` after a request, including when it raises, to distinguish a
    deadline from user cancellation. Parent cancellation always takes precedence.
    """

    def __init__(
        self, timeout: float, cancellation_token: CancellationToken | None = None
    ) -> None:
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("search timeout must be finite and nonnegative")
        self.timeout = timeout
        self.token = CancellationToken()
        self._parent = cancellation_token
        self._deadline: float | None = None
        self._expired = threading.Event()
        self._wake = threading.Event()
        self.token.register(self._wake.set)
        self._timer: threading.Timer | None = None
        self._unregister_parent = lambda: None

    def __enter__(self) -> SearchBudget:
        if self._deadline is not None:
            raise RuntimeError("a search budget cannot be reused")
        self._deadline = monotonic() + self.timeout
        if self._parent is not None:
            self._unregister_parent = self._parent.register(self.token.cancel)
        try:
            self.check()
            self._timer = threading.Timer(self.remaining(), self._expire)
            self._timer.daemon = True
            self._timer.start()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._unregister_parent()

    def _expire(self) -> None:
        self._expired.set()
        self.token.cancel()

    def check(self) -> None:
        if self._parent is not None:
            self._parent.throw_if_cancelled()
        if self._deadline is None:
            raise RuntimeError("search budget must be used in a context manager")
        if self._expired.is_set() or monotonic() >= self._deadline:
            self._expire()
            if self._parent is not None:
                self._parent.throw_if_cancelled()
            raise SearchDeadlineExceeded(
                f"search exceeded its {self.timeout:g} second total time limit"
            )
        self.token.throw_if_cancelled()

    def remaining(self) -> float:
        self.check()
        # check() establishes that a deadline exists; clamp a subsequent tick.
        return max(0.0, self._deadline - monotonic())

    def wait(self, seconds: float) -> None:
        """Wait for retry backoff, interrupted by cancellation or the deadline."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("search wait must be finite and nonnegative")
        until = monotonic() + seconds
        while True:
            self.check()
            remaining = until - monotonic()
            if remaining <= 0:
                return
            # Timed waits can return slightly early on Windows. Recheck the
            # monotonic target before re-entering the shared cooldown gate.
            self._wake.wait(min(remaining, self.remaining()))


class SearchCoordinator:
    """Serialize search HTTP requests and share start spacing and cooldowns."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._in_flight = False
        self._last_started: float | None = None
        self._cooldown_until = 0.0
        self._cooldown_reason = ""

    def _wake_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def cooldown(self, seconds: float, reason: str) -> None:
        """Extend the shared cooldown and reject queued requests immediately."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("search cooldown must be finite and nonnegative")
        with self._condition:
            until = monotonic() + seconds
            if until > self._cooldown_until:
                self._cooldown_until = until
                self._cooldown_reason = reason
            self._condition.notify_all()

    @contextmanager
    def request(self, budget: SearchBudget, min_interval: float):
        """Hold the single request slot, with interruptible bounded queueing."""
        if not math.isfinite(min_interval) or min_interval < 0:
            raise ValueError("search spacing must be finite and nonnegative")
        acquired = False
        unregister = budget.token.register(self._wake_waiters)
        try:
            with self._condition:
                while True:
                    budget.check()
                    now = monotonic()
                    if self._cooldown_until > now:
                        raise SearchCooldownError(
                            self._cooldown_reason, self._cooldown_until - now
                        )
                    spacing = (
                        0.0
                        if self._last_started is None
                        else max(0.0, self._last_started + min_interval - now)
                    )
                    if not self._in_flight and spacing <= 0:
                        self._in_flight = True
                        self._last_started = now
                        acquired = True
                        break
                    wait_time = budget.remaining()
                    if not self._in_flight:
                        wait_time = min(wait_time, spacing)
                    self._condition.wait(wait_time)
            budget.check()
            yield
        finally:
            if acquired:
                with self._condition:
                    self._in_flight = False
                    self._condition.notify_all()
            unregister()
