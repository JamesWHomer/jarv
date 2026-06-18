"""Background model catalog refresh coordination for settings."""

import threading
from collections.abc import Callable

class _ModelCatalogRefresher:
    """Deduplicate delayed catalog refreshes and keep network work off the UI thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._timers: dict[str, threading.Timer] = {}
        self._inflight: set[str] = set()
        self._latest: dict[str, tuple[int, dict, Callable]] = {}
        self._closed = False

    def request(
        self,
        config: dict,
        callback: Callable[[str, list[tuple[str, str]], int], None],
        *,
        delay: float = 0,
    ) -> int:
        from .model_catalog import catalog_cache_key

        snapshot = dict(config)
        key = catalog_cache_key(snapshot)
        with self._lock:
            if self._closed:
                return self._generation
            self._generation += 1
            generation = self._generation
            self._latest[key] = (generation, snapshot, callback)
            timer = self._timers.pop(key, None)
            if timer is not None:
                timer.cancel()
            if key in self._inflight:
                return generation
            timer = threading.Timer(delay, self._launch, args=(key,))
            timer.daemon = True
            self._timers[key] = timer
            timer.start()
        return generation

    def _launch(self, key: str) -> None:
        with self._lock:
            self._timers.pop(key, None)
            pending = self._latest.get(key)
            if pending is None or key in self._inflight:
                return
            self._inflight.add(key)
            _generation, snapshot, _callback = pending
        threading.Thread(
            target=self._refresh,
            args=(key, snapshot),
            daemon=True,
            name="jarv-model-catalog",
        ).start()

    def _refresh(self, key: str, snapshot: dict) -> None:
        from .model_catalog import get_cached_model_choices, refresh_model_choices

        try:
            choices = refresh_model_choices(snapshot)
        except Exception:
            choices = get_cached_model_choices(snapshot)
        with self._lock:
            self._inflight.discard(key)
            pending = self._latest.pop(key, None)
            closed = self._closed
        if pending is None or closed:
            return
        generation, _latest_snapshot, callback = pending
        callback(str(snapshot.get("provider", "openai")), choices, generation)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            timers = list(self._timers.values())
            self._timers.clear()
            self._latest.clear()
        for timer in timers:
            timer.cancel()

    def cancel_pending(self) -> None:
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
            for key in list(self._latest):
                if key not in self._inflight:
                    self._latest.pop(key, None)
        for timer in timers:
            timer.cancel()
