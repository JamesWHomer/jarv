"""Coalesced background work for the session picker; never paints the terminal."""

import threading


class BrowserWorker:
    def __init__(self, deliver):
        self.deliver = deliver
        self.condition = threading.Condition()
        self.pending = {}
        self.versions = {}
        self.closed = False
        self.thread = None

    def current(self, key, version):
        with self.condition:
            return not self.closed and self.versions.get(key) == version

    def submit(self, key, work, *, priority=1):
        with self.condition:
            if self.closed:
                return
            version = self.versions.get(key, 0) + 1
            self.versions[key] = version
            self.pending[key] = (priority, version, work)
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True, name="jarv-sessions")
                self.thread.start()
            self.condition.notify()

    def cancel(self, key):
        with self.condition:
            self.versions[key] = self.versions.get(key, 0) + 1
            self.pending.pop(key, None)

    def close(self):
        with self.condition:
            self.closed = True
            self.pending.clear()
            self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending)
                if self.closed:
                    return
                key = min(self.pending, key=lambda k: self.pending[k][0])
                _, version, work = self.pending.pop(key)
            cancelled = lambda: not self.current(key, version)
            try:
                result = work(cancelled)
                error = None
            except Exception as exc:
                result, error = None, str(exc)
            if not cancelled():
                self.deliver((key, version, result, error))
