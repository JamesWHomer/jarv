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
        self.running = None

    def current(self, key, version):
        with self.condition:
            return not self.closed and self.versions.get(key) == version

    def submit(self, key, work, *, priority=1, preemptible=False):
        with self.condition:
            if self.closed:
                return
            version = self.versions.get(key, 0) + 1
            self.versions[key] = version
            self.pending[key] = (priority, version, work, preemptible)
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True, name="jarv-sessions")
                self.thread.start()
            self.condition.notify()

    def cancel(self, key):
        with self.condition:
            self.versions[key] = self.versions.get(key, 0) + 1
            self.pending.pop(key, None)

    def prioritize(self, key, priority):
        """Promote queued work without restarting an in-flight request."""
        with self.condition:
            if key in self.pending:
                _, version, work, preemptible = self.pending[key]
                self.pending[key] = (priority, version, work, preemptible)
            if self.running is not None and self.running[0] == key:
                self.running = (key, self.running[1], priority)

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
                priority, version, work, preemptible = self.pending.pop(key)
                self.running = (key, version, priority)
            yielded = False

            def cancelled():
                nonlocal yielded
                with self.condition:
                    if self.closed or self.versions.get(key) != version:
                        return True
                    # Neighbour formatting yields between messages when the
                    # selected session needs this worker, then resumes later.
                    if preemptible and any(job[0] < min(0, self.running[2]) for job in self.pending.values()):
                        yielded = True
                    return yielded
            try:
                result = work(cancelled)
                error = None
            except Exception as exc:
                result, error = None, str(exc)
            with self.condition:
                priority = self.running[2]
                self.running = None
                if self.closed or self.versions.get(key) != version:
                    continue
                if yielded:
                    self.pending[key] = (priority, version, work, preemptible)
                    continue
            self.deliver((key, version, result, error))
