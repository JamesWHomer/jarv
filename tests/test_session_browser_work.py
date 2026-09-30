"""Ordering and cancellation for speculative session previews."""

import threading

from jarv.session_browser_work import BrowserWorker


def test_neighbour_yields_to_selected_preview_then_retries():
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    results, attempts = [], []

    def deliver(result):
        results.append(result)
        if len(results) == 2:
            completed.set()

    worker = BrowserWorker(deliver)

    def neighbour(cancelled):
        attempts.append(1)
        if len(attempts) == 1:
            entered.set()
            assert release.wait(2)
            assert cancelled()
            return None
        assert not cancelled()
        return "warm"

    try:
        worker.submit("neighbour", neighbour, priority=0, preemptible=True)
        assert entered.wait(2)
        worker.submit("selected", lambda cancelled: "visible", priority=-2)
        release.set()
        assert completed.wait(2)
        assert [(key, result, error) for key, _, result, error in results] == [
            ("selected", "visible", None), ("neighbour", "warm", None)]
        assert len(attempts) == 2
    finally:
        release.set()
        worker.close()
        worker.thread.join(timeout=2)


def test_selecting_an_inflight_neighbour_promotes_it_without_restarting():
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    results, attempts = [], []

    def deliver(result):
        results.append(result)
        if len(results) == 2:
            completed.set()

    worker = BrowserWorker(deliver)

    def neighbour(cancelled):
        attempts.append(1)
        entered.set()
        assert release.wait(2)
        assert not cancelled()
        return "new selection"

    try:
        worker.submit("neighbour", neighbour, priority=1, preemptible=True)
        assert entered.wait(2)
        worker.submit("other", lambda cancelled: "other", priority=0)
        worker.prioritize("neighbour", -2)
        release.set()
        assert completed.wait(2)
        assert [result[2] for result in results] == ["new selection", "other"]
        assert all(result[3] is None for result in results)
        assert len(attempts) == 1
    finally:
        release.set()
        worker.close()
        worker.thread.join(timeout=2)


def test_selecting_a_queued_neighbour_moves_it_ahead_of_other_work():
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    results = []

    def deliver(result):
        results.append(result)
        if len(results) == 3:
            completed.set()

    worker = BrowserWorker(deliver)

    def blocked(cancelled):
        entered.set()
        assert release.wait(2)
        return "first"

    try:
        worker.submit("first", blocked)
        assert entered.wait(2)
        worker.submit("nearby", lambda cancelled: "nearby", priority=0)
        worker.submit("selected", lambda cancelled: "selected", priority=1, preemptible=True)
        worker.prioritize("selected", -2)
        release.set()
        assert completed.wait(2)
        assert [result[2] for result in results] == ["first", "selected", "nearby"]
    finally:
        release.set()
        worker.close()
        worker.thread.join(timeout=2)
