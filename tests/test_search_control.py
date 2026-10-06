import threading
from concurrent.futures import ThreadPoolExecutor
from time import monotonic

import pytest

from jarv import search_control
from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.search_control import (
    SearchBudget,
    SearchCooldownError,
    SearchCoordinator,
    SearchDeadlineExceeded,
)


def test_coordinator_serializes_requests_and_releases_after_exception():
    coordinator = SearchCoordinator()
    queued = threading.Event()
    entered = threading.Event()

    def second_request():
        with SearchBudget(2) as budget:
            queued.set()
            with coordinator.request(budget, 0):
                entered.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with SearchBudget(2) as budget:
            with pytest.raises(ValueError, match="request failed"):
                with coordinator.request(budget, 0):
                    future = pool.submit(second_request)
                    assert queued.wait(1)
                    assert not entered.wait(0.03)
                    raise ValueError("request failed")
        future.result(timeout=1)
    assert entered.is_set()


def test_coordinator_spaces_request_starts():
    coordinator = SearchCoordinator()
    with SearchBudget(2) as budget:
        started = monotonic()
        with coordinator.request(budget, 0.03):
            pass
        with coordinator.request(budget, 0.03):
            assert monotonic() - started >= 0.03


def test_cancellation_interrupts_queued_request(monkeypatch):
    coordinator = SearchCoordinator()
    parent = CancellationToken()
    queued = threading.Event()
    original_wait = coordinator._condition.wait

    def observe_wait(timeout):
        queued.set()
        return original_wait(timeout)

    monkeypatch.setattr(coordinator._condition, "wait", observe_wait)

    def queued_request():
        with SearchBudget(5, parent) as budget:
            with coordinator.request(budget, 0):
                pytest.fail("cancelled request acquired occupied slot")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with SearchBudget(5) as budget, coordinator.request(budget, 0):
            future = pool.submit(queued_request)
            assert queued.wait(1)
            parent.cancel()
            with pytest.raises(TurnCancelled):
                future.result(timeout=1)


def test_cancellation_interrupts_retry_backoff():
    parent = CancellationToken()
    waiting = threading.Event()

    def retry_wait():
        with SearchBudget(10, parent) as budget:
            waiting.set()
            budget.wait(8)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(retry_wait)
        assert waiting.wait(1)
        parent.cancel()
        with pytest.raises(TurnCancelled):
            future.result(timeout=1)


def test_cooldown_rejects_queued_requests_and_records_reason(monkeypatch):
    coordinator = SearchCoordinator()
    queued = threading.Event()
    original_wait = coordinator._condition.wait

    def observe_wait(timeout):
        queued.set()
        return original_wait(timeout)

    monkeypatch.setattr(coordinator._condition, "wait", observe_wait)

    def queued_request():
        with SearchBudget(5) as budget:
            with coordinator.request(budget, 0):
                pytest.fail("cooldown request entered")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with SearchBudget(5) as budget, coordinator.request(budget, 0):
            future = pool.submit(queued_request)
            assert queued.wait(1)
            coordinator.cooldown(30, "human verification")
            with pytest.raises(SearchCooldownError) as exc:
                future.result(timeout=1)
    assert exc.value.reason == "human verification"
    assert 28 < exc.value.remaining <= 30


def test_cooldown_extends_but_never_shortens_and_eventually_expires(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(search_control, "monotonic", lambda: now[0])
    coordinator = SearchCoordinator()
    coordinator.cooldown(10, "challenge")
    coordinator.cooldown(2, "shorter pause")
    with SearchBudget(30) as budget:
        with pytest.raises(SearchCooldownError) as exc:
            with coordinator.request(budget, 0):
                pytest.fail("cooldown request entered")
        assert exc.value.remaining == 10
        assert exc.value.reason == "challenge"
        now[0] += 11
        with coordinator.request(budget, 0):
            pass


def test_deadline_cancels_active_resource_and_check_translates_cancellation():
    closed = threading.Event()
    with SearchBudget(0.03) as budget:
        budget.token.register(closed.set)
        assert closed.wait(1)
        assert budget.token.cancelled
        with pytest.raises(SearchDeadlineExceeded, match="total time limit"):
            budget.check()


def test_deadline_interrupts_queued_request_and_leaves_coordinator_usable():
    coordinator = SearchCoordinator()

    def queued_request():
        with SearchBudget(0.03) as budget:
            with coordinator.request(budget, 0):
                pytest.fail("expired request acquired occupied slot")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with SearchBudget(2) as budget, coordinator.request(budget, 0):
            future = pool.submit(queued_request)
            with pytest.raises(SearchDeadlineExceeded):
                future.result(timeout=1)
        with SearchBudget(2) as budget, coordinator.request(budget, 0):
            pass


def test_deadline_interrupts_spacing_and_retry_backoff():
    coordinator = SearchCoordinator()
    with SearchBudget(2) as budget, coordinator.request(budget, 0):
        pass
    with SearchBudget(0.03) as budget:
        with pytest.raises(SearchDeadlineExceeded):
            with coordinator.request(budget, 10):
                pytest.fail("expired request entered during spacing")
    with SearchBudget(0.03) as budget:
        with pytest.raises(SearchDeadlineExceeded):
            budget.wait(10)


def test_parent_cancellation_takes_priority_over_deadline():
    parent = CancellationToken()
    with SearchBudget(2, parent) as budget:
        budget._expire()
        parent.cancel()
        with pytest.raises(TurnCancelled):
            budget.check()


def test_budget_unregisters_parent_and_stops_timer_on_exit():
    parent = CancellationToken()
    with SearchBudget(5, parent) as budget:
        timer = budget._timer
        assert budget.remaining() <= 5
    parent.cancel()
    assert not budget.token.cancelled
    assert timer.finished.is_set()


def test_parent_cancellation_closes_an_active_resource():
    parent = CancellationToken()
    closed = threading.Event()
    with SearchBudget(5, parent) as budget:
        budget.token.register(closed.set)
        parent.cancel()
        assert closed.is_set()
        with pytest.raises(TurnCancelled):
            budget.check()


def test_already_cancelled_parent_never_starts_timer():
    parent = CancellationToken()
    parent.cancel()
    budget = SearchBudget(1, parent)
    with pytest.raises(TurnCancelled):
        with budget:
            pytest.fail("cancelled budget entered")
    assert budget._timer is None


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_invalid_deadlines_are_rejected(timeout):
    with pytest.raises(ValueError):
        SearchBudget(timeout)


def test_retry_backoff_rechecks_elapsed_time_after_early_wakeup(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(search_control, "monotonic", lambda: now[0])
    waits = []
    with SearchBudget(10) as budget:
        def wake_early(seconds):
            waits.append(seconds)
            now[0] += 0.25 if len(waits) == 1 else seconds

        monkeypatch.setattr(budget._wake, "wait", wake_early)
        budget.wait(1)
    assert waits == [1.0, 0.75]
    assert now[0] == 101.0
