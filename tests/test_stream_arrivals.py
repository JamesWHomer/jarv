"""Text arrival timestamps precede rendering and Windows queue consumption."""

from types import SimpleNamespace

import pytest

from jarv import provider
from jarv.cancellation import CancellationToken
from jarv.provider import StreamDone, TextDelta


class ImmediateThread:
    """Run the producer to completion before the consumer drains its queue."""

    def __init__(self, *, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


@pytest.mark.parametrize("platform,with_cancellation", [
    ("linux", False), ("win32", False), ("win32", True),
])
def test_text_events_timestamped_at_ingress_before_delivery(monkeypatch, platform, with_cancellation):
    now = [100.0]
    monkeypatch.setattr(provider, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(provider, "perf_counter", lambda: now[0])
    monkeypatch.setattr(provider, "threading", SimpleNamespace(Thread=ImmediateThread))

    def direct(*_args, **_kwargs):
        now[0] = 105.0
        yield TextDelta("First")
        now[0] = 107.0
        yield TextDelta(" second")
        yield StreamDone({})

    monkeypatch.setattr(provider, "_stream_response_direct", direct)
    token = CancellationToken() if with_cancellation else None
    stream = provider.stream_response(
        object(), {}, "test-model", "system", [], [], cancellation_token=token,
    )
    first = next(stream)
    now[0] = 900.0  # Slow consumer; the Windows producer has already queued both.
    second = next(stream)

    assert first.received_at == 105.0
    assert second.received_at == 107.0
    assert isinstance(next(stream), StreamDone)
    with pytest.raises(StopIteration):
        next(stream)


def test_existing_arrival_timestamp_is_preserved_and_not_part_of_event_equality(monkeypatch):
    monkeypatch.setattr(provider, "_stream_response_direct", lambda *_args: iter([
        TextDelta("pre-stamped", received_at=42.0),
    ]))
    monkeypatch.setattr(provider, "perf_counter", lambda: pytest.fail("already timestamped"))

    events = list(provider.stream_response(object(), {}, "test-model", "system", [], []))

    assert events[0].received_at == 42.0
    assert events == [TextDelta("pre-stamped")]


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_closing_timestamped_stream_promptly_closes_retained_direct_generator(monkeypatch, platform):
    closed = []

    def direct():
        try:
            yield TextDelta("First")
            yield TextDelta(" second")
        finally:
            closed.append(True)

    # Retain the generator so refcount/GC cleanup cannot hide missing close
    # propagation through the timestamping wrapper.
    retained_direct = direct()
    monkeypatch.setattr(provider, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(provider, "_stream_response_direct", lambda *_args: retained_direct)
    stream = provider.stream_response(object(), {}, "test-model", "system", [], [])
    try:
        assert next(stream).delta == "First"
        stream.close()

        assert closed == [True]
    finally:
        retained_direct.close()
