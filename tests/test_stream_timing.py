import pytest

from jarv import turn_loop
from jarv.provider import ReasoningStarted, RetryableStreamError, StreamDone, TextDelta, ToolCallDone


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(turn_loop, "perf_counter", clock)
    return clock


def test_stream_duration_includes_setup_and_stops_before_completion_ui(clock, monkeypatch):
    observed = []

    def make_stream():
        # Ordinary function work before the iterator is returned is timed too.
        clock.advance(2.0)

        def events():
            clock.advance(3.0)
            yield TextDelta("answer")
            clock.advance(4.0)
            yield StreamDone(response={"output_text": "answer"})
            clock.advance(50.0)  # Provider iterator cleanup after completion.

        return events()

    def on_event(event, result):
        if isinstance(event, StreamDone):
            observed.append(result.elapsed_seconds)
            clock.advance(100.0)

    def extract_text(response):
        clock.advance(200.0)
        return response["output_text"]

    def on_attempt_end(result, retry):
        observed.append(result.elapsed_seconds)
        clock.advance(400.0)

    monkeypatch.setattr(turn_loop, "response_output_text", extract_text)
    result = turn_loop.collect_stream_response(
        make_stream, on_event=on_event, on_attempt_end=on_attempt_end
    )

    assert result.elapsed_seconds == 9.0
    assert observed == [9.0, 9.0]
    assert result.reply_text == "answer"


def test_stream_duration_restarts_after_replay_and_retry_cleanup(clock):
    attempts = []
    ended = []

    def make_stream():
        attempts.append(True)
        if len(attempts) == 1:
            clock.advance(10.0)
            yield TextDelta("discarded")
            raise RetryableStreamError("retry")
        clock.advance(2.5)
        yield TextDelta("answer")
        clock.advance(1.5)
        yield StreamDone(response={"output_text": "answer", "usage": {"output_tokens": 20}})

    def on_attempt_end(result, retry):
        ended.append((result.elapsed_seconds, retry))
        clock.advance(100.0)

    result = turn_loop.collect_stream_response(
        make_stream,
        on_attempt_end=on_attempt_end,
        on_retry=lambda: clock.advance(200.0),
    )

    assert result.elapsed_seconds == 4.0
    assert result.reply_text == "answer"
    assert result.final_response["usage"]["output_tokens"] == 20
    assert ended == [(None, True), (4.0, False)]


def test_stream_duration_without_done_stops_before_extraction_and_finalizer(clock, monkeypatch):
    ended = []

    def make_stream():
        clock.advance(1.0)
        yield TextDelta("answer")
        clock.advance(2.0)

    def extract_text(response):
        clock.advance(100.0)
        return ""

    def on_attempt_end(result, retry):
        ended.append(result.elapsed_seconds)
        clock.advance(200.0)

    monkeypatch.setattr(turn_loop, "response_output_text", extract_text)
    result = turn_loop.collect_stream_response(make_stream, on_attempt_end=on_attempt_end)

    assert result.elapsed_seconds == 3.0
    assert result.reply_text == "answer"
    assert ended == [3.0]


@pytest.mark.parametrize("completed", [False, True])
def test_stream_duration_is_never_negative(clock, completed):
    def make_stream():
        clock.advance(-1.0)
        if completed:
            yield StreamDone(response={})

    result = turn_loop.collect_stream_response(make_stream)

    assert result.elapsed_seconds == 0.0


def test_text_stream_duration_excludes_initial_wait_and_completion_delay(clock):
    def make_stream():
        clock.advance(40.0)
        yield ReasoningStarted("reasoning")
        clock.advance(10.0)
        yield TextDelta("")
        yield TextDelta("First chunk")
        clock.advance(0.5)
        yield TextDelta(" then")
        clock.advance(1.5)
        yield TextDelta(" last.")
        clock.advance(20.0)
        yield StreamDone({"output_text": "First chunk then last."})

    result = turn_loop.collect_stream_response(make_stream)

    assert result.elapsed_seconds == 72.0
    assert result.text_stream_seconds == 2.0
    assert result.first_text_chunk == "First chunk"
    assert result.streamed_text == "First chunk then last."
    assert result.text_chunk_count == 3


def test_text_stream_timestamps_precede_callback_and_ignore_queued_render_delay(clock):
    def make_stream():
        clock.advance(10.0)
        yield TextDelta("First", received_at=110.0)
        yield TextDelta(" second", received_at=112.0)
        yield StreamDone({})

    def on_event(event, result):
        if isinstance(event, TextDelta):
            # A slow UI drains already timestamped provider events much later.
            clock.advance(100.0)
            assert result.text_chunk_count in (1, 2)

    result = turn_loop.collect_stream_response(make_stream, on_event=on_event)

    assert result.text_stream_seconds == 2.0
    assert result.elapsed_seconds == 210.0


def test_text_stream_fallback_stops_at_arrival_before_last_callback(clock):
    def make_stream():
        clock.advance(10.0)
        yield TextDelta("First")
        clock.advance(2.0)
        yield TextDelta(" second")
        yield StreamDone({})

    def on_event(event, _result):
        if isinstance(event, TextDelta) and event.delta == " second":
            clock.advance(100.0)

    result = turn_loop.collect_stream_response(make_stream, on_event=on_event)

    assert result.text_stream_seconds == 2.0


@pytest.mark.parametrize("events,streamed,first,count", [
    ([TextDelta("Only chunk")], "Only chunk", "Only chunk", 1),
    ([TextDelta("")], "", "", 0),
    ([ToolCallDone("call_1", "call_1", "run_command", "{}")], "", "", 0),
    ([], "", "", 0),
])
def test_one_chunk_or_no_text_has_no_measurable_decode_interval(clock, events, streamed, first, count):
    result = turn_loop.collect_stream_response(lambda: iter([*events, StreamDone({})]))

    assert result.text_stream_seconds is None
    assert result.streamed_text == streamed
    assert result.first_text_chunk == first
    assert result.text_chunk_count == count


def test_recovery_text_does_not_inflate_observed_stream(clock):
    def make_stream():
        yield TextDelta("Partial")
        clock.advance(1.0)
        yield TextDelta(" answer")
        clock.advance(100.0)
        yield StreamDone({"output_text": "Partial answer with an unstreamed recovered ending"})

    result = turn_loop.collect_stream_response(make_stream)

    assert result.reply_text == "Partial answer with an unstreamed recovered ending"
    assert result.streamed_text == "Partial answer"
    assert result.first_text_chunk == "Partial"
    assert result.text_chunk_count == 2
    assert result.text_stream_seconds == 1.0


def test_text_stream_measurements_reset_after_replay(clock):
    attempts = []

    def make_stream():
        attempts.append(True)
        if len(attempts) == 1:
            yield TextDelta("Discarded")
            clock.advance(20.0)
            yield TextDelta(" partial")
            raise RetryableStreamError("retry")
        clock.advance(10.0)
        yield TextDelta("New")
        clock.advance(2.0)
        yield TextDelta(" answer")
        yield StreamDone({})

    result = turn_loop.collect_stream_response(make_stream)

    assert result.streamed_text == "New answer"
    assert result.first_text_chunk == "New"
    assert result.text_chunk_count == 2
    assert result.text_stream_seconds == 2.0
    assert result.elapsed_seconds == 12.0


def test_zero_text_interval_stays_zero(clock):
    result = turn_loop.collect_stream_response(
        lambda: iter([TextDelta("First"), TextDelta(" second"), StreamDone({})])
    )

    assert result.text_chunk_count == 2
    assert result.text_stream_seconds == 0.0
