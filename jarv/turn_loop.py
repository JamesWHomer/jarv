"""Shared stream collection for root and subagent turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .provider import (
    ReasoningDone,
    ReasoningStarted,
    RetryableStreamError,
    StreamDone,
    TextDelta,
    ToolCallDone,
    response_output_text,
)


@dataclass
class StreamCollection:
    reply_text: str = ""
    tool_calls: list[ToolCallDone] = field(default_factory=list)
    reasoning_items: list[ReasoningDone] = field(default_factory=list)
    final_response: Any = None
    final_text: str = ""
    got_text: bool = False
    saw_reasoning: bool = False


def collect_stream_response(
    make_stream: Callable[[], Iterable[Any]],
    *,
    on_event: Callable[[Any, StreamCollection], None] | None = None,
    on_attempt_end: Callable[[StreamCollection, bool], None] | None = None,
    on_retry: Callable[[], None] | None = None,
    max_replays: int = 1,
) -> StreamCollection:
    """Collect normalized provider events, replaying one retryable stream failure."""
    stream_replays = 0
    while True:
        result = StreamCollection()
        retry_stream = False
        try:
            for event in make_stream():
                if on_event is not None:
                    on_event(event, result)
                if isinstance(event, TextDelta):
                    result.reply_text += event.delta
                    result.got_text = True
                elif isinstance(event, ToolCallDone):
                    result.tool_calls.append(event)
                elif isinstance(event, ReasoningStarted):
                    result.saw_reasoning = True
                elif isinstance(event, ReasoningDone):
                    result.saw_reasoning = True
                    result.reasoning_items.append(event)
                elif isinstance(event, StreamDone):
                    result.final_response = event.response

            result.final_text = response_output_text(result.final_response)
            if result.final_text and len(result.final_text) >= len(result.reply_text):
                result.reply_text = result.final_text
                result.got_text = True
            return result
        except RetryableStreamError:
            retry_stream = stream_replays < max_replays
            if not retry_stream:
                raise
            stream_replays += 1
        finally:
            if on_attempt_end is not None:
                on_attempt_end(result, retry_stream)
        if on_retry is not None:
            on_retry()
