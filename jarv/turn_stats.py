"""Agent-turn token totals and measured final-response throughput for either UI."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from rich.text import Text

from .generation_speed import provider_output_speed
from .turn_summary_settings import TURN_SUMMARY_FIELDS
from .usage import (
    estimated_usage_from_context, format_cost, format_int,
    usage_cost_summary, usage_from_response,
)


def _value(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _has_count(usage: Any, *keys: str) -> bool:
    for key in keys:
        value = _value(usage, key)
        if value is None or isinstance(value, bool):
            continue
        try:
            if int(value) >= 0:
                return True
        except (TypeError, ValueError, OverflowError):
            pass
    return False


def _duration(value: float | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


@dataclass
class _TokenCounts:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_reported: bool = False
    reasoning_reported: bool = False
    available: bool = False
    estimated: bool = False

    def include(self, other: _TokenCounts) -> None:
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens"):
            current, incoming = getattr(self, key), getattr(other, key)
            setattr(self, key, current + incoming if current is not None and incoming is not None else None)
        self.cache_reported |= other.cache_reported
        self.reasoning_reported |= other.reasoning_reported
        self.available |= other.available
        self.estimated |= other.estimated


def _response_token_counts(
    response, *, model, context_breakdown=None, output_text=None, provider=None,
) -> _TokenCounts:
    raw_usage = _value(response, "usage")
    usage = usage_from_response(response)
    estimated = usage is None
    if estimated:
        usage = estimated_usage_from_context(model, context_breakdown, output_text)
        input_known = context_breakdown is not None
        output_known = output_text is not None
        total_known = input_known and output_known
    else:
        input_known = _has_count(raw_usage, "input_tokens", "prompt_tokens")
        output_known = _has_count(raw_usage, "output_tokens", "completion_tokens")
        total_known = _has_count(raw_usage, "total_tokens") or (input_known and output_known)
    if usage is None:
        return _TokenCounts()

    output_tokens = usage["output_tokens"]
    total_tokens = usage["total_tokens"]
    # Gemini candidate output excludes thoughts; count them once in turn totals.
    if not estimated and str(provider or "").lower() == "gemini":
        output_tokens += usage["reasoning_output_tokens"]
        if not _has_count(raw_usage, "total_tokens"):
            total_tokens += usage["reasoning_output_tokens"]
    input_details = _value(raw_usage, "input_tokens_details") or _value(raw_usage, "prompt_tokens_details")
    output_details = _value(raw_usage, "output_tokens_details") or _value(raw_usage, "completion_tokens_details")
    cache_known = not estimated and (
        _has_count(raw_usage, "cached_input_tokens", "cached_tokens")
        or _has_count(input_details, "cached_tokens", "cached_input_tokens")
    )
    reasoning_known = not estimated and (
        _has_count(raw_usage, "reasoning_output_tokens")
        or _has_count(output_details, "reasoning_tokens", "reasoning_output_tokens")
    )
    return _TokenCounts(
        input_tokens=usage["input_tokens"] if input_known else None,
        output_tokens=output_tokens if output_known else None,
        total_tokens=total_tokens if total_known else None,
        cached_tokens=usage["cached_input_tokens"] if cache_known else None,
        reasoning_tokens=usage["reasoning_output_tokens"] if reasoning_known else None,
        cache_reported=cache_known,
        reasoning_reported=reasoning_known,
        available=True,
        estimated=estimated,
    )


class TurnSummaryStats:
    """Accumulate completed model calls within one user-facing agent turn.

    Unknown counts remain unknown when summed. Only the final response is kept
    for speed measurements; request times exclude all intervening tool work.
    """

    def __init__(self) -> None:
        self.request_count = 0
        self.elapsed_seconds: float | None = 0.0
        self._counts = _TokenCounts()
        self._last_response = None
        self._last_provider: str | None = None
        self._last_stream_result = None
        self._last_usage: dict | None = None

    def add_response(
        self,
        response,
        *,
        model: str,
        elapsed_seconds: float | None,
        context_breakdown: dict | None = None,
        output_text: str | None = None,
        provider: str | None = None,
        stream_result=None,
    ) -> None:
        counts = _response_token_counts(
            response, model=model, context_breakdown=context_breakdown,
            output_text=output_text, provider=provider,
        )
        if self.request_count:
            self._counts.include(counts)
        else:
            self._counts = counts
        self.request_count += 1
        seconds = _duration(elapsed_seconds)
        if elapsed_seconds == 0 and not isinstance(elapsed_seconds, bool):
            seconds = 0.0
        if self.elapsed_seconds is None or seconds is None:
            self.elapsed_seconds = None
        else:
            combined = self.elapsed_seconds + seconds
            self.elapsed_seconds = combined if math.isfinite(combined) else None
        self._last_response = response
        self._last_provider = provider
        self._last_stream_result = stream_result
        self._last_usage = usage_from_response(response)


def _stream_output_speed(response, usage, stream_result, provider) -> float | None:
    """Estimate visible-text speed over a usable first-to-last arrival window.

    Stream chunks are not tokens. Apportion the provider's visible token count
    by text length to exclude the first chunk, whose generation preceded this
    window. This remains an estimate even with exact response token counts.
    """
    if stream_result is None or usage is None:
        return None
    seconds = _duration(stream_result.text_stream_seconds)
    text = stream_result.streamed_text
    first_chunk = stream_result.first_text_chunk
    if (
        seconds is None or seconds < 0.1
        or stream_result.text_chunk_count < 2
        or not text or not first_chunk or not text.startswith(first_chunk)
        or text != stream_result.reply_text
        or (stream_result.final_text and text != stream_result.final_text)
        or stream_result.tool_calls
    ):
        return None
    raw_usage = _value(response, "usage")
    if not _has_count(raw_usage, "output_tokens", "completion_tokens"):
        return None
    visible_tokens = usage["output_tokens"]
    if str(provider or "").lower() != "gemini":
        details = _value(raw_usage, "output_tokens_details") or _value(raw_usage, "completion_tokens_details")
        reasoning_known = (
            _has_count(raw_usage, "reasoning_output_tokens")
            or _has_count(details, "reasoning_tokens", "reasoning_output_tokens")
        )
        if stream_result.saw_reasoning and not reasoning_known:
            return None
        visible_tokens -= usage["reasoning_output_tokens"]
    try:
        count = float(visible_tokens)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(count):
        return None
    remaining_tokens = count * ((len(text) - len(first_chunk)) / len(text))
    if remaining_tokens < 1:
        return None
    rate = remaining_tokens / seconds
    return rate if math.isfinite(rate) else None


def format_turn_summary(
    response: Any,
    *,
    model: str,
    elapsed_seconds: float | None,
    fields: Collection[str],
    context_breakdown: dict | None = None,
    output_text: str | None = None,
    provider: str | None = None,
    session_usage: dict | None = None,
    stream_result=None,
    turn_stats: TurnSummaryStats | None = None,
) -> Text | None:
    """Format a completed agent turn, or one response without an accumulator.

    ``elapsed_seconds`` covers the complete request up to the final response,
    before rendering, saving usage, or executing tools. Speed uses server
    generation timing when reported, otherwise an explicitly estimated visible
    text stream rate. Full request time is never used as generation time.
    """
    selected = set(fields)
    if not selected:
        return None
    multiple_responses = turn_stats is not None and turn_stats.request_count > 1
    if turn_stats is not None and turn_stats.request_count:
        counts = turn_stats._counts
        elapsed_seconds = turn_stats.elapsed_seconds
        response = turn_stats._last_response
        provider = turn_stats._last_provider
        stream_result = turn_stats._last_stream_result
        usage = turn_stats._last_usage
    else:
        counts = _response_token_counts(
            response, model=model, context_breakdown=context_breakdown,
            output_text=output_text, provider=provider,
        )
        usage = usage_from_response(response)

    parts: dict[str, Text] = {}
    if counts.available:
        token_counts = Text(style="dim")
        for value, label in (
            (counts.input_tokens, "in"),
            (counts.output_tokens, "out"),
            (counts.total_tokens, "total"),
        ):
            if token_counts:
                token_counts.append(" · ")
            known = value is not None
            token_counts.append(format_int(value) if known else "?", style="bold" if known else "dim")
            token_counts.append(f" {label}")
        parts["tokens"] = token_counts

        # Details are independent fields, including when the main token counts
        # are hidden. Missing metadata is not a measured zero.
        if counts.cache_reported:
            cached = format_int(counts.cached_tokens) if counts.cached_tokens is not None else "?"
            parts["cache"] = Text(f"{cached} cached", style="dim")
        if counts.reasoning_reported:
            reasoning = format_int(counts.reasoning_tokens) if counts.reasoning_tokens is not None else "?"
            parts["reasoning"] = Text(f"{reasoning} reasoning", style="dim")
    else:
        parts["tokens"] = Text("token usage unavailable", style="dim")

    seconds = _duration(elapsed_seconds)
    if seconds is not None:
        duration = f"{seconds * 1000:.1f}ms" if seconds < 0.01 else f"{seconds:.2f}s"
        parts["time"] = Text(duration, style="dim")
    else:
        parts["time"] = Text("time unavailable", style="dim")
    parts["speed"] = Text("tok/s unavailable", style="dim")
    if "speed" in selected:
        rate = provider_output_speed(response)
        if rate is not None:
            parts["speed"] = Text(f"{rate:,.1f} tok/s (server)", style="dim")
        else:
            rate = _stream_output_speed(response, usage, stream_result, provider)
            if rate is not None:
                parts["speed"] = Text(f"~{rate:,.1f} tok/s (stream)", style="dim")
        if multiple_responses:
            parts["speed"] = Text("final response ", style="dim") + parts["speed"]

    totals = _value(session_usage, "totals")
    if isinstance(totals, dict) and _has_count(totals, "total_tokens"):
        parts["session"] = Text(f"{format_int(int(totals['total_tokens']))} session tokens", style="dim")
    else:
        parts["session"] = Text("session usage unavailable", style="dim")
    cost = usage_cost_summary(totals) if isinstance(totals, dict) else None
    if cost and (cost["exact_requests"] or cost["estimated_requests"] or cost["has_tracked_cost"]):
        label = "session cost" if cost["exact_requests"] and not cost["estimated_requests"] else "est. session cost"
        cost_part = Text(f"{label} {format_cost(cost['total_usd'])}", style="green")
        if cost["unknown_requests"] or cost["contract_requests"]:
            cost_part.append(" (incomplete)", style="yellow")
        parts["cost"] = cost_part
    else:
        parts["cost"] = Text("session cost unavailable", style="dim")

    visible = [parts[field.name] for field in TURN_SUMMARY_FIELDS if field.name in selected and field.name in parts]
    if not visible:
        return None
    line = Text("Turn: ", style="dim")
    line.append_text(Text(" · ", style="dim").join(visible))
    if counts.estimated and "tokens" in selected:
        line.append(" · usage estimated", style="yellow")
    return line
