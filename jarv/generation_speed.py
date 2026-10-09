"""Read generation throughput from provider-reported inference measurements."""

from __future__ import annotations

import math
from typing import Any


def _value(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _tokens_per_second(tokens: Any, duration: Any, *, units_per_second: float = 1) -> float | None:
    count = _number(tokens)
    elapsed = _number(duration)
    if count is None or not count.is_integer() or elapsed is None or elapsed <= 0:
        return None
    rate = count / elapsed * units_per_second
    return rate if math.isfinite(rate) else None


def _reported_rate(metrics: Any, key: str, duration_key: str) -> float | None:
    rate = _number(_value(metrics, key))
    duration = _value(metrics, duration_key)
    if duration is not None:
        seconds = _number(duration)
        if seconds is None or seconds <= 0:
            return None
    return rate


def provider_output_speed(response: Any) -> float | None:
    """Return server-measured output tok/s, or ``None`` if unreported.

    Groq reports completion time in seconds, llama.cpp reports prediction time
    in milliseconds, and Ollama's native endpoint reports nanoseconds. LM
    Studio's native-compatible endpoint reports the rate directly. No request
    wall time or token estimates are substituted for these measurements.
    """
    usage = _value(response, "usage")
    groq_usage = _value(_value(response, "x_groq"), "usage")
    # Keep nested Groq timing paired with the count from that same report.
    for reported_usage in (groq_usage, usage):
        rate = _tokens_per_second(
            _value(reported_usage, "completion_tokens"),
            _value(reported_usage, "completion_time"),
        )
        if rate is not None:
            return rate

    timings = _value(response, "timings")
    rate = _reported_rate(timings, "predicted_per_second", "predicted_ms")
    if rate is None:
        rate = _tokens_per_second(
            _value(timings, "predicted_n"), _value(timings, "predicted_ms"),
            units_per_second=1_000,
        )
    if rate is not None:
        return rate

    rate = _tokens_per_second(
        _value(response, "eval_count"), _value(response, "eval_duration"),
        units_per_second=1_000_000_000,
    )
    if rate is not None:
        return rate

    return _reported_rate(_value(response, "stats"), "tokens_per_second", "generation_time")
