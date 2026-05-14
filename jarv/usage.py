import json
from pathlib import Path
from threading import Lock
from typing import Any

from .display import console
from .history import isoformat_utc, utc_now

USAGE_VERSION = 1
RECENT_REQUEST_LIElastic License 2.0 = 50
TOKENS_PER_MILLION = 1_000_000

_usage_lock = Lock()


def usage_file_for(history_path: Path) -> Path:
    return history_path.with_name(history_path.name.replace("history", "usage", 1))


def _empty_usage(session_id: str | None = None) -> dict:
    return {
        "version": USAGE_VERSION,
        "session_id": session_id,
        "totals": {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "uncached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
        },
        "sources": {},
        "models": {},
        "last_request": None,
        "last_root_request": None,
        "recent_requests": [],
    }


def load_usage(path: Path, session_id: str | None = None, warn: bool = True) -> dict:
    if not path.exists():
        return _empty_usage(session_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        if warn:
            console.print(f"[yellow]Ignoring malformed usage data:[/yellow] {e}")
        return _empty_usage(session_id)
    if not isinstance(data, dict):
        return _empty_usage(session_id)

    empty = _empty_usage(session_id)
    for key, value in empty.items():
        data.setdefault(key, value)
    data["session_id"] = data.get("session_id") or session_id
    _normalize_usage_data(data)
    return data


def save_usage(data: dict, path: Path, warn: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        if warn:
            console.print(f"[yellow]Could not save usage data:[/yellow] {e}")


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _int_value(obj: Any, key: str) -> int | None:
    value = _value(obj, key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_value(obj: Any, key: str) -> float | None:
    value = _value(obj, key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def usage_from_response(response: Any) -> dict | None:
    usage = _value(response, "usage")
    if usage is None:
        return None

    input_tokens = _int_value(usage, "input_tokens")
    input_details = _value(usage, "input_tokens_details") or _value(usage, "prompt_tokens_details")
    cached_input_tokens = _first_present(
        _int_value(usage, "cached_input_tokens"),
        _int_value(usage, "cached_tokens"),
        _int_value(input_details, "cached_tokens"),
        _int_value(input_details, "cached_input_tokens"),
    )
    if cached_input_tokens is None:
        cached_input_tokens = 0
    if input_tokens is not None:
        cached_input_tokens = min(max(cached_input_tokens, 0), max(input_tokens, 0))
    uncached_input_tokens = None
    if input_tokens is not None:
        uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)

    output_tokens = _int_value(usage, "output_tokens")
    output_details = _value(usage, "output_tokens_details") or _value(usage, "completion_tokens_details")
    reasoning_output_tokens = _first_present(
        _int_value(usage, "reasoning_output_tokens"),
        _int_value(output_details, "reasoning_tokens"),
        _int_value(output_details, "reasoning_output_tokens"),
    )
    total_tokens = _int_value(usage, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    return {
        "input_tokens": input_tokens or 0,
        "cached_input_tokens": cached_input_tokens or 0,
        "uncached_input_tokens": uncached_input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "reasoning_output_tokens": reasoning_output_tokens or 0,
        "total_tokens": total_tokens or 0,
    }


def _add_tokens(bucket: dict, record: dict) -> None:
    for key in (
        "input_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "total_tokens",
    ):
        bucket[key] = int(bucket.get(key, 0)) + int(record.get(key, 0))
    if "estimated_cost_usd" in record:
        bucket["estimated_cost_usd"] = float(bucket.get("estimated_cost_usd", 0.0)) + float(
            record.get("estimated_cost_usd", 0.0)
        )
    bucket["request_count"] = int(bucket.get("request_count", 0)) + 1


def _normalize_token_bucket(bucket: dict) -> None:
    input_tokens = int(bucket.get("input_tokens") or 0)
    cached_input_tokens = int(bucket.get("cached_input_tokens") or 0)
    if "uncached_input_tokens" not in bucket:
        bucket["uncached_input_tokens"] = max(input_tokens - cached_input_tokens, 0)
    bucket.setdefault("cached_input_tokens", cached_input_tokens)
    bucket.setdefault("output_tokens", 0)
    bucket.setdefault("reasoning_output_tokens", 0)
    bucket.setdefault("total_tokens", input_tokens + int(bucket.get("output_tokens") or 0))
    bucket.setdefault("request_count", 0)


def _normalize_usage_data(data: dict) -> None:
    totals = data.get("totals")
    if isinstance(totals, dict):
        _normalize_token_bucket(totals)
    for section in ("sources", "models"):
        buckets = data.get(section)
        if isinstance(buckets, dict):
            for bucket in buckets.values():
                if isinstance(bucket, dict):
                    _normalize_token_bucket(bucket)
    for key in ("last_request", "last_root_request"):
        record = data.get(key)
        if isinstance(record, dict):
            _normalize_token_bucket(record)
    recent = data.get("recent_requests")
    if isinstance(recent, list):
        for record in recent:
            if isinstance(record, dict):
                _normalize_token_bucket(record)


def _litellm_model_info(model: str | None) -> dict | None:
    if not model:
        return None
    try:
        from litellm import get_model_info
        info = get_model_info(model=model)
    except Exception:
        return None
    return info if isinstance(info, dict) else None


def known_context_window(model: str | None) -> int | None:
    info = _litellm_model_info(model)
    window = _int_value(info, "max_input_tokens")
    if window is None or window <= 0:
        return None
    return window


def token_prices_for_model(model: str | None) -> dict[str, float] | None:
    info = _litellm_model_info(model)
    if info is None:
        return None

    input_price = _first_present(
        _float_value(info, "input_cost_per_token"),
        _float_value(info, "prompt_cost_per_token"),
    )
    cached_input_price = _first_present(
        _float_value(info, "cache_read_input_token_cost"),
        _float_value(info, "cached_input_cost_per_token"),
    )
    output_price = _first_present(
        _float_value(info, "output_cost_per_token"),
        _float_value(info, "completion_cost_per_token"),
    )

    if input_price is None or output_price is None:
        return None
    if input_price < 0 or output_price < 0:
        return None
    if cached_input_price is not None and cached_input_price < 0:
        return None
    prices = {
        "input": input_price * TOKENS_PER_MILLION,
        "output": output_price * TOKENS_PER_MILLION,
    }
    if cached_input_price is not None:
        prices["cached_input"] = cached_input_price * TOKENS_PER_MILLION
    return prices


def estimate_token_cost_usd(record: dict, model: str | None) -> float | None:
    prices = token_prices_for_model(model)
    if prices is None:
        return None

    input_tokens = int(record.get("input_tokens") or 0)
    cached_input_tokens = int(record.get("cached_input_tokens") or 0)
    cached_input_tokens = min(max(cached_input_tokens, 0), max(input_tokens, 0))
    cached_input_price = prices.get("cached_input")
    if cached_input_tokens and cached_input_price is None:
        return None
    uncached_input_tokens = record.get("uncached_input_tokens")
    if uncached_input_tokens is None:
        uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    else:
        uncached_input_tokens = max(int(uncached_input_tokens or 0), 0)
    output_tokens = int(record.get("output_tokens") or 0)

    return (
        (uncached_input_tokens * prices["input"])
        + (cached_input_tokens * (cached_input_price or 0.0))
        + (output_tokens * prices["output"])
    ) / TOKENS_PER_MILLION


def record_response_usage(
    usage_path: Path | None,
    session_id: str | None,
    model: str,
    response: Any,
    source: str,
) -> None:
    try:
        if usage_path is None:
            return
        token_usage = usage_from_response(response)
        if token_usage is None:
            return

        record = {
            "created_at": isoformat_utc(utc_now()),
            "model": model,
            "source": source,
            **token_usage,
        }
        estimated_cost = estimate_token_cost_usd(record, model)
        if estimated_cost is not None:
            record["estimated_cost_usd"] = estimated_cost

        with _usage_lock:
            data = load_usage(usage_path, session_id, warn=False)
            data["version"] = USAGE_VERSION
            data["session_id"] = data.get("session_id") or session_id
            data["updated_at"] = record["created_at"]

            _add_tokens(data.setdefault("totals", {}), record)
            _add_tokens(data.setdefault("sources", {}).setdefault(source, {}), record)
            _add_tokens(data.setdefault("models", {}).setdefault(model, {}), record)

            data["last_request"] = record
            if source == "root":
                data["last_root_request"] = record

            recent = data.setdefault("recent_requests", [])
            if isinstance(recent, list):
                recent.append(record)
                del recent[:-RECENT_REQUEST_LIElastic License 2.0]

            save_usage(data, usage_path, warn=False)
    except Exception:
        return


def format_int(value: int | None) -> str:
    return f"{int(value or 0):,}"


def format_cost(value: float | None) -> str:
    if value is None:
        return "Unknown"
    if value == 0:
        return "$0.00"
    if abs(value) < 0.01:
        return f"${value:.4f}"
    if abs(value) < 1:
        return f"${value:.3f}"
    return f"${value:.2f}"

