import json
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

from .config import CONFIG_DIR
from .display import console
from .history import isoformat_utc, parse_timestamp, utc_now

_BREAKDOWN_KEYS = ("system", "tools", "history", "tool_io", "reasoning")


def _estimated_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _token_count(_model: str, text: str) -> int:
    return _estimated_token_count(text)


def estimate_item_tokens(model: str, item: dict) -> int:
    """Estimate tokens for one API input item using the len//4 heuristic."""
    return _token_count(model, _item_text(item))


def _item_text(item: dict) -> str:
    """Extract meaningful text from an API input item for token estimation."""
    role = item.get("role")
    typ = item.get("type")
    if role in ("user", "assistant"):
        return str(item.get("content") or "")
    if typ == "function_call":
        return f"{item.get('name', '')} {item.get('arguments', '')}"
    if typ == "function_call_output":
        return str(item.get("output") or "")
    if typ == "reasoning":
        summary = item.get("summary") or []
        return " ".join(str(s) for s in summary) if isinstance(summary, list) else str(summary)
    return json.dumps(item)


def estimate_context_breakdown(
    model: str,
    instructions: str,
    tools: list,
    input_items: list,
    *,
    precise: bool = False,
) -> dict:
    """Estimate token counts split by context category.

    Returns a dict with keys: system, tools, history, tool_io, reasoning.
    Uses a cheap character-based heuristic. ``precise`` remains accepted for
    API compatibility but no longer imports a provider SDK.
    """
    count_tokens = _token_count
    try:
        system_tokens = count_tokens(model, instructions)
        tools_tokens = count_tokens(model, json.dumps(tools))

        history_tokens = 0
        tool_io_tokens = 0
        reasoning_tokens = 0

        for item in input_items:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            typ = item.get("type")
            count = count_tokens(model, _item_text(item))
            if role in ("user", "assistant"):
                history_tokens += count
            elif typ in ("function_call", "function_call_output"):
                tool_io_tokens += count
            elif typ == "reasoning":
                reasoning_tokens += count
            else:
                history_tokens += count

        return {
            "system": system_tokens,
            "tools": tools_tokens,
            "history": history_tokens,
            "tool_io": tool_io_tokens,
            "reasoning": reasoning_tokens,
        }
    except Exception:
        return {k: 0 for k in _BREAKDOWN_KEYS}

USAGE_VERSION = 1
RECENT_REQUEST_LIMIT = 50
GLOBAL_USAGE_RETENTION_DAYS = 90
TOKENS_PER_MILLION = 1_000_000

_usage_lock = Lock()


def usage_file_for(history_path: Path) -> Path:
    return history_path.with_name(history_path.name.replace("history", "usage", 1))


def global_usage_file() -> Path:
    return CONFIG_DIR / "usage.json"


def global_usage_jsonl_file(path: Path | None = None) -> Path:
    return (path or global_usage_file()).with_suffix(".jsonl")


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


def _empty_global_usage() -> dict:
    return {
        "version": USAGE_VERSION,
        "records": [],
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
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    except OSError as e:
        if warn:
            console.print(f"[yellow]Could not save usage data:[/yellow] {e}")


def load_global_usage(path: Path | None = None, warn: bool = True) -> dict:
    path = path or global_usage_file()
    if not path.exists():
        return _empty_global_usage()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        if warn:
            console.print(f"[yellow]Ignoring malformed usage data:[/yellow] {e}")
        return _empty_global_usage()
    if not isinstance(data, dict):
        return _empty_global_usage()

    data.setdefault("version", USAGE_VERSION)
    records = data.get("records")
    if not isinstance(records, list):
        data["records"] = []
    else:
        data["records"] = [record for record in records if isinstance(record, dict)]
        for record in data["records"]:
            _normalize_token_bucket(record, include_request_count=False)
    return data


def save_global_usage(data: dict, path: Path | None = None, warn: bool = True) -> None:
    path = path or global_usage_file()
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

    input_tokens = _first_present(
        _int_value(usage, "input_tokens"),
        _int_value(usage, "prompt_tokens"),
    )
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

    output_tokens = _first_present(
        _int_value(usage, "output_tokens"),
        _int_value(usage, "completion_tokens"),
    )
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


def estimated_usage_from_context(
    model: str,
    context_breakdown: dict | None = None,
    output_text: str | None = None,
) -> dict | None:
    """Build a best-effort usage record when a provider omits token usage."""
    input_tokens = 0
    if isinstance(context_breakdown, dict):
        input_tokens = sum(int(context_breakdown.get(k) or 0) for k in _BREAKDOWN_KEYS)
    output_tokens = _token_count(model, output_text or "")

    if input_tokens <= 0 and output_tokens <= 0:
        return None

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "uncached_input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": 0,
        "total_tokens": input_tokens + output_tokens,
        "estimated": True,
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


def _normalize_token_bucket(bucket: dict, *, include_request_count: bool = True) -> None:
    input_tokens = int(bucket.get("input_tokens") or 0)
    cached_input_tokens = int(bucket.get("cached_input_tokens") or 0)
    if "uncached_input_tokens" not in bucket:
        bucket["uncached_input_tokens"] = max(input_tokens - cached_input_tokens, 0)
    bucket.setdefault("cached_input_tokens", cached_input_tokens)
    bucket.setdefault("output_tokens", 0)
    bucket.setdefault("reasoning_output_tokens", 0)
    bucket.setdefault("total_tokens", input_tokens + int(bucket.get("output_tokens") or 0))
    if include_request_count:
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


def _prune_global_usage_jsonl(
    jsonl_path: Path,
    *,
    now: datetime | None = None,
    warn: bool = True,
) -> None:
    if not jsonl_path.exists():
        return
    cutoff = (now or utc_now()) - timedelta(days=GLOBAL_USAGE_RETENTION_DAYS)
    try:
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as e:
        if warn:
            console.print(f"[yellow]Could not read usage data:[/yellow] {e}")
        return

    kept: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        created_at = parse_timestamp(str(payload.get("created_at") or ""))
        if created_at is None or created_at >= cutoff:
            kept.append(line)

    try:
        jsonl_path.write_text(
            ("\n".join(kept) + "\n") if kept else "",
            encoding="utf-8",
        )
    except OSError as e:
        if warn:
            console.print(f"[yellow]Could not save usage data:[/yellow] {e}")


def _append_global_usage_record_unlocked(
    record: dict,
    path: Path | None = None,
    warn: bool = True,
) -> None:
    jsonl_path = global_usage_jsonl_file(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(record), separators=(",", ":")))
            f.write("\n")
        _prune_global_usage_jsonl(jsonl_path, warn=warn)
    except OSError as e:
        if warn:
            console.print(f"[yellow]Could not save usage data:[/yellow] {e}")


def append_global_usage_record(
    record: dict,
    path: Path | None = None,
    warn: bool = True,
) -> None:
    with _usage_lock:
        _append_global_usage_record_unlocked(record, path, warn=warn)


def load_global_usage_records(
    path: Path | None = None,
    *,
    since: timedelta | None = None,
    now: datetime | None = None,
    warn: bool = True,
) -> list[dict]:
    legacy_records = load_global_usage(path, warn=warn).get("records", [])
    valid_records = [record for record in legacy_records if isinstance(record, dict)] if isinstance(legacy_records, list) else []

    jsonl_path = global_usage_jsonl_file(path)
    if jsonl_path.exists():
        try:
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    _normalize_token_bucket(record, include_request_count=False)
                    valid_records.append(record)
        except (OSError, UnicodeDecodeError) as e:
            if warn:
                console.print(f"[yellow]Could not read usage data:[/yellow] {e}")
    if since is None:
        return valid_records

    cutoff = (now or utc_now()) - since
    filtered: list[dict] = []
    for record in valid_records:
        created_at = parse_timestamp(str(record.get("created_at") or ""))
        if created_at is not None and created_at >= cutoff:
            filtered.append(record)
    return filtered


def aggregate_usage_records(records: list[dict]) -> dict:
    aggregate = _empty_usage(None)
    aggregate["session_id"] = None
    for record in records:
        if not isinstance(record, dict):
            continue
        _normalize_token_bucket(record, include_request_count=False)
        source = str(record.get("source") or "unknown")
        model = str(record.get("model") or "unknown")
        _add_tokens(aggregate.setdefault("totals", {}), record)
        _add_tokens(aggregate.setdefault("sources", {}).setdefault(source, {}), record)
        _add_tokens(aggregate.setdefault("models", {}).setdefault(model, {}), record)
        aggregate["last_request"] = record
        if source == "root":
            aggregate["last_root_request"] = record
    return aggregate


_MODEL_METADATA = {
    "gpt-5.5": (1_050_000, 5.0, 0.5, 30.0),
    "gpt-5.4-mini": (272_000, 0.75, 0.075, 4.5),
    "gpt-5.4-nano": (272_000, 0.2, 0.02, 1.25),
    "claude-opus-4-7": (1_000_000, 5.0, 0.5, 25.0),
    "claude-sonnet-4-6": (1_000_000, 3.0, 0.3, 15.0),
    "claude-haiku-4-5": (200_000, 1.0, 0.1, 5.0),
    "gemini-3.1-pro-preview": (1_048_576, 2.0, 0.2, 12.0),
    "gemini-3-flash-preview": (1_048_576, 0.5, 0.05, 3.0),
    "anthropic/claude-opus-4.7": (1_000_000, 5.0, 0.5, 25.0),
    "anthropic/claude-sonnet-4.6": (1_000_000, 3.0, 0.3, 15.0),
    "anthropic/claude-opus-4.6": (1_000_000, 5.0, 0.5, 25.0),
    "google/gemini-3-flash-preview": (1_048_576, 0.5, 0.05, 3.0),
    "deepseek/deepseek-v3.2": (163_840, 0.28, None, 0.4),
    "google/gemini-2.5-flash": (1_048_576, 0.3, None, 2.5),
    "openai/gpt-oss-120b": (131_072, 0.15, 0.075, 0.6),
    "llama-3.3-70b-versatile": (128_000, 0.59, None, 0.79),
    "llama-3.1-8b-instant": (128_000, 0.05, None, 0.08),
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (
        None, 0.27, None, 0.85,
    ),
    "accounts/fireworks/models/qwen3-8b": (40_960, 0.2, None, 0.2),
}


def _model_info(model: str | None) -> dict | None:
    if not model:
        return None
    values = _MODEL_METADATA.get(model)
    if values is None and "/" in model:
        values = _MODEL_METADATA.get(model.split("/", 1)[-1])
    if values is None:
        return None
    context, input_price, cached_price, output_price = values
    return {
        "max_input_tokens": context,
        "input_price": input_price,
        "cached_input_price": cached_price,
        "output_price": output_price,
    }


def _context_window_from_catalog(model: str, config: dict) -> int | None:
    from .model_catalog import _read_cache

    provider = str(config.get("provider", "openai"))
    suffix = model.split("/", 1)[-1] if model else ""
    for catalog_model in _read_cache(provider):
        model_id = catalog_model.id
        if model_id != model and model_id != suffix and not model_id.endswith(f"/{suffix}"):
            continue
        metadata = catalog_model.metadata if isinstance(catalog_model.metadata, dict) else {}
        for key in ("context_length", "inputTokenLimit"):
            value = metadata.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return None


def known_context_window(model: str | None, config: dict | None = None) -> int | None:
    if model and config:
        catalog_window = _context_window_from_catalog(model, config)
        if catalog_window is not None:
            return catalog_window
    info = _model_info(model)
    window = _int_value(info, "max_input_tokens")
    if window is None or window <= 0:
        return None
    return window


def resolve_context_window(model: str | None, config: dict | None = None) -> int:
    """Return the best-known context window, falling back to config/default."""
    from .config import DEFAULT_CONFIG

    window = known_context_window(model, config=config)
    if window is not None and window > 0:
        return window
    fallback = DEFAULT_CONFIG["context_window_fallback"]
    if config is not None:
        try:
            fallback = int(config.get("context_window_fallback", fallback))
        except (TypeError, ValueError):
            pass
    return max(fallback, 1)


def token_prices_for_model(model: str | None) -> dict[str, float] | None:
    info = _model_info(model)
    if info is None:
        return None

    input_price = _first_present(
        _float_value(info, "input_price"),
    )
    cached_input_price = _first_present(
        _float_value(info, "cached_input_price"),
    )
    output_price = _first_present(
        _float_value(info, "output_price"),
    )

    if input_price is None or output_price is None:
        return None
    if input_price < 0 or output_price < 0:
        return None
    if cached_input_price is not None and cached_input_price < 0:
        return None
    prices = {
        "input": input_price,
        "output": output_price,
    }
    if cached_input_price is not None:
        prices["cached_input"] = cached_input_price
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
    context_breakdown: dict | None = None,
    output_text: str | None = None,
    *,
    record_global: bool = True,
    global_usage_path: Path | None = None,
) -> None:
    try:
        if usage_path is None and not record_global:
            return
        token_usage = usage_from_response(response)
        if token_usage is None:
            token_usage = estimated_usage_from_context(model, context_breakdown, output_text)
        if token_usage is None:
            return

        record = {
            "created_at": isoformat_utc(utc_now()),
            "session_id": session_id,
            "model": model,
            "source": source,
            **token_usage,
        }
        if context_breakdown is not None and any(context_breakdown.get(k, 0) for k in _BREAKDOWN_KEYS):
            record["context_breakdown"] = {k: int(context_breakdown.get(k, 0)) for k in _BREAKDOWN_KEYS}
        estimated_cost = estimate_token_cost_usd(record, model)
        if estimated_cost is not None:
            record["estimated_cost_usd"] = estimated_cost

        with _usage_lock:
            if usage_path is not None:
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
                    del recent[:-RECENT_REQUEST_LIMIT]

                save_usage(data, usage_path, warn=False)
            if record_global:
                _append_global_usage_record_unlocked(record, global_usage_path, warn=False)
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

