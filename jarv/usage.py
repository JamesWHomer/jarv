import importlib.util
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


def _litellm_token_count(model: str, text: str) -> int:
    if not text:
        return 0
    try:
        from .litellm_compat import import_litellm

        token_counter = import_litellm().token_counter
        return max(0, int(token_counter(model=model, text=text)))
    except Exception:
        return _estimated_token_count(text)


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
    Uses a cheap character-based heuristic by default so request startup is not
    blocked by importing/running LiteLLM tokenization. Set precise=True for
    LiteLLM's token_counter, falling back to the same heuristic.
    """
    count_tokens = _litellm_token_count if precise else (lambda _model, text: _estimated_token_count(text))
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
    output_tokens = _litellm_token_count(model, output_text or "")

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


_price_map_cache: dict | None = None
_price_map_lock = Lock()

# Litellm ships the data under different filenames across versions.
_PRICE_MAP_FILENAMES = [
    "model_prices_and_context_window.json",
    "model_prices_and_context_window_backup.json",
]


def _expand_aliases(price_map: dict) -> dict:
    """Promote alias entries to top-level keys (mirrors litellm's own logic)."""
    to_add: dict = {}
    for entry in price_map.values():
        if not isinstance(entry, dict):
            continue
        for alias in entry.get("aliases") or []:
            if isinstance(alias, str) and alias not in price_map and alias not in to_add:
                to_add[alias] = entry
    price_map.update(to_add)
    return price_map


def _load_price_map() -> dict:
    global _price_map_cache
    if _price_map_cache is not None:
        return _price_map_cache
    with _price_map_lock:
        if _price_map_cache is not None:
            return _price_map_cache
        try:
            spec = importlib.util.find_spec("litellm")
            if spec is not None and spec.origin:
                pkg_dir = Path(spec.origin).parent
                for filename in _PRICE_MAP_FILENAMES:
                    json_path = pkg_dir / filename
                    if json_path.exists():
                        data = json.loads(json_path.read_text(encoding="utf-8"))
                        if isinstance(data, dict) and data:
                            _price_map_cache = _expand_aliases(data)
                            return _price_map_cache
        except Exception:
            pass
        _price_map_cache = {}
        return _price_map_cache


def _litellm_model_info(model: str | None) -> dict | None:
    if not model:
        return None
    price_map = _load_price_map()
    info = price_map.get(model)
    if info is None:
        # Try stripping provider prefix (e.g. "openai/gpt-4o" -> "gpt-4o")
        short = model.split("/", 1)[-1] if "/" in model else None
        if short:
            info = price_map.get(short)
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

