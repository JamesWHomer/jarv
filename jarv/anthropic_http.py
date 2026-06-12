"""Direct HTTP transport and protocol conversion for Anthropic Messages."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from .cancellation import CancellationToken
from .http_transport import (
    ProviderHTTPError,
    iter_sse_json,
    request_json,
    send_with_retries,
)
from .unicode_safety import sanitize_json_value


ANTHROPIC_API_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_SUBAGENT_MAX_TOKENS = 16384
_CACHE_CONTROL = {"type": "ephemeral"}
_THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "xhigh": 8192,
    "max": 16384,
}


class AnthropicHTTPError(ProviderHTTPError):
    """An error returned by Anthropic's HTTP API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            "Anthropic",
            message,
            status_code=status_code,
            error_type=error_type,
            request_id=request_id,
        )


def create_client(config: dict, api_key: str):
    """Create a persistent httpx client for Anthropic."""
    import httpx

    base_url = config.get("base_url") or ANTHROPIC_API_URL
    timeout = httpx.Timeout(
        float(config.get("anthropic_timeout", 600)),
        connect=float(config.get("anthropic_connect_timeout", 10)),
    )
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "user-agent": "jarv",
        },
        timeout=timeout,
    )


def list_models(client, *, max_retries: int = 0) -> dict:
    """List all models visible to the current Anthropic account."""
    models: list[dict] = []
    after_id: str | None = None
    while True:
        params: dict[str, Any] = {"limit": 1000}
        if after_id:
            params["after_id"] = after_id
        page = request_json(
            "Anthropic",
            client,
            "GET",
            "/v1/models",
            params=params,
            max_retries=max_retries,
        )
        data = page.get("data")
        if isinstance(data, list):
            models.extend(item for item in data if isinstance(item, dict))
        if not page.get("has_more"):
            break
        next_id = page.get("last_id")
        if not isinstance(next_id, str) or not next_id or next_id == after_id:
            break
        after_id = next_id
    return {"data": models, "has_more": False}


def _append_message(messages: list[dict], role: str, blocks: list[dict]) -> None:
    if not blocks:
        return
    if messages and messages[-1]["role"] == role:
        existing = messages[-1]["content"]
        if isinstance(existing, list):
            existing.extend(blocks)
            return
    messages.append({"role": role, "content": blocks})


def _tool_input(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments if arguments is not None else {}
    try:
        return json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}


def to_messages(input_items: list[dict]) -> list[dict]:
    """Convert Jarv's Responses-style history to Anthropic content blocks."""
    messages: list[dict] = []
    i = 0
    while i < len(input_items):
        item = input_items[i]
        role = item.get("role")
        typ = item.get("type")

        if role in ("user", "assistant"):
            _append_message(
                messages,
                role,
                [{"type": "text", "text": str(item.get("content") or "")}],
            )
            i += 1
            continue

        if typ == "reasoning":
            provider_content = item.get("provider_content")
            if isinstance(provider_content, list):
                blocks = [
                    block
                    for block in provider_content
                    if isinstance(block, dict)
                    and block.get("type") in ("thinking", "redacted_thinking")
                ]
                _append_message(messages, "assistant", blocks)
            i += 1
            continue

        if typ == "function_call":
            blocks = []
            while i < len(input_items) and input_items[i].get("type") == "function_call":
                call = input_items[i]
                call_id = str(call.get("call_id") or call.get("id") or "")
                blocks.append({
                    "type": "tool_use",
                    "id": call_id,
                    "name": str(call.get("name") or ""),
                    "input": _tool_input(call.get("arguments")),
                })
                i += 1
            _append_message(messages, "assistant", blocks)
            continue

        if typ == "function_call_output":
            blocks = []
            while (
                i < len(input_items)
                and input_items[i].get("type") == "function_call_output"
            ):
                result = input_items[i]
                blocks.append({
                    "type": "tool_result",
                    "tool_use_id": str(result.get("call_id") or ""),
                    "content": str(result.get("output") or ""),
                })
                i += 1
            _append_message(messages, "user", blocks)
            continue

        i += 1

    return messages


def to_tools(tools: list[dict]) -> list[dict]:
    result = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if tool.get("type") != "function" or not function.get("name"):
            continue
        converted = {
            "name": function["name"],
            "description": function.get("description", ""),
            "input_schema": function.get("parameters", {"type": "object"}),
        }
        if tool.get("cache_control"):
            converted["cache_control"] = tool["cache_control"]
        result.append(converted)
    return result


def _uses_adaptive_thinking(model: str) -> bool:
    lowered = model.lower()
    return any(version in lowered for version in ("-4-6", "-4.6", "-4-7", "-4.7"))


def _mark_last_block_cached(content: Any) -> list[dict]:
    """Return content blocks with cache_control on the final block."""
    if isinstance(content, str):
        return [{"type": "text", "text": content, "cache_control": _CACHE_CONTROL}]
    if not isinstance(content, list) or not content:
        return [{"type": "text", "text": "", "cache_control": _CACHE_CONTROL}]
    blocks: list[dict] = []
    for block in content:
        blocks.append(dict(block) if isinstance(block, dict) else block)
    last = dict(blocks[-1])
    last["cache_control"] = _CACHE_CONTROL
    blocks[-1] = last
    return blocks


def _apply_prompt_caching(payload: dict) -> None:
    """Place explicit cache breakpoints on tools, system, and stable message prefix."""
    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        last_tool = dict(tools[-1])
        last_tool["cache_control"] = _CACHE_CONTROL
        tools[-1] = last_tool

    instructions = payload.get("system")
    if instructions:
        text = instructions if isinstance(instructions, str) else str(instructions)
        payload["system"] = [{
            "type": "text",
            "text": text,
            "cache_control": _CACHE_CONTROL,
        }]

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    target = messages[-2] if len(messages) >= 2 else messages[0]
    if not isinstance(target, dict):
        return
    target = dict(target)
    target["content"] = _mark_last_block_cached(target.get("content"))
    messages[-2 if len(messages) >= 2 else 0] = target


def _apply_reasoning(payload: dict, model: str, reasoning: dict | None) -> None:
    effort = str((reasoning or {}).get("effort") or "").lower()
    if not effort or effort == "none":
        return
    if _uses_adaptive_thinking(model):
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": effort}
        return
    budget = _THINKING_BUDGETS.get(effort)
    if budget is None:
        raise ValueError(f"Unsupported Anthropic reasoning effort: {effort}")
    payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
    if payload["max_tokens"] <= budget:
        payload["max_tokens"] = budget + 1024


def build_payload(
    config: dict,
    model: str,
    instructions: str,
    tools: list[dict],
    input_items: list[dict],
    *,
    reasoning: dict | None = None,
    stream: bool = False,
    max_tokens: int | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": int(
            max_tokens
            if max_tokens is not None
            else config.get("anthropic_max_tokens", DEFAULT_MAX_TOKENS)
        ),
        "messages": to_messages(input_items),
    }
    if instructions:
        payload["system"] = instructions
    converted_tools = to_tools(tools)
    if converted_tools:
        payload["tools"] = converted_tools
    if stream:
        payload["stream"] = True
    from .provider_catalog import provider_service_tier

    service_tier = provider_service_tier(config, "anthropic")
    if service_tier:
        payload["service_tier"] = service_tier
    _apply_reasoning(payload, model, reasoning)
    if config.get("anthropic_prompt_caching", True):
        _apply_prompt_caching(payload)
    return sanitize_json_value(payload)


def _response_error(response, data: dict | None = None) -> AnthropicHTTPError:
    if data is None:
        try:
            data = response.json()
        except Exception:
            data = {}
    error = data.get("error") if isinstance(data, dict) else None
    error = error if isinstance(error, dict) else {}
    try:
        response_text = response.text
    except Exception:
        response_text = ""
    message = str(error.get("message") or response_text or "request failed")
    return AnthropicHTTPError(
        message,
        status_code=getattr(response, "status_code", None),
        error_type=str(error.get("type") or "") or None,
        request_id=(
            response.headers.get("request-id")
            or response.headers.get("x-request-id")
            or (str(data.get("request_id")) if isinstance(data, dict) and data.get("request_id") else None)
        ),
    )


def _provider_error(exc: ProviderHTTPError) -> AnthropicHTTPError:
    message = str(exc)
    prefix = "Anthropic API error"
    if message.startswith(prefix):
        message = message[len(prefix):].lstrip(": ")
    return AnthropicHTTPError(
        message,
        status_code=exc.status_code,
        error_type=exc.error_type,
        request_id=exc.request_id,
    )


def create_message(
    client,
    payload: dict,
    *,
    cancellation_token: CancellationToken | None = None,
    max_retries: int = 2,
) -> dict:
    """Create one non-streaming Anthropic message."""
    response = send_with_retries(
        client,
        "POST",
        "/v1/messages",
        json_body=payload,
        stream=False,
        cancellation_token=cancellation_token,
        max_retries=max_retries,
    )
    try:
        if response.status_code >= 400:
            raise _response_error(response)
        data = response.json()
        if not isinstance(data, dict):
            raise AnthropicHTTPError("response was not a JSON object")
        return normalize_response(data)
    finally:
        response.close()


def iter_sse(response) -> Iterator[tuple[str, dict]]:
    """Parse an Anthropic SSE response, including multiline data fields."""
    try:
        yield from iter_sse_json("Anthropic", response)
    except ProviderHTTPError as exc:
        raise _provider_error(exc) from exc


def _normalized_usage(usage: dict | None) -> dict:
    source = usage if isinstance(usage, dict) else {}
    uncached = int(source.get("input_tokens") or 0)
    cache_write = int(source.get("cache_creation_input_tokens") or 0)
    cached = int(source.get("cache_read_input_tokens") or 0)
    output = int(source.get("output_tokens") or 0)
    return {
        **source,
        "input_tokens": uncached + cache_write + cached,
        "uncached_input_tokens": uncached + cache_write,
        "cached_input_tokens": cached,
        "output_tokens": output,
        "total_tokens": uncached + cache_write + cached + output,
    }


def normalize_response(data: dict) -> dict:
    content = data.get("content")
    blocks = content if isinstance(content, list) else []
    return {
        **data,
        "content": blocks,
        "output_text": "".join(
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ),
        "usage": _normalized_usage(data.get("usage")),
    }


def stream_message(
    client,
    payload: dict,
    *,
    cancellation_token: CancellationToken | None = None,
    max_retries: int = 2,
) -> Iterator[dict]:
    """Yield normalized protocol events from an Anthropic SSE message."""
    response = send_with_retries(
        client,
        "POST",
        "/v1/messages",
        json_body=payload,
        stream=True,
        cancellation_token=cancellation_token,
        max_retries=max_retries,
    )
    if response.status_code >= 400:
        try:
            response.read()
            data = response.json()
        except Exception:
            data = None
        response.close()
        raise _response_error(response, data)

    unregister = (
        cancellation_token.register(response.close)
        if cancellation_token is not None else lambda: None
    )
    message: dict[str, Any] = {"content": [], "usage": {}}
    blocks: dict[int, dict] = {}
    try:
        for event_name, event in iter_sse(response):
            if cancellation_token is not None:
                cancellation_token.throw_if_cancelled()
            event_type = str(event.get("type") or event_name)

            if event_type == "error":
                raise AnthropicHTTPError(
                    str((event.get("error") or {}).get("message") or "stream failed"),
                    error_type=str((event.get("error") or {}).get("type") or "") or None,
                    request_id=str(event.get("request_id") or "") or None,
                )
            if event_type == "message_start":
                started = event.get("message")
                if isinstance(started, dict):
                    message.update(started)
                    message["content"] = []
                    message["usage"] = dict(started.get("usage") or {})
                continue
            if event_type == "content_block_start":
                index = int(event.get("index") or 0)
                block = dict(event.get("content_block") or {})
                if block.get("type") == "tool_use":
                    block["_partial_json"] = ""
                blocks[index] = block
                if block.get("type") in ("thinking", "redacted_thinking"):
                    yield {"type": "reasoning_started", "id": f"thinking_{index}"}
                elif block.get("type") == "tool_use":
                    yield {
                        "type": "tool_call_started",
                        "id": str(block.get("id") or ""),
                        "name": str(block.get("name") or ""),
                    }
                continue
            if event_type == "content_block_delta":
                index = int(event.get("index") or 0)
                delta = event.get("delta") or {}
                block = blocks.setdefault(index, {})
                delta_type = delta.get("type")
                if delta_type == "text_delta":
                    text = str(delta.get("text") or "")
                    block["text"] = str(block.get("text") or "") + text
                    if text:
                        yield {"type": "text_delta", "delta": text}
                elif delta_type == "thinking_delta":
                    block["thinking"] = (
                        str(block.get("thinking") or "") + str(delta.get("thinking") or "")
                    )
                elif delta_type == "signature_delta":
                    block["signature"] = (
                        str(block.get("signature") or "") + str(delta.get("signature") or "")
                    )
                elif delta_type == "input_json_delta":
                    block["_partial_json"] = (
                        str(block.get("_partial_json") or "")
                        + str(delta.get("partial_json") or "")
                    )
                continue
            if event_type == "content_block_stop":
                index = int(event.get("index") or 0)
                block = blocks.pop(index, {})
                block_type = block.get("type")
                if block_type == "tool_use":
                    raw = str(block.pop("_partial_json", "") or "")
                    if raw:
                        try:
                            block["input"] = json.loads(raw)
                        except json.JSONDecodeError:
                            block["input"] = {}
                            arguments = raw
                        else:
                            arguments = json.dumps(
                                block["input"],
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                    else:
                        arguments = json.dumps(
                            block.get("input") if block.get("input") is not None else {},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    yield {
                        "type": "tool_call",
                        "id": str(block.get("id") or ""),
                        "name": str(block.get("name") or ""),
                        "arguments": arguments,
                    }
                elif block_type in ("thinking", "redacted_thinking"):
                    yield {
                        "type": "reasoning_done",
                        "id": f"thinking_{index}",
                        "provider_content": [block],
                    }
                message.setdefault("content", []).append(block)
                continue
            if event_type == "message_delta":
                delta = event.get("delta")
                if isinstance(delta, dict):
                    message.update(delta)
                usage = event.get("usage")
                if isinstance(usage, dict):
                    message.setdefault("usage", {}).update(usage)
                continue
            if event_type == "message_stop":
                yield {"type": "done", "response": normalize_response(message)}
                return
            # ping and forward-compatible unknown events are intentionally ignored.
    finally:
        unregister()
        response.close()

    if cancellation_token is not None:
        cancellation_token.throw_if_cancelled()
    raise AnthropicHTTPError("stream ended before message_stop")
