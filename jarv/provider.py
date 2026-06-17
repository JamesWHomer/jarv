"""Multi-provider abstraction layer over direct HTTP transports."""

import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator

from .provider_catalog import KEY_PATTERNS, LOCAL_PROVIDERS, PROVIDERS
from .provider_auth import resolve_api_key
from .response_items import responses_input_id
from .tool_schemas import strict_openai_tools
from .tool_outputs import to_chat_tool_content
from .unicode_safety import sanitize_json_value
from .cancellation import CancellationToken, TurnCancelled
from .http_transport import ProviderHTTPError


# ---------------------------------------------------------------------------
# Normalized stream events
# ---------------------------------------------------------------------------

@dataclass
class TextDelta:
    delta: str


@dataclass
class ToolCallStarted:
    id: str
    call_id: str
    name: str


@dataclass
class ToolCallDone:
    id: str
    call_id: str
    name: str
    arguments: str
    provider_content: list[dict] | None = None


@dataclass
class ReasoningDone:
    id: str
    summary: list
    provider_content: list[dict] | None = None


@dataclass
class ReasoningStarted:
    id: str


@dataclass
class StreamDone:
    response: Any


class ProviderError(Exception):
    pass


class RetryableStreamError(ProviderError):
    """A provider stream failed before its response could be recovered."""


_OPENAI_RECOVERY_ATTEMPTS = 16
_OPENAI_RECOVERY_MAX_DELAY = 2.0


def _sleep_for_openai_recovery(
    delay: float,
    cancellation_token: CancellationToken | None,
) -> None:
    deadline = time.monotonic() + delay
    while True:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def _value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


_REASONING_SIGNAL_KEYS = (
    "reasoning_content",
    "reasoningContent",
    "reasoning",
    "reasoning_details",
    "thinking",
    "thinking_blocks",
    "reasoning_items",
)

_REASONING_CONTAINER_KEYS = (
    "additional_kwargs",
    "model_extra",
    "provider_specific_fields",
)

_REASONING_BLOCK_TYPES = (
    "thinking",
    "thinking_delta",
    "redacted_thinking",
    "redacted_thinking_delta",
    "reasoning",
    "reasoning_text",
)


def _truthy_reasoning_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _has_reasoning_block(value: Any) -> bool:
    if isinstance(value, dict):
        typ = value.get("type")
        if typ in _REASONING_BLOCK_TYPES:
            return True
        for key in _REASONING_SIGNAL_KEYS:
            if key in value and (
                _truthy_reasoning_value(value[key]) or _has_reasoning_block(value[key])
            ):
                return True
        return any(_has_reasoning_block(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_reasoning_block(v) for v in value)
    return False


def _text_from_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("type") in ("text", "output_text"):
            return str(value.get("text") or "")
        return ""
    if isinstance(value, list):
        return "".join(_text_from_content(item) for item in value)
    typ = _value(value, "type")
    if typ in ("text", "output_text"):
        return str(_value(value, "text") or "")
    return ""


def response_output_text(response: Any) -> str:
    """Extract assistant-visible text from a Responses API response object."""
    direct = _value(response, "output_text")
    if isinstance(direct, str):
        return direct
    chunks: list[str] = []
    output = _value(response, "output")
    if isinstance(output, list):
        for item in output:
            if _value(item, "type") == "message":
                chunks.append(_text_from_content(_value(item, "content")))
    return "".join(chunks)


def _has_reasoning_signal(obj: Any) -> bool:
    """Return True when a provider stream object exposes reasoning/thinking data."""
    if obj is None:
        return False
    for key in _REASONING_SIGNAL_KEYS:
        value = _value(obj, key)
        if _truthy_reasoning_value(value) or _has_reasoning_block(value):
            return True
    if _has_reasoning_block(_value(obj, "content")):
        return True
    for container_key in _REASONING_CONTAINER_KEYS:
        extra = _value(obj, container_key)
        if not isinstance(extra, dict):
            continue
        if _has_reasoning_block(extra):
            return True
        for key in _REASONING_SIGNAL_KEYS:
            value = extra.get(key)
            if _truthy_reasoning_value(value) or _has_reasoning_block(value):
                return True
        if _has_reasoning_block(extra.get("content")):
            return True
    return False


def _response_event_has_reasoning_started(event: Any) -> bool:
    typ = str(_value(event, "type") or "")
    if typ in (
        "response.reasoning_text.delta",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_part.added",
    ):
        return True
    if typ == "response.output_item.added":
        return _value(_value(event, "item"), "type") == "reasoning"
    if typ == "response.content_part.added":
        return _value(_value(event, "part"), "type") == "reasoning_text"
    return False


def _response_output_items(response: Any) -> list:
    output = _value(response, "output")
    return output if isinstance(output, list) else []


def _is_response_complete(response: Any) -> bool:
    status = _value(response, "status")
    if status in (None, "completed"):
        return bool(response_output_text(response) or _response_output_items(response))
    return False


def _events_from_recovered_response(
    response: Any,
    yielded_tool_call_ids: set[str],
    yielded_reasoning_ids: set[str],
) -> Iterator:
    for item in _response_output_items(response):
        typ = _value(item, "type")
        if typ == "function_call":
            call_id = str(_value(item, "call_id") or "")
            item_id = str(_value(item, "id") or call_id)
            if item_id in yielded_tool_call_ids or call_id in yielded_tool_call_ids:
                continue
            yielded_tool_call_ids.update({item_id, call_id})
            yield ToolCallDone(
                id=item_id,
                call_id=call_id,
                name=str(_value(item, "name") or ""),
                arguments=str(_value(item, "arguments") or ""),
            )
        elif typ == "reasoning":
            item_id = str(_value(item, "id") or "")
            if item_id in yielded_reasoning_ids:
                continue
            yielded_reasoning_ids.add(item_id)
            yield ReasoningDone(
                id=item_id,
                summary=_value(item, "summary") or [],
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_backend(config: dict) -> str:
    provider_name = config.get("provider", "openai")
    info = PROVIDERS.get(provider_name)
    if info:
        return info["backend"]
    if config.get("base_url"):
        return "openai_compat"
    return "responses"


def create_client(config: dict):
    backend = get_backend(config)
    api_key = resolve_api_key(config)

    if backend in ("responses", "openai_compat"):
        from .openai_http import create_client as create_openai_client

        base_url = config.get("base_url")
        if not base_url:
            provider_name = config.get("provider", "openai")
            info = PROVIDERS.get(provider_name, {})
            base_url = info.get("base_url")
        return create_openai_client(config, api_key, base_url)

    if backend == "anthropic":
        from .anthropic_http import create_client as create_anthropic_client

        return create_anthropic_client(config, api_key)

    if backend == "gemini":
        from .gemini_http import create_client as create_gemini_client

        return create_gemini_client(config, api_key)

    raise ProviderError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Input format conversion (Responses API → Chat Completions messages)
# ---------------------------------------------------------------------------

def _to_chat_messages(instructions: str, input_items: list) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": instructions}]
    i = 0
    while i < len(input_items):
        item = input_items[i]
        role = item.get("role")
        typ = item.get("type")

        if role in ("user", "assistant"):
            messages.append({"role": role, "content": item.get("content", "") or ""})
            i += 1

        elif typ == "reasoning":
            i += 1

        elif typ == "function_call":
            tool_calls = []
            while i < len(input_items) and input_items[i].get("type") == "function_call":
                fc = input_items[i]
                tool_calls.append({
                    "id": fc.get("call_id", fc.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": fc["name"],
                        "arguments": fc.get("arguments", "{}"),
                    },
                })
                i += 1
            messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            while i < len(input_items) and input_items[i].get("type") == "function_call_output":
                fco = input_items[i]
                messages.append({
                    "role": "tool",
                    "tool_call_id": fco["call_id"],
                    "content": to_chat_tool_content(fco.get("output")),
                })
                i += 1

        elif typ == "function_call_output":
            i += 1

        else:
            i += 1

    return messages


# ---------------------------------------------------------------------------
# Tool format conversion (Responses API → Chat Completions)
# ---------------------------------------------------------------------------

def _to_chat_tools(tools: list) -> list:
    """Convert Responses API flat tool format to Chat Completions nested format."""
    result = []
    for tool in strict_openai_tools(tools):
        if tool.get("type") == "function" and "name" in tool and "function" not in tool:
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                    "strict": bool(tool.get("strict", True)),
                },
            })
        else:
            result.append(tool)
    return result


# ---------------------------------------------------------------------------
# Chat Completions tool-call accumulation
# ---------------------------------------------------------------------------

def _flush_tool_calls(accumulators: dict[int, dict]) -> Iterator[ToolCallDone]:
    for idx in sorted(accumulators):
        acc = accumulators[idx]
        call_id = acc["id"] or f"call_{uuid.uuid4().hex[:12]}"
        yield ToolCallDone(
            id=call_id,
            call_id=call_id,
            name=acc["name"],
            arguments=acc["arguments"],
        )
    accumulators.clear()


def _accumulate_tool_delta(accumulators: dict[int, dict], tc_delta) -> None:
    idx = getattr(tc_delta, "index", 0)
    if idx not in accumulators:
        accumulators[idx] = {"id": "", "name": "", "arguments": ""}
    acc = accumulators[idx]
    if getattr(tc_delta, "id", None):
        acc["id"] = tc_delta.id
    fn = getattr(tc_delta, "function", None)
    if fn:
        if getattr(fn, "name", None):
            acc["name"] += fn.name
        if getattr(fn, "arguments", None):
            acc["arguments"] += fn.arguments


# ---------------------------------------------------------------------------
# Backend: OpenAI Responses API
# ---------------------------------------------------------------------------

def _stream_responses_api(
    client, model, instructions, tools, input_items, reasoning=None, prompt_cache_key=None,
    service_tier: str | None = None,
    cancellation_token: CancellationToken | None = None,
) -> Iterator:
    from .openai_http import (
        build_responses_payload,
        retrieve_response,
        stream_response as stream_openai_response,
    )

    payload = build_responses_payload(
        model,
        instructions,
        tools,
        input_items,
        reasoning=reasoning,
        prompt_cache_key=prompt_cache_key,
        service_tier=service_tier,
    )
    reasoning_started = False
    response_id: str | None = None
    started_tool_call_ids: set[str] = set()
    yielded_tool_call_ids: set[str] = set()
    yielded_reasoning_ids: set[str] = set()
    try:
        for event in stream_openai_response(
            client,
            payload,
            cancellation_token=cancellation_token,
        ):
            event_type = str(event.get("type") or "")
            response = event.get("response")
            if isinstance(response, dict) and response.get("id"):
                response_id = str(response["id"])
            elif event.get("response_id"):
                response_id = str(event["response_id"])
            if event_type == "response.created":
                continue
            if not reasoning_started and _response_event_has_reasoning_started(event):
                reasoning_started = True
                item = event.get("item") if isinstance(event.get("item"), dict) else {}
                yield ReasoningStarted(id=str(item.get("id") or ""))
            if event_type == "response.output_text.delta":
                yield TextDelta(str(event.get("delta") or ""))
            elif event_type == "response.output_item.added":
                item = event.get("item") if isinstance(event.get("item"), dict) else {}
                if item.get("type") == "function_call":
                    item_id = str(item.get("id") or "")
                    call_id = str(item.get("call_id") or item_id)
                    started_tool_call_ids.update(value for value in (item_id, call_id) if value)
                    yield ToolCallStarted(
                        id=item_id,
                        call_id=call_id,
                        name=str(item.get("name") or ""),
                    )
            elif event_type == "response.function_call_arguments.delta":
                item_id = str(event.get("item_id") or "")
                if item_id and item_id not in started_tool_call_ids:
                    started_tool_call_ids.add(item_id)
                    yield ToolCallStarted(
                        id=item_id,
                        call_id=item_id,
                        name="",
                    )
            elif event_type == "response.output_item.done":
                item = event.get("item") if isinstance(event.get("item"), dict) else {}
                if item.get("type") == "function_call":
                    item_id = str(item.get("id") or "")
                    call_id = str(item.get("call_id") or "")
                    if not any(
                        value and value in started_tool_call_ids
                        for value in (item_id, call_id)
                    ):
                        yield ToolCallStarted(
                            id=item_id,
                            call_id=call_id,
                            name=str(item.get("name") or ""),
                        )
                    yielded_tool_call_ids.update(
                        {item_id, call_id}
                    )
                    yield ToolCallDone(
                        id=item_id,
                        call_id=call_id,
                        name=str(item.get("name") or ""),
                        arguments=str(item.get("arguments") or "{}"),
                    )
                elif item.get("type") == "reasoning":
                    item_id = str(item.get("id") or "")
                    yielded_reasoning_ids.add(item_id)
                    yield ReasoningDone(
                        id=item_id,
                        summary=item.get("summary") or [],
                    )
            elif event_type == "response.completed":
                yield StreamDone(response=response)
                return
        raise ProviderError("OpenAI response stream ended before response.completed")
    except TurnCancelled:
        raise
    except Exception as stream_error:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        last_recovery_status: str | None = None
        last_retrieval_error: Exception | None = None
        if response_id:
            recovered_response = None
            for attempt in range(_OPENAI_RECOVERY_ATTEMPTS):
                if cancellation_token is not None:
                    cancellation_token.throw_if_cancelled()
                try:
                    candidate = retrieve_response(
                        client,
                        response_id,
                        cancellation_token=cancellation_token,
                    )
                except Exception as retrieval_error:
                    last_retrieval_error = retrieval_error
                    candidate = None
                else:
                    last_retrieval_error = None
                    status = _value(candidate, "status")
                    last_recovery_status = str(status) if status is not None else None
                if candidate is not None and _is_response_complete(candidate):
                    recovered_response = candidate
                    break
                if attempt < _OPENAI_RECOVERY_ATTEMPTS - 1:
                    _sleep_for_openai_recovery(
                        min(0.25 * (2 ** attempt), _OPENAI_RECOVERY_MAX_DELAY),
                        cancellation_token,
                    )
            if recovered_response is not None:
                yield from _events_from_recovered_response(
                    recovered_response,
                    yielded_tool_call_ids,
                    yielded_reasoning_ids,
                )
                yield StreamDone(response=recovered_response)
                return
        recovery_details = []
        if response_id is None:
            recovery_details.append("response id was not observed")
        elif last_retrieval_error is not None:
            recovery_details.append(
                f"last retrieval error: {last_retrieval_error}"
            )
        elif last_recovery_status is not None:
            recovery_details.append(
                f"last recovery status: {last_recovery_status}"
            )
        else:
            recovery_details.append("response retrieval returned no usable result")
        message = (
            f"{stream_error}; recovery failed ({'; '.join(recovery_details)})"
        )
        if isinstance(stream_error, ProviderHTTPError):
            raise ProviderError(message) from stream_error
        raise RetryableStreamError(message) from stream_error


# ---------------------------------------------------------------------------
# Backend: OpenAI-compatible Chat Completions over direct HTTP
# ---------------------------------------------------------------------------

def _stream_chat_completions(
    client, model, instructions, tools, input_items, reasoning=None,
    service_tier: str | None = None,
    cancellation_token: CancellationToken | None = None,
    config: dict | None = None,
) -> Iterator:
    from .openai_http import build_chat_payload, stream_chat

    payload = build_chat_payload(
        model,
        sanitize_json_value(_to_chat_messages(instructions, input_items)),
        sanitize_json_value(_to_chat_tools(tools)) if tools else None,
        reasoning=reasoning,
        service_tier=service_tier,
        provider_name=str((config or {}).get("provider") or ""),
    )
    accumulators: dict[int, dict] = {}
    started_tool_indices: set[int] = set()
    final_chunk: dict[str, Any] = {}
    reasoning_started = False
    for chunk in stream_chat(
        client,
        payload,
        cancellation_token=cancellation_token,
    ):
        if chunk.get("usage"):
            final_chunk["usage"] = chunk["usage"]
        for key in ("id", "model", "created", "service_tier"):
            if key in chunk:
                final_chunk[key] = chunk[key]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        if not reasoning_started and (
            _has_reasoning_signal(delta) or _has_reasoning_signal(choice)
        ):
            reasoning_started = True
            yield ReasoningStarted(id="")
        text_delta = _text_from_content(delta.get("content"))
        if text_delta:
            yield TextDelta(text_delta)
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc_delta in tool_calls:
                if not isinstance(tc_delta, dict):
                    continue
                idx = int(tc_delta.get("index") or 0)
                acc = accumulators.setdefault(
                    idx, {"id": "", "name": "", "arguments": ""}
                )
                if tc_delta.get("id"):
                    acc["id"] = str(tc_delta["id"])
                function = tc_delta.get("function")
                if isinstance(function, dict):
                    acc["name"] += str(function.get("name") or "")
                    acc["arguments"] += str(function.get("arguments") or "")
                if idx not in started_tool_indices:
                    started_tool_indices.add(idx)
                    call_id = acc["id"] or f"chat_tool_{idx}"
                    yield ToolCallStarted(
                        id=call_id,
                        call_id=call_id,
                        name=acc["name"],
                    )
        if choice.get("finish_reason"):
            yield from _flush_tool_calls(accumulators)

    if cancellation_token is not None:
        cancellation_token.throw_if_cancelled()
    yield StreamDone(response=final_chunk)


# ---------------------------------------------------------------------------
# Backend: Anthropic Messages over direct HTTP
# ---------------------------------------------------------------------------

def _stream_anthropic(
    client, config, model, instructions, tools, input_items, reasoning=None,
    max_tokens: int | None = None,
    cancellation_token: CancellationToken | None = None,
) -> Iterator:
    from .anthropic_http import build_payload, stream_message

    payload = build_payload(
        config,
        model,
        instructions,
        tools,
        input_items,
        reasoning=reasoning,
        stream=True,
        max_tokens=max_tokens,
    )
    for event in stream_message(
        client,
        payload,
        cancellation_token=cancellation_token,
        max_retries=int(config.get("anthropic_max_retries", 2)),
    ):
        event_type = event.get("type")
        if event_type == "text_delta":
            yield TextDelta(str(event.get("delta") or ""))
        elif event_type == "reasoning_started":
            yield ReasoningStarted(id=str(event.get("id") or ""))
        elif event_type == "reasoning_done":
            yield ReasoningDone(
                id=str(event.get("id") or ""),
                summary=[],
                provider_content=event.get("provider_content"),
            )
        elif event_type == "tool_call_started":
            call_id = str(event.get("id") or "")
            yield ToolCallStarted(
                id=call_id,
                call_id=call_id,
                name=str(event.get("name") or ""),
            )
        elif event_type == "tool_call":
            call_id = str(event.get("id") or f"call_{uuid.uuid4().hex[:12]}")
            yield ToolCallDone(
                id=call_id,
                call_id=call_id,
                name=str(event.get("name") or ""),
                arguments=str(event.get("arguments") or "{}"),
            )
        elif event_type == "done":
            yield StreamDone(response=event.get("response"))


# ---------------------------------------------------------------------------
# Backend: Gemini over direct HTTP
# ---------------------------------------------------------------------------

def _stream_gemini(
    client, config, model, instructions, tools, input_items, reasoning=None,
    cancellation_token: CancellationToken | None = None,
) -> Iterator:
    from .gemini_http import build_payload, stream_content

    reasoning_parts: list[dict] = []
    for event in stream_content(
        client,
        model,
        build_payload(
            config,
            model,
            instructions,
            tools,
            input_items,
            reasoning=reasoning,
        ),
        cancellation_token=cancellation_token,
        max_retries=int(config.get("gemini_max_retries", 2)),
    ):
        event_type = event.get("type")
        if event_type == "text_delta":
            yield TextDelta(str(event.get("delta") or ""))
        elif event_type == "reasoning_started":
            yield ReasoningStarted(id=str(event.get("id") or ""))
        elif event_type == "reasoning_part":
            content = event.get("provider_content")
            if isinstance(content, list):
                reasoning_parts.extend(content)
        elif event_type == "reasoning_done":
            yield ReasoningDone(
                id=str(event.get("id") or ""),
                summary=[],
                provider_content=reasoning_parts,
            )
        elif event_type == "tool_call":
            call_id = str(event.get("id") or f"call_{uuid.uuid4().hex[:12]}")
            yield ToolCallStarted(
                id=call_id,
                call_id=call_id,
                name=str(event.get("name") or ""),
            )
            yield ToolCallDone(
                id=call_id,
                call_id=call_id,
                name=str(event.get("name") or ""),
                arguments=str(event.get("arguments") or "{}"),
                provider_content=event.get("provider_content"),
            )
        elif event_type == "done":
            yield StreamDone(response=event.get("response"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _stream_response_direct(
    client,
    config: dict,
    model: str,
    instructions: str,
    tools: list,
    input_items: list,
    reasoning: dict | None = None,
    prompt_cache_key: str | None = None,
    max_tokens: int | None = None,
    cancellation_token: CancellationToken | None = None,
) -> Iterator:
    backend = get_backend(config)
    from .provider_catalog import provider_service_tier

    service_tier = provider_service_tier(config)
    try:
        if backend == "responses":
            yield from _stream_responses_api(
                client, model, instructions, tools, input_items, reasoning, prompt_cache_key,
                service_tier, cancellation_token,
            )
        elif backend == "openai_compat":
            yield from _stream_chat_completions(
                client, model, instructions, tools, input_items, reasoning,
                service_tier, cancellation_token, config,
            )
        elif backend == "anthropic":
            yield from _stream_anthropic(
                client, config, model, instructions, tools, input_items, reasoning,
                max_tokens,
                cancellation_token,
            )
        elif backend == "gemini":
            yield from _stream_gemini(
                client, config, model, instructions, tools, input_items, reasoning,
                cancellation_token,
            )
        else:
            raise ProviderError(f"Unknown backend: {backend}")
    except ProviderError:
        raise
    except TurnCancelled:
        raise
    except Exception as e:
        raise ProviderError(str(e)) from e


def stream_response(
    client,
    config: dict,
    model: str,
    instructions: str,
    tools: list,
    input_items: list,
    reasoning: dict | None = None,
    prompt_cache_key: str | None = None,
    max_tokens: int | None = None,
    cancellation_token: CancellationToken | None = None,
) -> Iterator:
    """Stream a normalized response, with interruptible waits on Windows."""
    direct = _stream_response_direct(
        client,
        config,
        model,
        instructions,
        tools,
        input_items,
        reasoning,
        prompt_cache_key,
        max_tokens,
        cancellation_token,
    )
    if sys.platform != "win32" or cancellation_token is None:
        yield from direct
        return

    events: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()

    def produce() -> None:
        try:
            for event in direct:
                events.put(("event", event))
        except BaseException as exc:
            events.put(("error", exc))
        else:
            events.put(("done", None))

    threading.Thread(target=produce, daemon=True, name="jarv-provider-stream").start()
    try:
        while True:
            cancellation_token.throw_if_cancelled()
            try:
                kind, value = events.get(timeout=0.05)
            except queue.Empty:
                continue
            if kind == "event":
                yield value
            elif kind == "error":
                raise value
            else:
                return
    except (KeyboardInterrupt, GeneratorExit):
        cancellation_token.cancel()
        raise
