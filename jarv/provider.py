"""Multi-provider abstraction layer over direct HTTP transports."""

import queue
import sys
import threading
import uuid
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterator

from .history_convert import iter_history_segments, parse_json_arguments, provider_metadata
from .provider_catalog import KEY_PATTERNS, LOCAL_PROVIDERS, PROVIDERS
from .provider_auth import resolve_api_key
from .provider_registry import ProviderError, create_client, get_backend
from .response_items import responses_input_id
from .tool_schemas import strict_openai_tools
from .tool_outputs import to_chat_tool_content
from .unicode_safety import sanitize_json_value
from .cancellation import CancellationToken, TurnCancelled
from .http_transport import ProviderHTTPError, RETRYABLE_STATUS_CODES
from .http_transport import _sleep as _sleep_for_openai_recovery


# ---------------------------------------------------------------------------
# Normalized stream events
# ---------------------------------------------------------------------------

@dataclass
class TextDelta:
    delta: str
    received_at: float | None = field(default=None, compare=False)


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
    provider_metadata: dict | None = None


@dataclass
class ReasoningDone:
    id: str
    summary: list
    provider_content: list[dict] | None = None
    provider_metadata: dict | None = None


@dataclass
class ReasoningStarted:
    id: str


@dataclass
class StreamDone:
    response: Any
    provider_metadata: dict | None = None


class RetryableStreamError(ProviderError):
    """A provider stream failed before its response could be recovered."""


_OPENAI_RECOVERY_ATTEMPTS = 16
_OPENAI_RECOVERY_MAX_DELAY = 2.0


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


def provider_response_notice(config: dict, response: Any) -> str | None:
    """Provider-reported condition the user should see (e.g. a safety fallback)."""
    if get_backend(config) != "anthropic":
        return None
    from .anthropic_http import fallback_notice

    return fallback_notice(response)


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
                provider_metadata={"provider": "openai"},
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_chat_messages(
    instructions: str, input_items: list, *, provider_name: str = "",
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": instructions}]
    assistant: dict | None = None

    def flush_assistant():
        nonlocal assistant
        if assistant is not None:
            messages.append(assistant)
            assistant = None

    def copy_reasoning(item):
        metadata = provider_metadata(item) or {}
        if (provider_name == "deepseek" and metadata.get("provider") == provider_name
                and isinstance(metadata.get("reasoning_content"), str)):
            assistant["reasoning_content"] = metadata["reasoning_content"]

    for segment in iter_history_segments(input_items):
        kind = segment[0]
        if kind == "message":
            _, role, item = segment
            flush_assistant()
            message = {"role": role, "content": item.get("content", "") or ""}
            if role == "assistant":
                assistant = message
                copy_reasoning(item)
            else:
                messages.append(message)
        elif kind == "function_calls":
            calls = segment[1]
            if assistant is None:
                assistant = {"role": "assistant", "content": None}
            tool_calls = []
            for fc in calls:
                copy_reasoning(fc)
                tool_calls.append({
                    "id": fc.get("call_id", fc.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": fc["name"],
                        "arguments": fc.get("arguments", "{}"),
                    },
                })
            assistant.setdefault("tool_calls", []).extend(tool_calls)
        elif kind == "function_outputs":
            flush_assistant()
            for fco in segment[1]:
                messages.append({
                    "role": "tool",
                    "tool_call_id": fco["call_id"],
                    "content": to_chat_tool_content(fco.get("output")),
                })
    flush_assistant()
    return messages


def requires_reasoning_history(config: dict, tools: list) -> bool:
    """DeepSeek thinking with tools requires all preceding assistant reasoning."""
    return (config.get("provider") == "deepseek" and bool(tools)
            and str(config.get("reasoning_effort") or "").strip().lower() != "none")


def validate_history_compatibility(config: dict, tools: list, input_items: list) -> None:
    """Reject unrecoverable reasoning history before any request or history write."""
    if not requires_reasoning_history(config, tools):
        return
    messages = _to_chat_messages("", input_items, provider_name="deepseek")
    if any(message.get("role") == "assistant"
           and not isinstance(message.get("reasoning_content"), str) for message in messages):
        raise ProviderError(
            "This chat lacks the reasoning history required by DeepSeek. "
            "Start a fresh chat with /new or --new."
        )


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

def _recover_openai_stream(
    client,
    response_id: str | None,
    stream_error: Exception,
    *,
    yielded_tool_call_ids: set[str],
    yielded_reasoning_ids: set[str],
    cancellation_token: CancellationToken | None,
) -> Iterator:
    """Poll ``retrieve_response`` after a dropped Responses stream.

    Yields the not-yet-seen tail of the recovered response followed by
    ``StreamDone``, or raises ``ProviderError``/``RetryableStreamError`` with
    the recovery details appended to ``stream_error``.
    """
    # Call-time import: tests patch jarv.openai_http.retrieve_response.
    from .openai_http import retrieve_response

    last_recovery_status: str | None = None
    last_retrieval_error: Exception | None = None
    permanent_failure = False
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
            except TurnCancelled:
                raise
            except Exception as retrieval_error:
                last_retrieval_error = retrieval_error
                candidate = None
                if (
                    isinstance(retrieval_error, ProviderHTTPError)
                    and retrieval_error.status_code is not None
                    and 400 <= retrieval_error.status_code < 500
                    and retrieval_error.status_code not in RETRYABLE_STATUS_CODES
                ):
                    permanent_failure = True
                    break
            else:
                last_retrieval_error = None
                status = _value(candidate, "status")
                last_recovery_status = str(status) if status is not None else None
            if candidate is not None and _is_response_complete(candidate):
                recovered_response = candidate
                break
            if last_recovery_status in {"failed", "incomplete", "cancelled", "completed"}:
                permanent_failure = True
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
    if permanent_failure or isinstance(stream_error, ProviderHTTPError):
        raise ProviderError(message) from stream_error
    raise RetryableStreamError(message) from stream_error


def _stream_responses_api(
    client, model, instructions, tools, input_items, reasoning=None, prompt_cache_key=None,
    service_tier: str | None = None,
    cancellation_token: CancellationToken | None = None,
) -> Iterator:
    from .openai_http import (
        build_responses_payload,
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
                        provider_metadata={"provider": "openai"},
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
        yield from _recover_openai_stream(
            client,
            response_id,
            stream_error,
            yielded_tool_call_ids=yielded_tool_call_ids,
            yielded_reasoning_ids=yielded_reasoning_ids,
            cancellation_token=cancellation_token,
        )


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

    provider_name = str((config or {}).get("provider") or "")
    validate_history_compatibility(
        {"provider": provider_name, "reasoning_effort": (reasoning or {}).get("effort")},
        tools, input_items,
    )
    messages = _to_chat_messages(instructions, input_items, provider_name=provider_name)
    payload = build_chat_payload(
        model,
        sanitize_json_value(messages),
        sanitize_json_value(_to_chat_tools(tools)) if tools else None,
        reasoning=reasoning,
        service_tier=service_tier,
        provider_name=provider_name,
    )
    accumulators: dict[int, dict] = {}
    started_tool_indices: set[int] = set()
    final_chunk: dict[str, Any] = {}
    reasoning_started = False
    reasoning_content: list[str] = []
    reasoning_content_seen = False
    finished = False
    for chunk in stream_chat(
        client,
        payload,
        cancellation_token=cancellation_token,
    ):
        if chunk.get("usage"):
            final_chunk["usage"] = chunk["usage"]
        for key in ("x_groq", "timings", "stats"):
            value = chunk.get(key)
            if isinstance(value, dict):
                final_chunk.setdefault(key, {}).update(value)
        for key in ("id", "model", "created", "service_tier", "eval_count", "eval_duration"):
            if key in chunk:
                final_chunk[key] = chunk[key]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        if provider_name == "deepseek" and isinstance(delta.get("reasoning_content"), str):
            reasoning_content_seen = True
            reasoning_content.append(delta["reasoning_content"])
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
            finished = True
            final_chunk["finish_reason"] = choice["finish_reason"]
            yield from _flush_tool_calls(accumulators)

    if cancellation_token is not None:
        cancellation_token.throw_if_cancelled()
    if not finished:
        raise RetryableStreamError("Chat Completions stream ended before finish_reason")
    groq_usage = final_chunk.get("x_groq", {}).get("usage")
    if isinstance(groq_usage, dict) and not final_chunk.get("usage"):
        # Groq can report streaming usage under x_groq. Keep each report's
        # counts and timings together instead of manufacturing a mixed pair.
        final_chunk["usage"] = dict(groq_usage)
    yield StreamDone(
        response=final_chunk,
        provider_metadata=(
            {"provider": "deepseek",
             **({"reasoning_content": "".join(reasoning_content)} if reasoning_content_seen else {})}
            if provider_name == "deepseek" else None
        ),
    )


# ---------------------------------------------------------------------------
# Backend: Anthropic / Gemini intermediate event mapping
# ---------------------------------------------------------------------------

def _map_dict_stream_events(
    events: Iterator,
    *,
    combine_tool_start_done: bool = False,
    provider_name: str = "",
) -> Iterator:
    """Map Anthropic/Gemini HTTP dict events to normalized provider dataclasses."""
    reasoning_parts: list[dict] = []
    for event in events:
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
                provider_content=(
                    reasoning_parts if combine_tool_start_done else event.get("provider_content")
                ),
                provider_metadata={"provider": provider_name} if provider_name else None,
            )
            if combine_tool_start_done:
                reasoning_parts = []
        elif event_type == "tool_call_started":
            call_id = str(event.get("id") or "")
            yield ToolCallStarted(
                id=call_id,
                call_id=call_id,
                name=str(event.get("name") or ""),
            )
        elif event_type == "tool_call":
            call_id = str(event.get("id") or f"call_{uuid.uuid4().hex[:12]}")
            if combine_tool_start_done:
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
                provider_metadata={"provider": provider_name} if provider_name else None,
            )
        elif event_type == "done":
            yield StreamDone(response=event.get("response"))


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
        yield from _map_dict_stream_events([event], provider_name="anthropic")


# ---------------------------------------------------------------------------
# Backend: Gemini over direct HTTP
# ---------------------------------------------------------------------------

def _stream_gemini(
    client, config, model, instructions, tools, input_items, reasoning=None,
    cancellation_token: CancellationToken | None = None,
) -> Iterator:
    from .gemini_http import build_payload, stream_content

    yield from _map_dict_stream_events(
        stream_content(
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
        ),
        combine_tool_start_done=True,
        provider_name="gemini",
    )


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

    try:
        service_tier = provider_service_tier(config, model=model, backend=backend)
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

    def timestamped_events() -> Iterator:
        try:
            for event in direct:
                if isinstance(event, TextDelta) and event.delta and event.received_at is None:
                    # Timestamp ingress before Windows queues or rendering delay it.
                    event.received_at = perf_counter()
                yield event
        finally:
            # Preserve yield-from's prompt close propagation to HTTP resources.
            close = getattr(direct, "close", None)
            if callable(close):
                close()

    if sys.platform != "win32" or cancellation_token is None:
        yield from timestamped_events()
        return

    events: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()

    def produce() -> None:
        try:
            for event in timestamped_events():
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
