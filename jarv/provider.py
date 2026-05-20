"""Multi-provider abstraction layer.

Supports two streaming backends:
- OpenAI Responses API (for OpenAI models — superior tool calling)
- Chat Completions API (for all other providers via OpenAI SDK or litellm)
"""

import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import Any, Iterator

from .provider_catalog import KEY_PATTERNS, LOCAL_PROVIDERS, PROVIDERS
from .unicode_safety import sanitize_json_value


# ---------------------------------------------------------------------------
# Normalized stream events
# ---------------------------------------------------------------------------

@dataclass
class TextDelta:
    delta: str


@dataclass
class ToolCallDone:
    id: str
    call_id: str
    name: str
    arguments: str


@dataclass
class ReasoningDone:
    id: str
    summary: list


@dataclass
class StreamDone:
    response: Any


class ProviderError(Exception):
    pass


def responses_input_id(item_id: str, prefix: str) -> str:
    """Return an id that is valid for Responses API input items."""
    valid_prefix = f"{prefix}_"
    if item_id.startswith(valid_prefix) and len(item_id) <= 64:
        return item_id
    digest_len = 64 - len(valid_prefix)
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:digest_len]
    return f"{valid_prefix}{digest}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_api_key(config: dict) -> str:
    provider_name = config.get("provider", "openai")
    # Per-provider key takes priority
    per_provider = config.get("api_keys", {}).get(provider_name, "")
    if per_provider:
        return per_provider
    # Legacy flat key
    key = config.get("api_key", "")
    if key:
        return key
    # Environment variable
    info = PROVIDERS.get(provider_name, {})
    env_key = info.get("env_key")
    if env_key:
        return os.environ.get(env_key, "")
    if provider_name in LOCAL_PROVIDERS:
        return "not-needed"
    return ""


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
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key or "not-needed"}
        base_url = config.get("base_url")
        if not base_url:
            provider_name = config.get("provider", "openai")
            info = PROVIDERS.get(provider_name, {})
            base_url = info.get("base_url")
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    # litellm doesn't use a persistent client
    return None


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
                    "content": str(fco.get("output", "")),
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
    for tool in tools:
        if tool.get("type") == "function" and "name" in tool and "function" not in tool:
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
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
    client, model, instructions, tools, input_items, reasoning=None,
) -> Iterator:
    kwargs: dict[str, Any] = dict(
        model=model,
        instructions=instructions,
        tools=tools,
        input=input_items,
    )
    if reasoning:
        kwargs["reasoning"] = reasoning
    kwargs = sanitize_json_value(kwargs)

    with client.responses.stream(**kwargs) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                yield TextDelta(event.delta)
            elif event.type == "response.output_item.done":
                if event.item.type == "function_call":
                    yield ToolCallDone(
                        id=event.item.id,
                        call_id=event.item.call_id,
                        name=event.item.name,
                        arguments=event.item.arguments,
                    )
                elif event.item.type == "reasoning":
                    yield ReasoningDone(
                        id=event.item.id,
                        summary=getattr(event.item, "summary", []),
                    )
        try:
            final_response = stream.get_final_response()
        except Exception:
            final_response = None
        yield StreamDone(response=final_response)


# ---------------------------------------------------------------------------
# Backend: Chat Completions via OpenAI SDK (OpenAI-compatible providers)
# ---------------------------------------------------------------------------

def _stream_chat_completions(
    client, model, instructions, tools, input_items, reasoning=None,
) -> Iterator:
    messages = sanitize_json_value(_to_chat_messages(instructions, input_items))

    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    if tools:
        kwargs["tools"] = sanitize_json_value(_to_chat_tools(tools))
    if reasoning and reasoning.get("effort"):
        kwargs["reasoning_effort"] = reasoning["effort"]

    accumulators: dict[int, dict] = {}
    final_chunk = None

    stream = client.chat.completions.create(**sanitize_json_value(kwargs))
    try:
        for chunk in stream:
            if getattr(chunk, "usage", None):
                final_chunk = chunk
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield TextDelta(delta.content)
            if delta and getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    _accumulate_tool_delta(accumulators, tc_delta)

            if chunk.choices[0].finish_reason:
                yield from _flush_tool_calls(accumulators)
    finally:
        if hasattr(stream, "close"):
            stream.close()

    yield StreamDone(response=final_chunk)


# ---------------------------------------------------------------------------
# Backend: litellm (Anthropic, Gemini, Ollama)
# ---------------------------------------------------------------------------

def _stream_litellm(
    config, model, instructions, tools, input_items, reasoning=None,
) -> Iterator:
    from .litellm_compat import import_litellm

    litellm = import_litellm()

    messages = sanitize_json_value(_to_chat_messages(instructions, input_items))
    provider_name = config.get("provider", "")

    litellm_model = model
    prefix = PROVIDERS.get(provider_name, {}).get("litellm_prefix")
    if prefix and "/" not in model:
        litellm_model = f"{prefix}/{model}"

    kwargs: dict[str, Any] = dict(
        model=litellm_model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    if tools:
        kwargs["tools"] = sanitize_json_value(_to_chat_tools(tools))

    api_key = resolve_api_key(config)
    if api_key and api_key != "not-needed":
        kwargs["api_key"] = api_key

    accumulators: dict[int, dict] = {}
    final_chunk = None

    for chunk in litellm.completion(**sanitize_json_value(kwargs)):
        if getattr(chunk, "usage", None):
            final_chunk = chunk
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            yield TextDelta(delta.content)
        if getattr(delta, "tool_calls", None):
            for tc_delta in delta.tool_calls:
                _accumulate_tool_delta(accumulators, tc_delta)

        if chunk.choices[0].finish_reason:
            yield from _flush_tool_calls(accumulators)

    yield StreamDone(response=final_chunk)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stream_response(
    client,
    config: dict,
    model: str,
    instructions: str,
    tools: list,
    input_items: list,
    reasoning: dict | None = None,
) -> Iterator:
    """Stream a response using the configured provider.

    Yields TextDelta, ToolCallDone, ReasoningDone, and StreamDone events.
    """
    backend = get_backend(config)
    try:
        if backend == "responses":
            yield from _stream_responses_api(
                client, model, instructions, tools, input_items, reasoning,
            )
        elif backend == "openai_compat":
            yield from _stream_chat_completions(
                client, model, instructions, tools, input_items, reasoning,
            )
        elif backend == "litellm":
            yield from _stream_litellm(
                config, model, instructions, tools, input_items, reasoning,
            )
        else:
            raise ProviderError(f"Unknown backend: {backend}")
    except ProviderError:
        raise
    except Exception as e:
        raise ProviderError(str(e)) from e
