"""Multi-provider abstraction layer.

Supports two streaming backends:
- OpenAI Responses API (for OpenAI models — superior tool calling)
- Chat Completions API (for all other providers via OpenAI SDK or litellm)
"""

import os
import uuid
from dataclasses import dataclass
from typing import Any, Iterator


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


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openai": {
        "backend": "responses",
        "base_url": None,
        "env_key": "OPENAI_API_KEY",
        "key_url": "https://platform.openai.com/api-keys",
        "label": "OpenAI",
    },
    "openrouter": {
        "backend": "openai_compat",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "key_url": "https://openrouter.ai/keys",
        "label": "OpenRouter",
    },
    "groq": {
        "backend": "openai_compat",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
        "label": "Groq",
    },
    "deepseek": {
        "backend": "openai_compat",
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "key_url": "https://platform.deepseek.com/api_keys",
        "label": "DeepSeek",
    },
    "together": {
        "backend": "openai_compat",
        "base_url": "https://api.together.ai/v1",
        "env_key": "TOGETHER_API_KEY",
        "key_url": "https://api.together.ai/settings/api-keys",
        "label": "Together AI",
    },
    "fireworks": {
        "backend": "openai_compat",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_key": "FIREWORKS_API_KEY",
        "key_url": "https://fireworks.ai/account/api-keys",
        "label": "Fireworks AI",
    },
    "anthropic": {
        "backend": "litellm",
        "base_url": None,
        "env_key": "ANTHROPIC_API_KEY",
        "key_url": "https://console.anthropic.com/settings/keys",
        "label": "Anthropic",
        "litellm_prefix": "anthropic",
    },
    "gemini": {
        "backend": "litellm",
        "base_url": None,
        "env_key": "GEMINI_API_KEY",
        "key_url": "https://aistudio.google.com/apikey",
        "label": "Google Gemini",
        "litellm_prefix": "gemini",
    },
    "ollama": {
        "backend": "litellm",
        "base_url": None,
        "env_key": None,
        "key_url": None,
        "label": "Ollama (local)",
        "litellm_prefix": "ollama",
    },
    "lm_studio": {
        "backend": "openai_compat",
        "base_url": "http://localhost:1234/v1",
        "env_key": None,
        "key_url": None,
        "label": "LM Studio (local)",
    },
    "vllm": {
        "backend": "openai_compat",
        "base_url": "http://localhost:8000/v1",
        "env_key": None,
        "key_url": None,
        "label": "vLLM (local)",
    },
}

LOCAL_PROVIDERS = {"ollama", "lm_studio", "vllm"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_api_key(config: dict) -> str:
    key = config.get("api_key", "")
    if key:
        return key
    provider_name = config.get("provider", "openai")
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
    messages = _to_chat_messages(instructions, input_items)

    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        stream=True,
    )
    if tools:
        kwargs["tools"] = _to_chat_tools(tools)
    if reasoning and reasoning.get("effort"):
        kwargs["reasoning_effort"] = reasoning["effort"]

    accumulators: dict[int, dict] = {}
    final_chunk = None

    stream = client.chat.completions.create(**kwargs)
    try:
        for chunk in stream:
            if not chunk.choices:
                if getattr(chunk, "usage", None):
                    final_chunk = chunk
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
    import litellm

    messages = _to_chat_messages(instructions, input_items)
    provider_name = config.get("provider", "")

    litellm_model = model
    prefix = PROVIDERS.get(provider_name, {}).get("litellm_prefix")
    if prefix and "/" not in model:
        litellm_model = f"{prefix}/{model}"

    kwargs: dict[str, Any] = dict(
        model=litellm_model,
        messages=messages,
        stream=True,
    )
    if tools:
        kwargs["tools"] = _to_chat_tools(tools)

    api_key = resolve_api_key(config)
    if api_key and api_key != "not-needed":
        kwargs["api_key"] = api_key

    accumulators: dict[int, dict] = {}
    final_chunk = None

    for chunk in litellm.completion(**kwargs):
        if not chunk.choices:
            if getattr(chunk, "usage", None):
                final_chunk = chunk
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
