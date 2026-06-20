"""Direct HTTP integrations for OpenAI Responses and compatible chat APIs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .cancellation import CancellationToken
from .http_transport import (
    ProviderHTTPError,
    create_client as create_http_client,
    iter_sse_json,
    open_stream_response,
    request_json,
    response_error,
    send_with_retries,
)
from .tool_schemas import strict_openai_tools
from .unicode_safety import sanitize_json_value


OPENAI_API_URL = "https://api.openai.com/v1"


def create_client(config: dict, api_key: str, base_url: str | None = None):
    return create_http_client(
        base_url or config.get("base_url") or OPENAI_API_URL,
        {
            "authorization": f"Bearer {api_key or 'not-needed'}",
            "content-type": "application/json",
            "user-agent": "jarv",
        },
        timeout=float(config.get("http_timeout", 600)),
        connect_timeout=float(config.get("http_connect_timeout", 10)),
    )


def list_models(client, *, max_retries: int = 0) -> dict:
    return request_json("OpenAI-compatible", client, "GET", "/models", max_retries=max_retries)


def build_responses_payload(
    model: str,
    instructions: str,
    tools: list,
    input_items: list,
    *,
    reasoning: dict | None = None,
    prompt_cache_key: str | None = None,
    service_tier: str | None = None,
    stream: bool = True,
) -> dict:
    clean_input = []
    for item in input_items:
        if isinstance(item, dict):
            item = {
                key: value
                for key, value in item.items()
                if key != "provider_content"
            }
        clean_input.append(item)
    payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "tools": strict_openai_tools(tools),
        "input": clean_input,
        "store": True,
        "stream": stream,
    }
    if reasoning:
        payload["reasoning"] = reasoning
    if prompt_cache_key:
        payload["prompt_cache_key"] = prompt_cache_key
    if service_tier:
        payload["service_tier"] = service_tier
    return sanitize_json_value(payload)


def retrieve_response(
    client,
    response_id: str,
    *,
    cancellation_token: CancellationToken | None = None,
) -> dict:
    return request_json(
        "OpenAI",
        client,
        "GET",
        f"/responses/{response_id}",
        cancellation_token=cancellation_token,
        max_retries=0,
    )


def stream_response(
    client,
    payload: dict,
    *,
    cancellation_token: CancellationToken | None = None,
    max_retries: int = 2,
) -> Iterator[dict]:
    response, unregister = open_stream_response(
        client,
        "POST",
        "/responses",
        provider="OpenAI",
        json_body=payload,
        cancellation_token=cancellation_token,
        max_retries=max_retries,
    )
    try:
        for event_name, event in iter_sse_json("OpenAI", response):
            if cancellation_token is not None:
                cancellation_token.throw_if_cancelled()
            event_type = str(event.get("type") or event_name)
            if event_type == "error":
                error = event.get("error") if isinstance(event.get("error"), dict) else event
                raise ProviderHTTPError(
                    "OpenAI",
                    str(error.get("message") or "stream failed"),
                    error_type=str(error.get("type") or error.get("code") or "") or None,
                )
            if event_type in ("response.failed", "response.incomplete"):
                failed = event.get("response") if isinstance(event.get("response"), dict) else {}
                error = failed.get("error") if isinstance(failed.get("error"), dict) else {}
                raise ProviderHTTPError(
                    "OpenAI",
                    str(error.get("message") or f"response ended with {event_type}"),
                    error_type=str(error.get("code") or "") or None,
                )
            yield event
    finally:
        unregister()
        response.close()


def build_chat_payload(
    model: str,
    messages: list,
    tools: list | None = None,
    *,
    reasoning: dict | None = None,
    stream: bool = True,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
    service_tier: str | None = None,
    provider_name: str | None = None,
) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if tools:
        payload["tools"] = tools
    if reasoning and reasoning.get("effort"):
        if provider_name == "openrouter":
            payload["reasoning"] = {"effort": reasoning["effort"]}
            payload["provider"] = {"require_parameters": True}
        else:
            payload["reasoning_effort"] = reasoning["effort"]
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if service_tier:
        payload["service_tier"] = service_tier
    return sanitize_json_value(payload)


def create_chat(
    client,
    payload: dict,
    *,
    cancellation_token: CancellationToken | None = None,
    max_retries: int = 2,
) -> dict:
    return request_json(
        "OpenAI-compatible",
        client,
        "POST",
        "/chat/completions",
        json_body=payload,
        cancellation_token=cancellation_token,
        max_retries=max_retries,
    )


def stream_chat(
    client,
    payload: dict,
    *,
    cancellation_token: CancellationToken | None = None,
    max_retries: int = 2,
) -> Iterator[dict]:
    response, unregister = open_stream_response(
        client,
        "POST",
        "/chat/completions",
        provider="OpenAI-compatible",
        json_body=payload,
        cancellation_token=cancellation_token,
        max_retries=max_retries,
    )
    try:
        for _event_name, chunk in iter_sse_json("OpenAI-compatible", response):
            if cancellation_token is not None:
                cancellation_token.throw_if_cancelled()
            if isinstance(chunk.get("error"), dict):
                error = chunk["error"]
                raise ProviderHTTPError(
                    "OpenAI-compatible",
                    str(error.get("message") or "stream failed"),
                    error_type=str(error.get("type") or error.get("code") or "") or None,
                )
            yield chunk
    finally:
        unregister()
        response.close()
