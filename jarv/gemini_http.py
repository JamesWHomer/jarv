"""Direct HTTP transport and history conversion for the Gemini API."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

from .cancellation import CancellationToken
from .history_convert import (
    append_grouped,
    convert_tools,
    iter_history_segments,
    parse_json_arguments,
)
from .http_transport import (
    ProviderHTTPError,
    create_client as create_http_client,
    iter_sse_json,
    open_stream_response,
    request_json,
    request_json_response,
    response_error,
    send_with_retries,
)
from .tool_outputs import to_gemini_function_response_parts
from .unicode_safety import sanitize_json_value


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
_THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
    "xhigh": 24576,
    "max": 24576,
}


def create_client(config: dict, api_key: str):
    return create_http_client(
        config.get("base_url") or GEMINI_API_URL,
        {
            "x-goog-api-key": api_key,
            "content-type": "application/json",
            "user-agent": "jarv",
        },
        timeout=float(config.get("gemini_timeout", 600)),
        connect_timeout=float(config.get("gemini_connect_timeout", 10)),
    )


def list_models(client, *, max_retries: int = 0) -> dict:
    models: list[dict] = []
    page_token: str | None = None
    while True:
        params = {"pageSize": 1000}
        if page_token:
            params["pageToken"] = page_token
        page = request_json(
            "Gemini",
            client,
            "GET",
            "/models",
            params=params,
            max_retries=max_retries,
        )
        data = page.get("models")
        if isinstance(data, list):
            models.extend(item for item in data if isinstance(item, dict))
        next_token = page.get("nextPageToken")
        if not isinstance(next_token, str) or not next_token or next_token == page_token:
            break
        page_token = next_token
    return {"models": models}


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value if value is not None else {}
    try:
        return json.loads(value or "{}")
    except json.JSONDecodeError:
        return {"result": value}


def _to_gemini_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    converted = {
        key: value
        for key, value in schema.items()
        if key not in {"additionalProperties"}
    }
    if isinstance(converted.get("properties"), dict):
        converted["properties"] = {
            key: _to_gemini_schema(value)
            for key, value in converted["properties"].items()
        }
    if "items" in converted:
        converted["items"] = _to_gemini_schema(converted["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(converted.get(key), list):
            converted[key] = [_to_gemini_schema(item) for item in converted[key]]
    typ = converted.get("type")
    if isinstance(typ, list):
        non_null = [item for item in typ if item != "null"]
        if len(non_null) == 1:
            converted["type"] = non_null[0]
            if len(non_null) != len(typ):
                converted["nullable"] = True
    return converted


def to_contents(input_items: list[dict]) -> list[dict]:
    contents: list[dict] = []
    call_names: dict[str, str] = {}
    call_ids: dict[str, str] = {}
    for segment in iter_history_segments(input_items):
        kind = segment[0]
        if kind == "message":
            _, role, item = segment
            append_grouped(
                contents,
                "model" if role == "assistant" else "user",
                [{"text": str(item.get("content") or "")}],
                content_key="parts",
            )
            continue
        if kind == "reasoning":
            item = segment[1]
            provider_content = item.get("provider_content")
            if isinstance(provider_content, list):
                append_grouped(
                    contents,
                    "model",
                    [dict(part) for part in provider_content if isinstance(part, dict)],
                    content_key="parts",
                )
            continue
        if kind == "function_calls":
            calls = segment[1]
            parts = []
            for call in calls:
                call_id = str(call.get("call_id") or call.get("id") or "")
                name = str(call.get("name") or "")
                call_names[call_id] = name
                provider_content = call.get("provider_content")
                if isinstance(provider_content, list) and provider_content:
                    for part in provider_content:
                        if not isinstance(part, dict):
                            continue
                        function_call = part.get("functionCall")
                        if isinstance(function_call, dict) and function_call.get("id"):
                            call_ids[call_id] = str(function_call["id"])
                    parts.extend(
                        dict(part) for part in provider_content if isinstance(part, dict)
                    )
                else:
                    if call_id:
                        call_ids[call_id] = call_id
                    parts.append({
                        "functionCall": {
                            "name": name,
                            "args": _json_value(call.get("arguments")),
                            **({"id": call_id} if call_id else {}),
                        }
                    })
            append_grouped(contents, "model", parts, content_key="parts")
            continue
        if kind == "function_outputs":
            outputs = segment[1]
            parts = []
            for result in outputs:
                output = result.get("output")
                if isinstance(output, list):
                    response_body, response_parts = to_gemini_function_response_parts(output)
                else:
                    response_body, response_parts = {"result": _json_value(output)}, []
                call_id = str(result.get("call_id") or "")
                response = {
                    "name": call_names.get(call_id, call_id),
                    "response": response_body,
                }
                if call_ids.get(call_id):
                    response["id"] = call_ids[call_id]
                if response_parts:
                    response["parts"] = response_parts
                parts.append({"functionResponse": response})
            append_grouped(contents, "user", parts, content_key="parts")
    return contents


def to_tools(tools: list[dict]) -> list[dict]:
    declarations = convert_tools(
        tools,
        convert_one=lambda _tool, function: {
            "name": function["name"],
            "description": function.get("description", ""),
            "parameters": _to_gemini_schema(
                function.get("parameters", {"type": "object"})
            ),
        },
    )
    return [{"functionDeclarations": declarations}] if declarations else []


def _apply_reasoning(config: dict, payload: dict, model: str, reasoning: dict | None) -> None:
    from .reasoning import get_reasoning_capabilities, require_reasoning_effort

    effort = str((reasoning or {}).get("effort") or "").lower()
    probe = dict(config)
    probe["provider"] = "gemini"
    probe["model"] = model
    if not effort:
        if get_reasoning_capabilities(probe).supported is True:
            payload.setdefault("generationConfig", {})["thinkingConfig"] = {
                "includeThoughts": True,
            }
        return
    effort = require_reasoning_effort(probe, effort)
    if effort == "none":
        payload.setdefault("generationConfig", {})["thinkingConfig"] = {
            "includeThoughts": False,
            "thinkingBudget": 0,
        }
        return

    thinking: dict[str, Any] = {"includeThoughts": True}
    if model.lower().startswith("gemini-3"):
        thinking["thinkingLevel"] = effort
    else:
        thinking["thinkingBudget"] = _THINKING_BUDGETS.get(effort, -1)
    payload.setdefault("generationConfig", {})["thinkingConfig"] = thinking


def build_payload(
    config: dict,
    model: str,
    instructions: str,
    tools: list[dict],
    input_items: list[dict],
    *,
    reasoning: dict | None = None,
    max_output_tokens: int | None = None,
) -> dict:
    payload: dict[str, Any] = {"contents": to_contents(input_items)}
    if instructions:
        payload["systemInstruction"] = {"parts": [{"text": instructions}]}
    converted_tools = to_tools(tools)
    if converted_tools:
        payload["tools"] = converted_tools
    if max_output_tokens is not None:
        payload.setdefault("generationConfig", {})["maxOutputTokens"] = max_output_tokens
    from .provider_catalog import provider_service_tier

    service_tier = provider_service_tier(config, "gemini")
    if service_tier:
        payload["service_tier"] = service_tier
    _apply_reasoning(config, payload, model, reasoning)
    return sanitize_json_value(payload)


def _path(model: str, method: str) -> str:
    return f"/models/{quote(model, safe='')}{method}"


def normalize_response(data: dict) -> dict:
    usage = data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else {}
    input_tokens = int(usage.get("promptTokenCount") or 0)
    cached = int(usage.get("cachedContentTokenCount") or 0)
    output_tokens = int(usage.get("candidatesTokenCount") or 0)
    reasoning_tokens = int(usage.get("thoughtsTokenCount") or 0)
    parts = []
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates:
        content = candidates[0].get("content")
        if isinstance(content, dict) and isinstance(content.get("parts"), list):
            parts = content["parts"]
    return {
        **data,
        "output_text": "".join(
            str(part.get("text") or "")
            for part in parts
            if isinstance(part, dict) and not part.get("thought")
        ),
        "usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached,
            "uncached_input_tokens": max(input_tokens - cached, 0),
            "output_tokens": output_tokens,
            "reasoning_output_tokens": reasoning_tokens,
            "total_tokens": int(
                usage.get("totalTokenCount")
                or input_tokens + output_tokens + reasoning_tokens
            ),
        },
    }


def generate_content(
    client,
    model: str,
    payload: dict,
    *,
    cancellation_token: CancellationToken | None = None,
    max_retries: int = 2,
) -> dict:
    response = send_with_retries(
        client,
        "POST",
        _path(model, ":generateContent"),
        json_body=payload,
        cancellation_token=cancellation_token,
        max_retries=max_retries,
    )
    try:
        if response.status_code >= 400:
            raise response_error("Gemini", response)
        data = response.json()
        if not isinstance(data, dict):
            raise ProviderHTTPError("Gemini", "response was not a JSON object")
        served_tier = response.headers.get("x-gemini-service-tier")
        if served_tier:
            data["service_tier"] = served_tier
        return normalize_response(data)
    finally:
        response.close()


def stream_content(
    client,
    model: str,
    payload: dict,
    *,
    cancellation_token: CancellationToken | None = None,
    max_retries: int = 2,
) -> Iterator[dict]:
    response, unregister = open_stream_response(
        client,
        "POST",
        _path(model, ":streamGenerateContent"),
        provider="Gemini",
        json_body=payload,
        params={"alt": "sse"},
        cancellation_token=cancellation_token,
        max_retries=max_retries,
    )
    final: dict[str, Any] = {"candidates": [], "usageMetadata": {}}
    served_tier = response.headers.get("x-gemini-service-tier")
    if served_tier:
        final["service_tier"] = served_tier
    reasoning_started = False
    try:
        for _event_name, chunk in iter_sse_json("Gemini", response):
            if cancellation_token is not None:
                cancellation_token.throw_if_cancelled()
            if isinstance(chunk.get("error"), dict):
                error = chunk["error"]
                raise ProviderHTTPError(
                    "Gemini",
                    str(error.get("message") or "stream failed"),
                    status_code=error.get("code"),
                    error_type=str(error.get("status") or "") or None,
                )
            if isinstance(chunk.get("usageMetadata"), dict):
                final["usageMetadata"] = chunk["usageMetadata"]
            candidates = chunk.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                continue
            final["candidates"] = candidates
            content = candidates[0].get("content")
            parts = content.get("parts") if isinstance(content, dict) else []
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("thought"):
                    if not reasoning_started:
                        reasoning_started = True
                        yield {"type": "reasoning_started", "id": "gemini-thinking"}
                    yield {
                        "type": "reasoning_part",
                        "provider_content": [dict(part)],
                    }
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    yield {"type": "text_delta", "delta": text}
                function_call = part.get("functionCall")
                if isinstance(function_call, dict):
                    call_id = str(
                        function_call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                    )
                    yield {
                        "type": "tool_call",
                        "id": call_id,
                        "name": str(function_call.get("name") or ""),
                        "arguments": json.dumps(
                            function_call.get("args") or {},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "provider_content": [dict(part)],
                    }
        if reasoning_started:
            yield {
                "type": "reasoning_done",
                "id": "gemini-thinking",
                "provider_content": [],
            }
        yield {"type": "done", "response": normalize_response(final)}
    finally:
        unregister()
        response.close()
