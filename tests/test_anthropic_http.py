import json

import httpx

from jarv.anthropic_http import (
    AnthropicHTTPError,
    build_payload,
    create_message,
    stream_message,
)
from jarv.provider import (
    ReasoningDone,
    ReasoningStarted,
    StreamDone,
    TextDelta,
    ToolCallDone,
    _stream_anthropic,
    get_backend,
)


def _sse(events):
    return "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n"
        for name, data in events
    ).encode()


def _client(handler):
    return httpx.Client(
        base_url="https://api.anthropic.test",
        transport=httpx.MockTransport(handler),
    )


def test_anthropic_uses_native_backend():
    assert get_backend({"provider": "anthropic"}) == "anthropic"


def test_payload_preserves_signed_thinking_and_tool_history():
    thinking = {
        "type": "thinking",
        "thinking": "private reasoning",
        "signature": "signed-value",
    }
    payload = build_payload(
        {},
        "claude-opus-4-7",
        "system",
        [{
            "type": "function",
            "name": "run_command",
            "description": "Run a command",
            "parameters": {"type": "object"},
        }],
        [
            {"role": "user", "content": "inspect"},
            {
                "type": "reasoning",
                "id": "thinking_0",
                "summary": [],
                "provider_content": [thinking],
            },
            {
                "type": "function_call",
                "id": "toolu_1",
                "call_id": "toolu_1",
                "name": "run_command",
                "arguments": '{"command":"git status"}',
            },
            {
                "type": "function_call_output",
                "call_id": "toolu_1",
                "output": "clean",
            },
        ],
        reasoning={"effort": "high"},
        stream=True,
    )

    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "high"}
    assert payload["messages"][1]["role"] == "assistant"
    assert payload["messages"][1]["content"][0] == thinking
    assert payload["messages"][1]["content"][1] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "run_command",
        "input": {"command": "git status"},
    }
    assert payload["messages"][2]["content"][0]["type"] == "tool_result"
    assert payload["tools"][0]["input_schema"] == {"type": "object"}


def test_legacy_model_maps_reasoning_effort_to_budget():
    payload = build_payload(
        {},
        "claude-haiku-4-5",
        "",
        [],
        [{"role": "user", "content": "hi"}],
        reasoning={"effort": "medium"},
    )

    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "output_config" not in payload


def test_stream_emits_thinking_text_tool_and_normalized_usage():
    events = [
        ("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [],
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 3,
                    "cache_read_input_tokens": 7,
                    "output_tokens": 0,
                },
            },
        }),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "reason"},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Running"},
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("content_block_start", {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "run_command",
                "input": {},
            },
        }),
        ("content_block_delta", {
            "type": "content_block_delta",
            "index": 2,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"command":"git status"}',
            },
        }),
        ("content_block_stop", {"type": "content_block_stop", "index": 2}),
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 12},
        }),
        ("message_stop", {"type": "message_stop"}),
    ]

    def handler(_request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(events),
        )

    client = _client(handler)
    try:
        normalized = list(_stream_anthropic(
            client,
            {},
            "claude-opus-4-7",
            "system",
            [],
            [{"role": "user", "content": "hi"}],
        ))
    finally:
        client.close()

    assert isinstance(normalized[0], ReasoningStarted)
    assert isinstance(normalized[1], ReasoningDone)
    assert normalized[1].provider_content == [{
        "type": "thinking",
        "thinking": "reason",
        "signature": "sig",
    }]
    assert isinstance(normalized[2], TextDelta)
    assert normalized[2].delta == "Running"
    assert isinstance(normalized[3], ToolCallDone)
    assert normalized[3].arguments == '{"command":"git status"}'
    assert isinstance(normalized[4], StreamDone)
    assert normalized[4].response["output_text"] == "Running"
    assert normalized[4].response["usage"]["input_tokens"] == 20
    assert normalized[4].response["usage"]["cached_input_tokens"] == 7
    assert normalized[4].response["usage"]["uncached_input_tokens"] == 13
    assert normalized[4].response["usage"]["total_tokens"] == 32


def test_non_streaming_request_retries_retryable_status(monkeypatch):
    calls = []

    def handler(_request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                529,
                json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )

    monkeypatch.setattr("jarv.anthropic_http._sleep", lambda _delay, _token: None)
    client = _client(handler)
    try:
        response = create_message(
            client,
            {"model": "claude", "max_tokens": 1, "messages": []},
            max_retries=1,
        )
    finally:
        client.close()

    assert len(calls) == 2
    assert response["output_text"] == "ok"


def test_stream_error_includes_anthropic_error_details():
    def handler(_request):
        return httpx.Response(
            401,
            headers={"request-id": "req_123"},
            json={
                "type": "error",
                "error": {"type": "authentication_error", "message": "bad key"},
            },
        )

    client = _client(handler)
    try:
        try:
            list(stream_message(
                client,
                {"model": "claude", "max_tokens": 1, "messages": [], "stream": True},
                max_retries=0,
            ))
        except AnthropicHTTPError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected AnthropicHTTPError")
    finally:
        client.close()

    assert "401" in message
    assert "authentication_error" in message
    assert "req_123" in message
