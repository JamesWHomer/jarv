import json

import httpx
import pytest

from jarv.gemini_http import (
    build_payload as build_gemini_payload,
    generate_content,
    normalize_response as normalize_gemini_response,
    stream_content,
    to_contents,
)
from jarv.openai_http import (
    build_chat_payload,
    build_responses_payload,
    stream_chat,
    stream_response,
)


def _sse_response(events):
    body = "".join(
        f"event: {event.get('type', 'message')}\n"
        f"data: {json.dumps(event)}\n\n"
        for event in events
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body,
    )


def test_openai_responses_sends_cache_key_and_parses_completed_event():
    captured = {}

    def handler(request):
        captured["payload"] = json.loads(request.content)
        return _sse_response([
            {"type": "response.created", "response": {"id": "resp_1"}},
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 30,
                        "input_tokens_details": {"cached_tokens": 20},
                        "output_tokens": 4,
                    },
                    "output": [],
                },
            },
        ])

    client = httpx.Client(
        base_url="https://api.openai.test/v1",
        transport=httpx.MockTransport(handler),
    )
    payload = build_responses_payload(
        "model",
        "system",
        [],
        [{"role": "user", "content": "hi"}],
        prompt_cache_key="jarv:session",
    )
    events = list(stream_response(client, payload))
    assert captured["payload"]["prompt_cache_key"] == "jarv:session"
    assert captured["payload"]["store"] is True
    assert events[-1]["response"]["usage"]["input_tokens_details"]["cached_tokens"] == 20


def test_openai_compatible_chat_parses_data_only_sse():
    def handler(_request):
        body = (
            'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body)

    client = httpx.Client(
        base_url="https://provider.test/v1",
        transport=httpx.MockTransport(handler),
    )
    chunks = list(stream_chat(client, {"model": "m", "messages": [], "stream": True}))
    assert chunks[0]["choices"][0]["delta"]["content"] == "hi"
    assert chunks[1]["usage"]["prompt_tokens"] == 2


def test_gemini_history_replays_thought_signature_on_function_call():
    contents = to_contents([
        {"role": "user", "content": "run it"},
        {
            "type": "function_call",
            "id": "call_1",
            "call_id": "call_1",
            "name": "run",
            "arguments": "{}",
            "provider_content": [{
                "functionCall": {"name": "run", "args": {}},
                "thoughtSignature": "signed",
            }],
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "done",
        },
    ])
    assert contents[1]["parts"][0]["thoughtSignature"] == "signed"
    assert contents[2]["parts"][0]["functionResponse"]["name"] == "run"


def test_gemini_stream_and_usage_preserve_cached_and_thought_tokens():
    chunk = {
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"text": "thinking", "thought": True},
                    {"text": "answer"},
                ],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 20,
            "cachedContentTokenCount": 8,
            "candidatesTokenCount": 3,
            "thoughtsTokenCount": 2,
            "totalTokenCount": 25,
        },
    }

    def handler(request):
        assert request.url.params["alt"] == "sse"
        return _sse_response([chunk])

    client = httpx.Client(
        base_url="https://generativelanguage.test/v1beta",
        transport=httpx.MockTransport(handler),
    )
    events = list(stream_content(
        client,
        "gemini-test",
        build_gemini_payload(
            {},
            "gemini-test",
            "system",
            [],
            [{"role": "user", "content": "hi"}],
            reasoning={"effort": "high"},
        ),
    ))
    assert events[0]["type"] == "reasoning_started"
    assert any(event.get("delta") == "answer" for event in events)
    usage = events[-1]["response"]["usage"]
    assert usage["cached_input_tokens"] == 8
    assert usage["reasoning_output_tokens"] == 2


def test_gemini_normalization_excludes_thought_text_from_output():
    response = normalize_gemini_response({
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "hidden", "thought": True},
                    {"text": "visible"},
                ]
            }
        }]
    })
    assert response["output_text"] == "visible"


def test_gemini_non_streaming_response_preserves_served_tier_header():
    def handler(_request):
        return httpx.Response(
            200,
            headers={"x-gemini-service-tier": "priority"},
            json={
                "candidates": [{"content": {"parts": [{"text": "visible"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 1,
                },
            },
        )

    client = httpx.Client(
        base_url="https://generativelanguage.test/v1beta",
        transport=httpx.MockTransport(handler),
    )
    try:
        response = generate_content(client, "gemini-test", {})
    finally:
        client.close()

    assert response["service_tier"] == "priority"


def test_gemini_pro_rejects_unsupported_effort_and_preserves_medium():
    with pytest.raises(ValueError, match="low, medium, high"):
        build_gemini_payload(
            {}, "gemini-3.1-pro-preview", "", [], [],
            reasoning={"effort": "minimal"},
        )
    medium = build_gemini_payload(
        {}, "gemini-3.1-pro-preview", "", [], [],
        reasoning={"effort": "medium"},
    )
    assert medium["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "medium"


def test_gemini_requests_thought_summaries_at_default_effort():
    payload = build_gemini_payload(
        {}, "gemini-3.1-pro-preview", "", [], [],
    )

    assert payload["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": True,
    }


def test_gemini_does_not_request_thoughts_for_unknown_models():
    payload = build_gemini_payload(
        {}, "gemini-test", "", [], [],
    )

    assert "generationConfig" not in payload


def test_gemini_25_uses_current_reasoning_budgets_and_explicit_none():
    medium = build_gemini_payload(
        {}, "gemini-2.5-flash", "", [], [],
        reasoning={"effort": "medium"},
    )
    disabled = build_gemini_payload(
        {}, "gemini-2.5-flash", "", [], [],
        reasoning={"effort": "none"},
    )

    assert medium["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 8192
    assert disabled["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": False,
        "thinkingBudget": 0,
    }


def test_openrouter_uses_standard_reasoning_object_and_strict_routing():
    payload = build_chat_payload(
        "openai/gpt-5.4-mini",
        [{"role": "user", "content": "hi"}],
        reasoning={"effort": "high"},
        provider_name="openrouter",
    )

    assert payload["reasoning"] == {"effort": "high"}
    assert payload["provider"] == {"require_parameters": True}
    assert "reasoning_effort" not in payload
