"""Regression coverage for buffered provider stream collection."""

import json
from copy import deepcopy

import httpx
import pytest

from jarv.anthropic_http import stream_message
from jarv.cancellation import TurnCancelled
from jarv.gemini_http import stream_content
from jarv.provider import RetryableStreamError, StreamDone, TextDelta
from jarv.tool_schemas import strict_openai_tools
from jarv.turn_loop import collect_stream_response


def _client(events):
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Client(
        base_url="https://provider.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=body)
        ),
    )


def _start(index, **block):
    return {"type": "content_block_start", "index": index, "content_block": block}


def _delta(index, delta_type, **values):
    return {
        "type": "content_block_delta", "index": index,
        "delta": {"type": delta_type, **values},
    }


def _stop(index):
    return {"type": "content_block_stop", "index": index}


def test_anthropic_interleaved_blocks_keep_initial_content_and_event_order():
    events = [
        {"type": "message_start", "message": {"id": "message", "usage": {"input_tokens": 3}}},
        _start(0, type="text", text="prefix:"),
        _start(1, type="thinking", thinking="initial", signature="sig"),
        _start(2, type="tool_use", id="tool", name="read", input={}),
        _delta(0, "text_delta", text="one"),
        _delta(1, "thinking_delta", thinking=" thought"),
        _delta(2, "input_json_delta", partial_json='{"input":'),
        _delta(0, "text_delta", text=""),
        _delta(1, "signature_delta", signature="nature"),
        _delta(0, "text_delta", text="two"),
        _delta(2, "input_json_delta", partial_json='"file.txt"}'),
        _stop(1), _stop(2), _stop(0),
        {"type": "message_delta", "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]
    original = deepcopy(events)
    with _client(events) as client:
        actual = list(stream_message(client, {}))

    thinking = {"type": "thinking", "thinking": "initial thought", "signature": "signature"}
    assert actual[:-1] == [
        {"type": "reasoning_started", "id": "thinking_1"},
        {"type": "tool_call_started", "id": "tool", "name": "read"},
        {"type": "text_delta", "delta": "one"},
        {"type": "text_delta", "delta": "two"},
        {"type": "reasoning_done", "id": "thinking_1", "provider_content": [thinking]},
        {"type": "tool_call", "id": "tool", "name": "read", "arguments": '{"input":"file.txt"}'},
    ]
    response = actual[-1]["response"]
    assert response["content"] == [
        thinking,
        {"type": "tool_use", "id": "tool", "name": "read", "input": {"input": "file.txt"}},
        {"type": "text", "text": "prefix:onetwo"},
    ]
    assert response["output_text"] == "prefix:onetwo"
    assert response["usage"]["total_tokens"] == 8
    assert events == original


def test_anthropic_restarted_block_discards_unfinished_fragments():
    events = [
        _start(0, type="text", text="old"),
        _delta(0, "text_delta", text=" discarded"),
        _start(0, type="text", text=42),
        _delta(0, "text_delta", text=""),
        _delta(0, [], text="unknown delta"),
        _delta(0, "text_delta", text=" answer"),
        _stop(0), {"type": "message_stop"},
    ]
    with _client(events) as client:
        actual = list(stream_message(client, {}))
    assert [event["delta"] for event in actual if event["type"] == "text_delta"] == [
        " discarded", " answer",
    ]
    assert actual[-1]["response"]["content"] == [{"type": "text", "text": "42 answer"}]


def test_anthropic_fragmented_invalid_tool_json_is_returned_for_salvage():
    events = [
        _start(0, type="tool_use", id="tool", name="read", input={}),
        _delta(0, "input_json_delta", partial_json='{"input":'),
        _delta(0, "input_json_delta", partial_json='"truncated'),
        _stop(0), {"type": "message_stop"},
    ]
    with _client(events) as client:
        actual = list(stream_message(client, {}))
    assert actual[1]["arguments"] == '{"input":"truncated'
    assert actual[-1]["response"]["content"][0] == {
        "type": "tool_use", "id": "tool", "name": "read", "input": {},
    }


def test_gemini_final_candidates_are_sorted_without_changing_delta_order():
    events = [
        {"candidates": [
            {"index": 2, "content": {"role": "model", "parts": [{"text": "two"}]}},
            {"index": 0, "content": {"parts": [{"text": "zero"}]}},
        ]},
        {"candidates": [
            {"index": 1, "finishReason": "STOP", "content": {"parts": [{"text": "one"}]}},
            {"index": 0, "content": {"role": "model", "parts": [{"text": " more"}]}},
        ], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4}},
    ]
    with _client(events) as client:
        actual = list(stream_content(client, "model", {}))
    assert actual[:-1] == [
        {"type": "text_delta", "delta": "two"},
        {"type": "text_delta", "delta": "one"},
    ]
    response = actual[-1]["response"]
    assert [candidate["index"] for candidate in response["candidates"]] == [0, 1, 2]
    assert response["candidates"][0]["content"] == {
        "role": "model", "parts": [{"text": "zero"}, {"text": " more"}],
    }
    assert response["output_text"] == "zero more"
    assert response["usage"]["total_tokens"] == 7


@pytest.mark.parametrize("error", [TurnCancelled, ValueError, RetryableStreamError])
def test_unobserved_stream_keeps_partial_text_on_failure(error):
    ended = []

    def make_stream():
        yield TextDelta("first")
        yield TextDelta("")
        yield TextDelta(" second")
        raise error("interrupted")

    with pytest.raises(error, match="interrupted"):
        collect_stream_response(
            make_stream, max_replays=0,
            on_attempt_end=lambda result, retry: ended.append((result, retry)),
        )
    result, retry = ended[0]
    assert not retry
    assert result.reply_text == result.streamed_text == "first second"
    assert result.got_text
    assert result.text_chunk_count == 2


def test_stream_retry_finalizer_sees_each_attempt_and_recovered_final_text():
    ended = []
    attempts = 0

    def make_stream():
        nonlocal attempts
        attempts += 1
        yield TextDelta("first")
        yield TextDelta(" second")
        if attempts == 1:
            raise RetryableStreamError("retry")
        yield StreamDone({"output_text": "first second recovered"})

    result = collect_stream_response(
        make_stream,
        on_attempt_end=lambda result, retry: ended.append(
            (result.reply_text, result.streamed_text, retry)
        ),
    )
    assert ended == [
        ("first second", "first second", True),
        ("first second recovered", "first second", False),
    ]
    assert result.reply_text == "first second recovered"


def test_observed_stream_preserves_callback_snapshot_and_mutations():
    observed = []

    def on_event(event, result):
        observed.append((result.reply_text, result.streamed_text))
        if isinstance(event, TextDelta) and event.delta == "first":
            result.reply_text = "prefix:"

    result = collect_stream_response(
        lambda: iter([TextDelta("first"), TextDelta(" second"), StreamDone({})]),
        on_event=on_event,
    )
    assert observed == [
        ("", "first"),
        ("prefix:first", "first second"),
        ("prefix:first second", "first second"),
    ]
    assert result.reply_text == "prefix:first second"


def test_strict_tools_keep_shared_required_and_optional_schemas_independent():
    child = {"type": "string", "enum": ["one", "two"]}
    shared = {
        "type": "object",
        "properties": {"required": child, "optional": child},
        "required": ["required"],
    }
    tool = {
        "type": "function", "name": "inspect",
        "parameters": {
            "type": "object", "properties": {"required": shared, "optional": shared},
            "required": ["required"],
        },
    }
    original = deepcopy(tool)
    converted = strict_openai_tools([tool])[0]
    properties = converted["parameters"]["properties"]
    assert properties["required"]["type"] == "object"
    assert properties["optional"]["type"] == ["object", "null"]
    for nested in properties.values():
        assert nested["properties"]["required"]["type"] == "string"
        assert nested["properties"]["optional"]["type"] == ["string", "null"]
        assert nested["required"] == ["required", "optional"]
        assert nested["additionalProperties"] is False
    properties["optional"]["properties"]["required"]["enum"].append("three")
    assert properties["required"]["properties"]["required"]["enum"] == ["one", "two"]
    assert tool == original
