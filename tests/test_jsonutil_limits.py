"""JSON recovery rejects payloads the decoder cannot inspect safely."""

import json
import sys

import pytest

from jarv.jsonutil import iter_json_objects, salvage_json_object


@pytest.fixture(params=["nesting", "integer"])
def excessive_json(request):
    if request.param == "integer":
        get_limit = getattr(sys, "get_int_max_str_digits", lambda: 0)
        limit = get_limit()
        if not limit:
            pytest.skip("interpreter does not limit integer string conversion")
        return '{"value":' + "9" * (limit + 1) + "}"
    depth = sys.getrecursionlimit() + 100
    payload = '{"value":' + "[" * depth + "0" + "]" * depth + "}"
    try:
        json.loads(payload)
    except RecursionError:
        return payload
    pytest.skip("interpreter's JSON decoder does not use the Python recursion limit")


@pytest.mark.parametrize("prefix,suffix", [
    ("", ""), ("```json\n", "\n```"), ("Here: ", ""),
    ('{"ok":true} trailing ', ""),
])
def test_salvage_rejects_decoder_limits(excessive_json, prefix, suffix):
    assert salvage_json_object(prefix + excessive_json + suffix) is None


def test_verdict_scan_stops_on_decoder_limits(excessive_json):
    assert list(iter_json_objects(excessive_json + ' {"allow":true}')) == []


def test_anthropic_stream_preserves_tool_arguments_exceeding_decoder_limits(excessive_json):
    import httpx

    from jarv.anthropic_http import stream_message, to_messages
    from jarv.orchestrator import _parse_tool_args

    events = [
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": "call", "name": "read", "input": {},
        }},
        {"type": "content_block_delta", "index": 0, "delta": {
            "type": "input_json_delta", "partial_json": excessive_json,
        }},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    with httpx.Client(
        base_url="https://provider.test",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)),
    ) as client:
        received = list(stream_message(client, {}))

    call = next(event for event in received if event["type"] == "tool_call")
    assert call["arguments"] == excessive_json
    assert received[-1]["type"] == "done"
    args, error = _parse_tool_args(call["arguments"])
    assert args is None
    assert error.startswith("[tool argument error:")
    # A rejected call still enters history; converting the next request must work.
    history = [{**call, "type": "function_call", "call_id": "call"}]
    assert to_messages(history)[0]["content"][0]["input"] == {}


def test_history_arguments_use_existing_fallback_for_decoder_limits(excessive_json):
    from jarv.history_convert import parse_json_arguments

    fallback = {"unavailable": True}
    assert parse_json_arguments(excessive_json) == {}
    assert parse_json_arguments(excessive_json, fallback=fallback) is fallback


def test_gemini_history_preserves_text_exceeding_decoder_limits(excessive_json):
    from jarv.gemini_http import to_contents

    contents = to_contents([
        {"type": "function_call", "call_id": "call", "name": "read",
         "arguments": excessive_json},
        {"type": "function_call_output", "call_id": "call", "output": excessive_json},
    ])

    assert contents[0]["parts"][0]["functionCall"]["args"] == {"result": excessive_json}
    assert contents[1]["parts"][0]["functionResponse"]["response"] == {
        "result": {"result": excessive_json},
    }
