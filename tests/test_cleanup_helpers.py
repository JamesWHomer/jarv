import pytest
from rich.text import Text

from jarv.provider import ReasoningDone, RetryableStreamError, StreamDone, TextDelta, ToolCallDone
from jarv.response_items import (
    function_call_history_item,
    function_call_output_item,
    reasoning_history_item,
    responses_input_id,
    to_response_input_item,
)
from jarv.tui_layout import append_bottom_footer, clip_text
from jarv.turn_loop import collect_stream_response


def test_response_item_builders_match_responses_input_shape():
    tool = ToolCallDone(
        id="raw-tool-id",
        call_id="call_1",
        name="run_command",
        arguments='{"command":"pwd"}',
        provider_content=[{"provider": "value"}],
    )
    metadata = {"session_id": "session-id"}

    function_call = function_call_history_item(tool, metadata)
    output = function_call_output_item(tool.call_id, "ok", metadata)

    api_call = to_response_input_item(function_call)
    api_output = to_response_input_item(output)

    assert function_call["session_id"] == "session-id"
    assert function_call["provider_content"] == [{"provider": "value"}]
    assert api_call == {
        "type": "function_call",
        "id": responses_input_id("raw-tool-id", "fc"),
        "call_id": "call_1",
        "name": "run_command",
        "arguments": '{"command":"pwd"}',
        "provider_content": [{"provider": "value"}],
    }
    assert api_output == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "ok",
    }


def test_reasoning_history_item_preserves_provider_content():
    item = ReasoningDone(
        id="reasoning-id",
        summary=["summary"],
        provider_content=[{"type": "thinking"}],
    )

    stored = reasoning_history_item(item, {"turn": 1})

    assert stored == {
        "type": "reasoning",
        "id": "reasoning-id",
        "summary": ["summary"],
        "turn": 1,
        "provider_content": [{"type": "thinking"}],
    }
    assert to_response_input_item(stored)["id"] == responses_input_id(
        "reasoning-id",
        "rs",
    )


def test_collect_stream_response_replays_once_and_uses_final_text():
    attempts = []

    def make_stream():
        attempts.append(1)
        if len(attempts) == 1:
            yield TextDelta("partial")
            raise RetryableStreamError("try again")
        yield TextDelta("visible")
        yield ToolCallDone(id="fc_1", call_id="call_1", name="finish", arguments="{}")
        yield StreamDone(response={"output_text": "visible final"})

    retries = []
    result = collect_stream_response(make_stream, on_retry=lambda: retries.append(True))

    assert len(attempts) == 2
    assert retries == [True]
    assert result.reply_text == "visible final"
    assert [tool.name for tool in result.tool_calls] == ["finish"]


def test_collect_stream_response_raises_after_second_retryable_error():
    def make_stream():
        raise RetryableStreamError("still broken")

    with pytest.raises(RetryableStreamError):
        collect_stream_response(make_stream)


def test_tui_layout_helpers_clip_and_pin_footer():
    assert clip_text("abcdef", 4) == "a..."
    assert clip_text("abcdef", 2) == "ab"

    parts = [Text("row")]
    append_bottom_footer(parts, 6, Text("footer"))

    assert [getattr(part, "plain", part) for part in parts] == [
        "row",
        "",
        "",
        "footer",
    ]
