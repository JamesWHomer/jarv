"""Assistant responses must survive tool execution and session reloads."""

from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from jarv.agent import run_agent
from jarv.artifacts import ArtifactStore
from jarv.cancellation import TurnCancelled
from jarv.config import DEFAULT_CONFIG
from jarv.context_budget import build_input
from jarv.history import SessionContext, load_history
from jarv.orchestrator import AgentNode, run_subagent_loop
from jarv.provider import ReasoningDone, StreamDone, TextDelta, ToolCallDone


def tool_response(text_source, call_count):
    yield ReasoningDone("rs_1", [], [{"type": "thinking", "signature": "signed"}])
    if text_source == "stream":
        yield TextDelta("Checking ")
        yield TextDelta("the files.")
    elif text_source == "final":
        yield TextDelta("Checking ")
    for index in range(call_count):
        yield ToolCallDone(
            id=f"fc_{index}", call_id=f"call_{index}", name="run_command",
            arguments='{"command": "echo ok"}',
        )
    yield StreamDone(
        {"output_text": "Checking the files."} if text_source in {"final", "final_only"} else None
    )


@pytest.mark.parametrize("text_source", ["stream", "final", "final_only", "none"])
@pytest.mark.parametrize("call_count", [1, 2])
def test_tool_response_survives_next_request_and_saved_history(tmp_path, text_source, call_count):
    context = SessionContext("test", "test", tmp_path / "history.json", datetime.now(timezone.utc))
    requests = []

    def stream(*args, **_kwargs):
        requests.append(deepcopy(args[5]))
        if len(requests) == 1:
            yield from tool_response(text_source, call_count)
        else:
            yield TextDelta("Done.")
            yield StreamDone(None)

    with (
        patch("jarv.agent.prepare_session_context", return_value=context),
        patch("jarv.agent.stream_response", side_effect=stream),
        patch("jarv.agent._dispatch_run_command_with_ui", return_value="ok"),
        patch("jarv.agent.record_response_usage"),
    ):
        result = run_agent("Check the files", DEFAULT_CONFIG, client=object())

    assert result.error is None
    assert len(requests) == 2
    saved = load_history(context.history_file)
    reloaded = build_input(saved, model=DEFAULT_CONFIG["model"], config=DEFAULT_CONFIG)
    expected_text = [] if text_source == "none" else ["Checking the files."]
    for items, final_text in [(requests[1], []), (reloaded, ["Done."])]:
        assert [item["content"] for item in items if item.get("role") == "assistant"] == expected_text + final_text
        assert [item.get("type", item.get("role")) for item in items] == (
            ["user", "reasoning"] + ["assistant"] * len(expected_text)
            + ["function_call"] * call_count + ["function_call_output"] * call_count
            + ["assistant"] * len(final_text)
        )
        assert items[1]["provider_content"][0]["signature"] == "signed"
        assert [item["call_id"] for item in items if item.get("type") == "function_call_output"] == [
            f"call_{index}" for index in range(call_count)
        ]
        assert [item["output"] for item in items if item.get("type") == "function_call_output"] == ["ok"] * call_count
        assert all("session_id" not in item for item in items)
    assert all(item["session_id"] == "test" for item in saved)


def test_subagent_keeps_text_accompanying_tool_calls():
    requests = []

    def stream(*args, **_kwargs):
        requests.append(deepcopy(args[5]))
        if len(requests) == 1:
            yield from tool_response("stream", 2)
        else:
            yield ToolCallDone("fc_finish", "call_finish", "finish", '{"longform": "done", "tldr": "done"}')
            yield StreamDone(None)

    node = AgentNode("child", 1, "root", "Check the files", True, incognito=True)
    with (
        patch("jarv.orchestrator.stream_response", side_effect=stream),
        patch("jarv.orchestrator.dispatch_tool", return_value="ok"),
    ):
        result = run_subagent_loop(node, ArtifactStore(), client=object(), config=DEFAULT_CONFIG)

    assert result == ("done", "done")
    assert len(requests) == 2
    assert [item["content"] for item in requests[1] if item.get("role") == "assistant"] == ["Checking the files."]
    assert [item.get("type", item.get("role")) for item in requests[1]] == [
        "user", "reasoning", "assistant", "function_call", "function_call",
        "function_call_output", "function_call_output",
    ]


@pytest.mark.parametrize("error", [TurnCancelled(), RuntimeError("tool failed")])
def test_interrupted_tools_keep_one_complete_response_and_all_results(tmp_path, error):
    context = SessionContext("test", "test", tmp_path / "history.json", datetime.now(timezone.utc))
    with (
        patch("jarv.agent.prepare_session_context", return_value=context),
        patch("jarv.agent.stream_response", return_value=tool_response("stream", 3)),
        patch("jarv.agent._dispatch_run_command_with_ui", side_effect=["ok", error]),
        patch("jarv.agent.record_response_usage"),
    ):
        result = run_agent("Check the files", DEFAULT_CONFIG, client=object())

    assert result.cancelled if isinstance(error, TurnCancelled) else result.error == "tool failed"
    saved = load_history(context.history_file)
    assert [item["content"] for item in saved if item.get("content") == "Checking the files."] == ["Checking the files."]
    calls = [item for item in saved if item.get("type") == "function_call"]
    outputs = [item for item in saved if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in calls] == ["call_0", "call_1", "call_2"]
    assert [item["call_id"] for item in outputs] == ["call_0", "call_1", "call_2"]
    assert outputs[0]["output"] == "ok"
    assert "may have made partial changes" in outputs[1]["output"]
    assert "before execution" in outputs[2]["output"]
    assert saved.index(calls[-1]) < saved.index(outputs[0])
