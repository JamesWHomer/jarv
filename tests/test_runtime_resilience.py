"""Regression coverage for stream completion, tool ownership and private turns."""

import io
import json
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from rich.console import Console

from jarv.agent import run_agent
from jarv.artifacts import ArtifactStore
from jarv.cancellation import TurnCancelled
from jarv.config import DEFAULT_CONFIG
from jarv.gemini_http import stream_content
from jarv.headsup import HeadsupApp
from jarv.history import SessionContext, ephemeral_session_context, load_history
from jarv.orchestrator import (
    AgentNode, PendingRunCommand, RunCommandDispatchResult, ToolExecutionHooks,
    dispatch_tool, execute_tool_calls, spawn_batch,
)
from jarv.provider import (
    ProviderError, RetryableStreamError, StreamDone, TextDelta, ToolCallDone,
    _stream_chat_completions,
)


def _call(index, name="run_command"):
    return ToolCallDone(
        id=f"fc_{index}", call_id=f"call_{index}", name=name,
        arguments=json.dumps({"command": f"command {index}"}),
    )


@pytest.mark.parametrize("delta", [None, {"content": "partial"}, {
    "tool_calls": [{"index": 0, "id": "call_1", "function": {
        "name": "run_command", "arguments": '{"command":"unfinished',
    }}],
}])
def test_chat_eof_requires_terminal_finish_reason(delta):
    chunks = [] if delta is None else [{"choices": [{"delta": delta}]}]
    with patch("jarv.openai_http.stream_chat", return_value=iter(chunks)):
        events = _stream_chat_completions(object(), "model", "system", [], [])
        seen = []
        with pytest.raises(RetryableStreamError, match="finish_reason"):
            for event in events:
                seen.append(event)
    assert not any(isinstance(event, (StreamDone, ToolCallDone)) for event in seen)


@pytest.mark.parametrize("finish", ["stop", "length", "content_filter"])
def test_chat_explicit_finish_and_trailing_usage_are_terminal(finish):
    chunks = [
        {"choices": [{"delta": {"content": "answer"}, "finish_reason": finish}]},
        {"choices": [], "usage": {"completion_tokens": 1}},
    ]
    with patch("jarv.openai_http.stream_chat", return_value=iter(chunks)):
        events = list(_stream_chat_completions(object(), "model", "", [], []))
    assert isinstance(events[-1], StreamDone)
    assert events[-1].response["finish_reason"] == finish
    assert events[-1].response["usage"]["completion_tokens"] == 1


def _gemini_events(chunks):
    response = httpx.Response(200, headers={"content-type": "text/event-stream"},
        content="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks))
    client = httpx.Client(base_url="https://gemini.test", transport=httpx.MockTransport(
        lambda _request: response,
    ))
    return client, response, stream_content(client, "model", {})


@pytest.mark.parametrize("parts", [None, [{"text": "unfinished"}], [{
    "functionCall": {"name": "run_command", "args": {"command": "pending"}},
}]])
def test_gemini_eof_requires_terminal_finish_reason(parts):
    chunks = [] if parts is None else [{"candidates": [{"content": {"parts": parts}}]}]
    client, response, events = _gemini_events(chunks)
    with client, pytest.raises(RetryableStreamError, match="finishReason"):
        list(events)
    assert response.is_closed


@pytest.mark.parametrize("terminal", [
    {"candidates": [{"finishReason": "STOP"}]},
    {"candidates": [{"finishReason": "MAX_TOKENS"}]},
    {"candidates": [{"finishReason": "SAFETY"}]},
    {"promptFeedback": {"blockReason": "SAFETY"}},
])
def test_gemini_explicit_finish_without_content_is_terminal(terminal):
    client, response, events = _gemini_events([terminal, {"usageMetadata": {"promptTokenCount": 3}}])
    with client:
        result = list(events)
    assert result[-1]["type"] == "done"
    assert result[-1]["response"]["usage"]["input_tokens"] == 3
    assert response.is_closed


@pytest.mark.parametrize("fail_during_trim", [False, True])
def test_pending_process_is_cleaned_up_on_provider_or_post_tool_error(fail_during_trim):
    pending = PendingRunCommand(process=Mock(), prepared=Mock(), call_id="", unregister_cancel=Mock())
    calls = 0
    tokens = []

    def stream(*_args, **kwargs):
        nonlocal calls
        calls += 1
        tokens.append(kwargs["cancellation_token"])
        if calls == 1:
            yield _call(1)
            yield StreamDone(None)
        else:
            raise ProviderError("provider disconnected")

    def trim(items, **_kwargs):
        if fail_during_trim:
            raise RuntimeError("trim failed")
        return items

    with (
        patch("jarv.agent.stream_response", side_effect=stream),
        patch("jarv.agent._dispatch_run_command_with_ui", return_value=RunCommandDispatchResult("waiting", pending)),
        patch("jarv.turn_loop.trim_turn_input", side_effect=trim),
        patch("jarv.agent.sys.stdout", io.StringIO()),
    ):
        result = run_agent("run", DEFAULT_CONFIG, client=object(), incognito=True)
    assert result.error == ("trim failed" if fail_during_trim else "provider disconnected")
    pending.process.kill_tree.assert_called_once()
    pending.unregister_cancel.assert_called_once()
    assert tokens[0].cancelled


def test_cancellation_marks_second_tool_as_started_and_third_as_unstarted(tmp_path):
    context = SessionContext("test", "test", tmp_path / "history.json", datetime.now(timezone.utc))
    with (
        patch("jarv.agent.prepare_session_context", return_value=context),
        patch("jarv.agent.stream_response", return_value=iter([_call(1), _call(2), _call(3), StreamDone(None)])),
        patch("jarv.agent._dispatch_run_command_with_ui", side_effect=["first completed", TurnCancelled()]),
        patch("jarv.agent.sys.stdout", io.StringIO()),
    ):
        result = run_agent("run", DEFAULT_CONFIG, client=object())
    outputs = {item["call_id"]: item["output"] for item in load_history(context.history_file)
               if item.get("type") == "function_call_output"}
    assert result.cancelled
    assert outputs["call_1"] == "first completed"
    assert "may have made partial changes" in outputs["call_2"]
    assert "before execution" in outputs["call_3"]


def test_incognito_never_loads_stores_or_changes_persistent_session(tmp_path):
    streams = iter([iter([_call(1), StreamDone(None)]), iter([TextDelta("done"), StreamDone(None)])])
    with (
        patch("jarv.history.SESSIONS_DIR", tmp_path / "sessions"),
        patch("jarv.agent.prepare_session_context") as prepare,
        patch("jarv.agent.forget_current_session") as forget,
        patch("jarv.agent.load_history") as history,
        patch("jarv.agent.load_artifact_store") as artifacts,
        patch("jarv.agent.load_retained_output_store") as retained,
        patch("jarv.agent.record_response_usage") as usage,
        patch("jarv.agent.stream_response", side_effect=lambda *_a, **_k: next(streams)),
        patch("jarv.agent.check_command", return_value=(True, "")) as audit,
        patch("jarv.agent._dispatch_run_command_oneshot", return_value="ok"),
        patch("jarv.agent.sys.stdout", io.StringIO()),
    ):
        result = run_agent("run", DEFAULT_CONFIG, client=object(), new_session=True, incognito=True)
    assert result.error is None
    for reader in (prepare, forget, history, artifacts, retained, usage):
        reader.assert_not_called()
    audit.assert_called_once()
    assert audit.call_args.kwargs["usage_path"] is None
    assert list(tmp_path.iterdir()) == []


def test_ephemeral_context_has_unique_identity_without_creating_storage(tmp_path):
    with patch("jarv.history.SESSIONS_DIR", tmp_path / "sessions"):
        first, second = ephemeral_session_context(), ephemeral_session_context()
    assert first.session_id != second.session_id
    assert first.history_file.parent == tmp_path / "sessions"
    assert not first.history_file.parent.exists()


def test_sterile_restriction_applies_to_dispatch_hooks_and_direct_spawn():
    node = AgentNode("child", 1, "root", "task", True)
    store = ArtifactStore()
    with patch("jarv.orchestrator.spawn_tool_output") as spawn:
        output = dispatch_tool("spawn", {"children": []}, node, store, None, DEFAULT_CONFIG)
    assert "sterile" in output
    spawn.assert_not_called()
    with pytest.raises(ValueError, match="sterile"):
        spawn_batch(node, [], store, None, DEFAULT_CONFIG)
    hook = Mock()
    results = []
    execute_tool_calls([_call(1, "spawn")], node=node, store=store, client=None,
                      config=DEFAULT_CONFIG, hooks=ToolExecutionHooks(run_spawn=hook),
                      append_tool_result=lambda _item, output: results.append(output))
    hook.assert_not_called()
    assert "sterile" in results[0]


def test_headsup_incognito_new_and_session_commands_do_not_touch_saved_sessions(tmp_path):
    ready = threading.Event()
    ready.set()
    handler = Mock(side_effect=lambda _cmd, _rest, config, client, *_a: (config, client))
    with (
        patch("jarv.history.SESSIONS_DIR", tmp_path / "sessions"),
        patch("jarv.headsup.prepare_session_context") as prepare,
        patch("jarv.headsup.load_history") as history,
        patch("jarv.headsup.load_usage") as usage,
    ):
        app = HeadsupApp(DEFAULT_CONFIG, object(), args=SimpleNamespace(incognito=True, new=True),
                         agent_loader=({"module": SimpleNamespace()}, ready), handle_slash=handler,
                         maybe_command=lambda *_a: None,
                         render_console=Console(file=io.StringIO(), width=80))
        old_id = app.session_context.session_id
        app._run_slash("/new", [])
        assert app.session_context.session_id != old_id
        for command in ("/archive", "/undo", "/redo", "/history", "/session", "/sessions", "/usage", "/tree"):
            app._run_slash(command, [])
        app._usage_status(80)
        assert app._load_prompt_history() == []
    handler.assert_not_called()
    prepare.assert_not_called()
    history.assert_not_called()
    usage.assert_not_called()
    assert list(tmp_path.iterdir()) == []
