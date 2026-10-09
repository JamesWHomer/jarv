"""Provider contracts across streaming, history, rendering and the next request."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from jarv import agent
from jarv.anthropic_http import to_messages
from jarv.config import DEFAULT_CONFIG
from jarv.context_budget import history_to_api_items, trim_items_to_budget
from jarv.gemini_http import stream_content, to_contents
from jarv.history import load_history, save_history
from jarv.openai_http import build_chat_payload, build_responses_payload
from jarv.provider import (
    ProviderError, ReasoningDone, StreamDone, TextDelta, ToolCallDone,
    _stream_chat_completions, _stream_gemini,
    _to_chat_messages, validate_history_compatibility,
)
from jarv.turn_loop import collect_stream_response
from jarv.turn_records import append_assistant_response_input_items, append_tool_result_input_items
from jarv.usage import estimate_context_breakdown, estimate_item_tokens


@pytest.mark.parametrize("container", ["additional_kwargs", "model_extra", "provider_specific_fields"])
@pytest.mark.parametrize("metadata,expected", [
    ({}, False),
    ({"reasoning_content": "  ", "content": [{"type": "text", "text": "answer"}]}, False),
    ({"reasoning_content": "thought"}, True),
    ({"content": [{"type": "thinking_delta", "text": ""}]}, True),
    ({"nested": {"content": [{"type": "reasoning", "text": "thought"}]}}, True),
    ({"thinking_blocks": []}, False),
    ({"thinking": False}, True),
])
def test_reasoning_detection_in_provider_metadata(container, metadata, expected):
    from jarv.provider import _has_reasoning_signal

    assert _has_reasoning_signal({container: metadata}) is expected
    assert _has_reasoning_signal(SimpleNamespace(**{container: metadata})) is expected


def sse(chunks):
    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          content="".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks))


@pytest.mark.parametrize("metadata_only_end", [False, True])
def test_gemini_complete_reply_survives_terminal_rendering(monkeypatch, metadata_only_end):
    chunks = [
        {"candidates": [{"content": {"parts": [{"text": "Earlier explanation. "}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "Final sentence."}]}}]},
    ]
    if metadata_only_end:
        chunks.append({"candidates": [{"finishReason": "STOP"}]})
    else:
        chunks[-1]["candidates"][0]["finishReason"] = "STOP"
    chunks.append({"usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 7}})
    monkeypatch.setattr(agent, "InPlaceLive", Mock())
    monkeypatch.setattr(agent.console, "print", Mock())
    renderer = agent._TurnRenderer(ui=None, interactive=True, status_items=[], metadata={})
    with httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(lambda _: sse(chunks))) as client:
        result = collect_stream_response(
            lambda: _stream_gemini(client, {}, "gemini-test", "", [], []),
            on_event=renderer.on_stream_event, on_attempt_end=renderer.on_stream_attempt_end,
        )
    expected = "Earlier explanation. Final sentence."
    assert result.reply_text == renderer.reply_text == result.final_response["output_text"] == expected
    assert result.final_response["usage"]["output_tokens"] == 7
    history = []
    append_assistant_response_input_items([], result.reasoning_items, result.reply_text,
                                          result.tool_calls, history=history)
    assert history[-1]["content"] == expected


def test_gemini_aggregate_preserves_signed_parts_and_blocked_prompt():
    parts = [{"text": "thought", "thought": True, "thoughtSignature": "sig1"},
             {"text": "answer"},
             {"functionCall": {"id": "g1", "name": "read", "args": {}}, "thoughtSignature": "sig2"}]
    chunks = [{"candidates": [{"index": 0, "content": {"role": "model", "parts": [part]}}]}
              for part in parts]
    chunks.append({"candidates": [{"index": 0, "finishReason": "STOP"}]})
    with httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(lambda _: sse(chunks))) as client:
        events = list(stream_content(client, "gemini-test", {}))
    final = events[-1]["response"]
    assert final["candidates"][0]["content"]["parts"] == parts
    assert final["output_text"] == "answer"
    assert len([event for event in events if event["type"] == "tool_call"]) == 1
    with httpx.Client(base_url="https://example.test", transport=httpx.MockTransport(
        lambda _: sse([{"promptFeedback": {"blockReason": "SAFETY"}}]),
    )) as client:
        assert list(stream_content(client, "gemini-test", {}))[-1]["response"]["output_text"] == ""


def deepseek_stream(monkeypatch, chunks, items=(), *, tools=None, effort=None):
    stream = Mock(return_value=iter(chunks))
    monkeypatch.setattr("jarv.openai_http.stream_chat", stream)
    result = collect_stream_response(lambda: _stream_chat_completions(
        None, "deepseek-v4-pro", "system", tools or [], list(items),
        reasoning={"effort": effort} if effort is not None else None,
        config={"provider": "deepseek"},
    ))
    return result, stream


def test_deepseek_reasoning_survives_tool_round_and_saved_final(monkeypatch, tmp_path):
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "First "}}]},
        {"choices": [{"delta": {"reasoning_content": "second", "content": "Inspecting."}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "read", "arguments": "{}"}},
            {"index": 1, "id": "c2", "function": {"name": "read", "arguments": "{}"}},
        ]}, "finish_reason": "tool_calls"}]},
    ]
    result, _ = deepseek_stream(monkeypatch, chunks)
    history = [{"role": "user", "content": "Inspect"}]
    append_assistant_response_input_items(
        [], result.reasoning_items, result.reply_text, result.tool_calls,
        history=history, provider_metadata=result.provider_metadata,
    )
    for tool in result.tool_calls:
        append_tool_result_input_items([], tool, "content", history=history, include_call=False)
    tools = [{"type": "function", "name": "read", "parameters": {"type": "object"}}]
    final, stream = deepseek_stream(monkeypatch, [
        {"choices": [{"delta": {"reasoning_content": "Finished reasoning", "content": "Done"}, "finish_reason": "stop"}]},
    ], history_to_api_items(history), tools=tools)
    messages = stream.call_args.args[1]["messages"]
    assistant = next(message for message in messages if message["role"] == "assistant")
    assert assistant["reasoning_content"] == "First second"
    assert assistant["content"] == "Inspecting."
    assert len(assistant["tool_calls"]) == 2
    assert all("provider_metadata" not in message for message in messages)
    append_assistant_response_input_items([], [], final.reply_text, [], history=history,
                                          provider_metadata=final.provider_metadata)
    path = tmp_path / "history.json"
    save_history(history, path)
    restored = history_to_api_items(load_history(path))
    restored.append({"role": "user", "content": "Continue"})
    _, stream = deepseek_stream(monkeypatch, [
        {"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]},
    ], restored, tools=tools)
    assert stream.call_args.args[1]["messages"][-2]["reasoning_content"] == "Finished reasoning"


@pytest.mark.parametrize("metadata", [None, {}, {"provider": "anthropic", "reasoning_content": "foreign"},
                                      {"provider": "deepseek", "reasoning_content": None}])
def test_deepseek_legacy_history_requires_new_chat_before_network(monkeypatch, metadata):
    history = [{"role": "user", "content": "Old"}, {"role": "assistant", "content": "Answer"}]
    if metadata is not None:
        history[-1]["provider_metadata"] = metadata
    before = copy.deepcopy(history)
    stream = Mock()
    monkeypatch.setattr("jarv.openai_http.stream_chat", stream)
    for effort in (None, "high"):
        with pytest.raises(ProviderError, match="/new or --new"):
            list(_stream_chat_completions(None, "deepseek-v4-pro", "", [{"type": "function"}], history,
                                         reasoning={"effort": effort}, config={"provider": "deepseek"}))
    stream.assert_not_called()
    assert history == before


@pytest.mark.parametrize("effort,tools,metadata", [
    ("none", [{}], None), (None, [], None),
    (None, [{}], {"provider": "deepseek", "reasoning_content": ""}),
])
def test_deepseek_compatible_histories_are_allowed(monkeypatch, effort, tools, metadata):
    item = {"role": "assistant", "content": "Prior", "provider_metadata": metadata}
    result, stream = deepseek_stream(monkeypatch, [
        {"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]},
    ], [item], tools=tools, effort=effort)
    stream.assert_called_once()
    assert result.provider_metadata == {"provider": "deepseek"}


@pytest.mark.parametrize("effort,thinking", [("none", "disabled"), ("high", "enabled"), (None, None)])
def test_deepseek_native_effort_payload(effort, thinking):
    payload = build_chat_payload("deepseek-v4-pro", [], provider_name="deepseek",
                                 reasoning={"effort": effort})
    assert payload.get("thinking") == ({"type": thinking} if thinking else None)
    assert payload.get("reasoning_effort") == ("high" if effort == "high" else None)
    assert "thinking" not in build_chat_payload("model", [], provider_name="openrouter",
                                                 reasoning={"effort": "none"})


@pytest.mark.parametrize("tagged", [False, True])
def test_opaque_blocks_are_isolated_across_providers(tagged):
    thinking = {"type": "thinking", "thinking": "private", "signature": "anthropic-sig"}
    gemini = {"text": "thought", "thought": True, "thoughtSignature": "gemini-sig"}
    call = {"functionCall": {"name": "read", "args": {}, "id": "g1"}, "thoughtSignature": "call-sig"}
    history = [{"role": "user", "content": "Inspect"},
               {"type": "reasoning", "id": "thinking_0", "summary": [], "provider_content": [thinking]},
               {"type": "reasoning", "id": "gemini-thinking", "summary": [], "provider_content": [gemini]},
               {"type": "function_call", "id": "g1", "call_id": "g1", "name": "read", "arguments": "{}",
                "provider_content": [call]},
               {"type": "function_call_output", "call_id": "g1", "output": "file"}]
    if tagged:
        for item, provider in zip(history[1:4], ["anthropic", "gemini", "gemini"]):
            item["provider_metadata"] = {"provider": provider}
    api_items = history_to_api_items(history)
    gemini_parts = to_contents(api_items)[1]["parts"]
    assert thinking not in gemini_parts and gemini in gemini_parts and call in gemini_parts
    anthropic_blocks = to_messages(api_items)[1]["content"]
    assert thinking in anthropic_blocks and gemini not in anthropic_blocks
    payload = build_responses_payload("model", "", [], api_items)
    assert not any(item.get("type") == "reasoning" for item in payload["input"])
    assert all("provider_metadata" not in item and "provider_content" not in item for item in payload["input"])
    assert any(item.get("type") == "function_call_output" for item in payload["input"])


def test_openai_legacy_reasoning_is_preserved_but_unknown_blocks_are_not():
    native = {"type": "reasoning", "id": "rs_native", "summary": []}
    unknown = {"type": "reasoning", "id": "thinking_unknown", "summary": [],
               "provider_content": [{"unexpected": "content"}]}
    payload = build_responses_payload("model", "", [], history_to_api_items([native, unknown]))
    assert payload["input"] == [native]
    assert to_contents(history_to_api_items([unknown])) == []
    unknown_without_blocks = {"type": "reasoning", "id": "other-protocol", "summary": []}
    normalized = history_to_api_items([unknown_without_blocks])
    assert build_responses_payload("model", "", [], normalized)["input"] == []


def test_deepseek_reasoning_is_counted_once_and_trimmed_with_whole_turn():
    metadata = {"provider": "deepseek", "reasoning_content": "x" * 4000}
    old = [{"role": "user", "content": "old"},
           {"role": "assistant", "content": "answer", "provider_metadata": metadata}]
    recent = [{"role": "user", "content": "new"}]
    assert estimate_item_tokens("model", old[-1]) == 1001
    assert estimate_context_breakdown("model", "", [], old)["reasoning"] == 1000
    assert trim_items_to_budget(old + recent, "model", 500) == recent


def test_root_records_deepseek_metadata_on_text_only_completion(monkeypatch):
    metadata = {"provider": "deepseek", "reasoning_content": "complete reasoning"}
    saved = []
    monkeypatch.setattr(agent, "build_instructions", lambda *args, **kwargs: "system")
    monkeypatch.setattr(agent, "stream_response", lambda *args, **kwargs: iter([
        TextDelta("Answer"), StreamDone({}, provider_metadata=metadata),
    ]))
    monkeypatch.setattr(agent.SessionPersistence, "save_turn", lambda self: saved.extend(self.history))
    result = agent.run_agent("hello", {**DEFAULT_CONFIG, "provider": "deepseek"},
                             client=object(), incognito=True, ui=SimpleNamespace())
    assert result.error is None
    assert saved[-1]["provider_metadata"] == metadata
    assert _to_chat_messages("", history_to_api_items(saved), provider_name="deepseek")[-1]["reasoning_content"] == "complete reasoning"


def test_cancel_checkpoint_preserves_provider_metadata_on_reasoning_and_calls():
    metadata = {"provider": "deepseek", "reasoning_content": "Tool reasoning"}
    persistence = agent.SessionPersistence(incognito=True)
    renderer = agent._TurnRenderer(ui=None, interactive=False, status_items=[], metadata={})
    renderer.provider_metadata = metadata
    renderer.reasoning_items = [ReasoningDone(
        "thinking_0", [], provider_content=[{"type": "thinking", "thinking": "native", "signature": "sig"}],
        provider_metadata={"provider": "anthropic"},
    )]
    renderer.tool_calls = [ToolCallDone("c1", "c1", "read", "{}"), ToolCallDone("c2", "c2", "read", "{}")]
    checkpointer = agent.TurnCheckpointer(persistence=persistence, renderer=renderer, status_items=[])
    checkpointer.checkpoint_cancelled_turn()
    assert persistence.history[0]["provider_metadata"] == {"provider": "anthropic"}
    calls = [item for item in persistence.history if item.get("type") == "function_call"]
    assert calls[0]["provider_metadata"] == metadata
    assert "provider_metadata" not in calls[1]


def test_interactive_continuation_replays_deepseek_metadata(monkeypatch):
    from jarv.cancellation import CancellationToken
    from jarv.orchestrator import PendingRunCommand, RunCommandPrepared

    metadata = {"provider": "deepseek", "reasoning_content": "Choose terminal input"}
    renderer = agent._TurnRenderer(ui=None, interactive=False, status_items=[], metadata={})
    renderer.reply_text = "hello"
    renderer.provider_metadata = metadata
    pending = PendingRunCommand(process=Mock(), prepared=RunCommandPrepared("cmd", 100, 100, 200), call_id="c0")
    # A rejected input still enters the live conversation before the next try.
    monkeypatch.setattr(agent, "_continue_interactive_command", lambda *args, **kwargs: (None, "bad input", "invalid", None))
    items, _ = agent._advance_interactive_continuation(
        pending, renderer, [{"role": "user", "content": "Continue command"}],
        config=DEFAULT_CONFIG, cancellation_token=CancellationToken(), retained_store=None,
        ui=None, interactive_help={"sent": True},
    )
    message = next(item for item in _to_chat_messages("", items, provider_name="deepseek") if item["role"] == "assistant")
    assert message["reasoning_content"] == metadata["reasoning_content"]


def test_legacy_tool_only_deepseek_turn_is_blocked(monkeypatch):
    stream = Mock()
    monkeypatch.setattr("jarv.openai_http.stream_chat", stream)
    items = [{"type": "function_call", "id": "c1", "call_id": "c1", "name": "read", "arguments": "{}"},
             {"type": "function_call_output", "call_id": "c1", "output": "file"}]
    with pytest.raises(ProviderError, match="/new or --new"):
        list(_stream_chat_completions(None, "deepseek-v4-pro", "", [{}], items, config={"provider": "deepseek"}))
    stream.assert_not_called()


@pytest.mark.parametrize("reasoning_delta,allowed", [({}, False), ({"reasoning_content": None}, False),
                                                   ({"reasoning_content": ""}, True)])
def test_deepseek_missing_reasoning_is_distinct_from_captured_empty(monkeypatch, reasoning_delta, allowed):
    result, _ = deepseek_stream(monkeypatch, [
        {"choices": [{"delta": {"content": "Answer", **reasoning_delta}, "finish_reason": "stop"}]},
    ], effort="none")
    assert ("reasoning_content" in result.provider_metadata) is allowed
    history = []
    append_assistant_response_input_items([], [], result.reply_text, [], history=history,
                                          provider_metadata=result.provider_metadata)
    if allowed:
        validate_history_compatibility({"provider": "deepseek"}, [{}], history)
    else:
        with pytest.raises(ProviderError, match="fresh chat"):
            validate_history_compatibility({"provider": "deepseek"}, [{}], history)


def test_root_legacy_preflight_leaves_history_settings_and_sidecars_unchanged(monkeypatch, tmp_path):
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps([
        {"role": "user", "content": "Legacy prompt"},
        {"role": "assistant", "content": "Legacy answer"},
        {"role": "user", "content": "Recent " * 1000},
        {"role": "assistant", "content": "Recent answer", "provider_metadata":
         {"provider": "deepseek", "reasoning_content": "captured"}},
    ]), encoding="utf-8")
    sidecars = [tmp_path / name for name in ("redo.json", "branches.json", "reads.json", "artifacts.json", "settings.json")]
    for path in sidecars:
        path.write_text('{"must":"remain unchanged"}', encoding="utf-8")
    before = {path: path.read_bytes() for path in [history_path, *sidecars]}
    config = {**DEFAULT_CONFIG, "provider": "deepseek", "model": "deepseek-v4-pro",
              "reasoning_effort": "high", "context_window": 1000}
    original_config = copy.deepcopy(config)
    prepare = Mock(return_value=SimpleNamespace(history_file=history_path))
    monkeypatch.setattr(agent, "prepare_session_context", prepare)
    provider = Mock()
    monkeypatch.setattr(agent, "stream_response", provider)
    monkeypatch.setattr(agent, "_prepare_client_and_instructions", Mock(side_effect=AssertionError("client preparation called")))
    save = Mock(side_effect=AssertionError("history saved"))
    monkeypatch.setattr(agent.SessionPersistence, "save", save)
    normalize = Mock(side_effect=AssertionError("history normalized"))
    monkeypatch.setattr("jarv.session_tree.preserve_redo_branches", normalize)
    tool = Mock(side_effect=AssertionError("tool executed"))
    monkeypatch.setattr(agent, "execute_tool_calls", tool)
    result = agent.run_agent("New prompt", config, client=object(), ui=SimpleNamespace())
    assert "/new or --new" in result.error
    prepare.assert_called_once_with(mark_message=False, persist_metadata=False)
    provider.assert_not_called()
    save.assert_not_called()
    normalize.assert_not_called()
    tool.assert_not_called()
    assert {path: path.read_bytes() for path in before} == before
    assert config == original_config
