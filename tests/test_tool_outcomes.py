"""Execution outcomes survive formatting, persistence, and history replay."""

import io
import json
from types import SimpleNamespace

import pytest
from rich.console import Console

from jarv import agent, edit_tool, orchestrator, web
from jarv.agent_ui import InteractiveCommandCard, _dispatch_spawn_with_ui
from jarv.artifacts import ArtifactStore
from jarv.config import DEFAULT_CONFIG
from jarv.history import load_history, save_history
from jarv.interactive_command import _finalize_interactive_record
from jarv.orchestrator import AgentNode, ToolExecutionHooks, execute_tool_calls
from jarv.provider import ToolCallDone
from jarv.response_items import function_call_output_item, to_response_input_item
from jarv.session_render import _tool_call_output, tool_call_card
from jarv.shell import CommandResult
from jarv.tool_outputs import (
    ToolOutcome, summarize_tool_output, tool_outcome, tool_output_failed,
    with_tool_outcome,
)
from jarv.turn_records import append_tool_result_input_items


def _render(card):
    stream = io.StringIO()
    Console(file=stream, width=120, color_system=None).print(card)
    return stream.getvalue()


def _run_and_reload(tmp_path, name, args, *, config=None, hooks=None):
    history, api_items = [], []
    call = ToolCallDone(id="fc_test", call_id="call_test", name=name, arguments=json.dumps(args))
    execute_tool_calls(
        [call],
        node=AgentNode("root", 0, None, "test", False),
        store=ArtifactStore(), client=None,
        config=config or {**DEFAULT_CONFIG, "command_safety": "none"},
        hooks=hooks,
        append_tool_result=lambda item, output: append_tool_result_input_items(
            api_items, item, output, history=history,
        ),
    )
    path = tmp_path / "history.json"
    save_history(history, path)
    reloaded = load_history(path)
    assert "outcome" not in api_items[-1]
    assert api_items[-1]["output"] == reloaded[-1]["output"]
    assert type(api_items[-1]["output"]) in (str, list)
    replay = _tool_call_output(reloaded, 0, call.call_id)
    return reloaded[-1], _render(tool_call_card(reloaded[0], replay))


@pytest.mark.parametrize("root_ui", [False, True])
@pytest.mark.parametrize("kind", ["nonzero", "timeout", "launch_error", "denied", "success"])
def test_command_outcome_survives_truncation_and_reload(tmp_path, monkeypatch, root_ui, kind):
    config = {**DEFAULT_CONFIG, "command_safety": "none", "interactive_commands": False}
    exit_code = {"nonzero": 7, "timeout": -1, "launch_error": None}.get(kind, 0)
    # Deliberately print an error-looking prefix on SUCCESS as well as failure.
    result = CommandResult("test", "[error: printed data]\n" * 100, "", exit_code,
                           timed_out=kind == "timeout")
    monkeypatch.setattr(orchestrator, "execute_command", lambda *a, **k: result)
    allowed = lambda *a, **k: (kind != "denied", "Approval was declined")
    monkeypatch.setattr(orchestrator, "check_run_command", allowed)
    monkeypatch.setattr(agent, "_agent_check_run_command", allowed)
    live_cards = []
    ui = SimpleNamespace(show_tool_card=live_cards.append, show_notice=lambda *a: None)
    hooks = ToolExecutionHooks(run_command=lambda args: agent._dispatch_run_command_with_ui(
        args, config, ui=ui,
    )) if root_ui else None
    stored, rendered = _run_and_reload(
        tmp_path, "run_command", {"command": "test", "head_chars": 24, "tail_chars": 0},
        config=config, hooks=hooks,
    )
    status = {"nonzero": "failed", "timeout": "timed_out", "launch_error": "failed"}.get(kind, kind)
    assert stored["outcome"] == {"status": status, "exit_code": None if kind == "denied" else exit_code}
    assert "[exit code" not in stored["output"]
    assert "[timed out" not in stored["output"]
    expected = "done" if kind == "success" else "failed"
    assert expected in rendered
    assert ("\u2713 done" in rendered) == (kind == "success")
    if root_ui and kind != "denied":
        live_cards[-1].display_mode = "fullscreen"
        assert expected in _render(live_cards[-1])


@pytest.mark.parametrize("kind", ["conflict", "denied", "success"])
def test_edit_outcome_survives_reload_without_recognizable_message(tmp_path, monkeypatch, kind):
    path = tmp_path / "edit.txt"
    path.write_text("before", encoding="utf-8")

    def check(*args, **kwargs):
        if kind == "conflict":
            path.write_text("external change", encoding="utf-8")
        return kind != "denied", "Approval was declined"

    monkeypatch.setattr(edit_tool, "_check_edit", check)
    monkeypatch.setattr(edit_tool, "_conflict", lambda path: "The file changed")
    stored, rendered = _run_and_reload(tmp_path, "edit", {
        "path": str(path), "old_text": "before", "new_text": "after",
    })
    assert stored["outcome"]["status"] == ("failed" if kind == "conflict" else kind)
    assert ("\u2713 done" in rendered) == (kind == "success")
    assert ("\u2717 failed" in rendered) == (kind != "success")


def test_parallel_search_failure_survives_nudge_and_truncation(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise web.WebToolError("x" * 1000)

    monkeypatch.setattr(web, "search_web", fail)
    stored, rendered = _run_and_reload(
        tmp_path, "web_search", {"query": "test"},
        config={**DEFAULT_CONFIG, "max_tool_output_chars": 8},
    )
    assert stored["outcome"]["status"] == "failed"
    assert "\u2717 failed" in rendered


@pytest.mark.parametrize("name,args,settings", [
    ("read", {"input": "missing-file"}, {}),
    ("run_command", {"command": ""}, {}),
    ("edit", {}, {}),
    ("finish", {}, {}),
    ("unknown", {}, {}),
    ("read", {"input": "file"}, {"disabled_tools": ["read"]}),
    ("run_command", {"command": "test"}, {"disabled_tools": ["run_command"]}),
])
def test_early_failures_persist_outcomes(tmp_path, name, args, settings):
    stored, rendered = _run_and_reload(tmp_path, name, args, config={**DEFAULT_CONFIG, **settings})
    assert stored["outcome"]["status"] in {"failed", "denied"}
    assert "\u2717 failed" in rendered


def test_multimodal_outcome_is_separate_from_provider_blocks(tmp_path):
    blocks = [{"type": "input_text", "text": "[error: merely image text]"},
              {"type": "input_image", "image_url": "data:image/png;base64,YQ=="}]
    output = with_tool_outcome(blocks, "success")
    assert not tool_output_failed(summarize_tool_output(output))
    item = function_call_output_item("image", output)
    save_history([item], tmp_path / "image.json")
    reloaded = load_history(tmp_path / "image.json")[0]
    assert reloaded["outcome"] == {"status": "success", "exit_code": None}
    assert to_response_input_item(reloaded) == {
        "type": "function_call_output", "call_id": "image", "output": blocks,
    }


@pytest.mark.parametrize("status", ["failed", "timed_out", "cancelled", "success"])
def test_interactive_finalization_replaces_running_outcome(status):
    stored = function_call_output_item("interactive", with_tool_outcome("waiting", "running"))
    pending = SimpleNamespace(output_item=stored, transcript_segments=["first output", "stdin> answer"])
    final = with_tool_outcome("last output", ToolOutcome(status, 0 if status == "success" else 1))
    _finalize_interactive_record(pending, final)
    history = [{"name": "run_command", "arguments": '{"command":"test"}'}, json.loads(json.dumps(stored))]
    replay = _tool_call_output(history, 0, "interactive")
    assert tool_outcome(replay) == tool_outcome(final)
    assert "first output\nstdin> answer\nlast output" == replay
    assert ("\u2713 done" in _render(tool_call_card(history[0], replay))) == (status == "success")


def test_grouped_calls_match_outcomes_by_call_id():
    calls = [{"type": "function_call", "call_id": call_id, "name": "read", "arguments": "{}"}
             for call_id in ("first", "second")]
    history = calls + [
        function_call_output_item("second", with_tool_outcome("data", "success")),
        function_call_output_item("first", with_tool_outcome("data", "failed")),
    ]
    history = json.loads(json.dumps(history))
    assert tool_output_failed(_tool_call_output(history, 0, "first"))
    assert not tool_output_failed(_tool_call_output(history, 1, "second"))


@pytest.mark.parametrize("output", [
    "[edit conflict: file changed]", "[command denied by user]",
    "some output\n[exit code 2]", "some output\n[timed out after 1 seconds]",
])
def test_legacy_failures_still_render_as_failed(output):
    item = {"name": "run_command", "arguments": '{"command":"test"}'}
    assert "\u2717 failed" in _render(tool_call_card(item, output))


def test_unanswered_call_does_not_claim_success():
    item = {"type": "function_call", "call_id": "pending", "name": "read", "arguments": "{}"}
    replay = _tool_call_output([item], 0, "pending")
    rendered = _render(tool_call_card(item, replay))
    assert "unknown" in rendered
    assert "\u2713 done" not in rendered


@pytest.mark.parametrize("exit_code", [None, 3, 0])
def test_interactive_live_card_uses_exit_code(exit_code):
    from rich.text import Text

    card = InteractiveCommandCard("test", "", "fullscreen", 0)
    card.add_step(Text("finished"), "", None, exited=True, exit_code=exit_code)
    rendered = _render(card)
    assert ("\u2713 done" in rendered) == (exit_code == 0)
    assert ("\u2717 failed" in rendered) == (exit_code != 0)


def test_spawn_failure_has_matching_live_and_history_status(tmp_path, monkeypatch):
    result = {"label": "child", "status": "failed", "reason": "Could not complete"}

    def finish(*args, observer=None, **kwargs):
        observer.on_child_done("root", "child", result)
        return [result]

    monkeypatch.setattr(orchestrator, "spawn_batch", finish)
    live = []
    ui = SimpleNamespace(show_tool_card=lambda card: live.append(_render(card)))
    config = {**DEFAULT_CONFIG, "tool_call_display": "fullscreen"}
    root = AgentNode("root", 0, None, "test", False)
    hooks = ToolExecutionHooks(run_spawn=lambda args: _dispatch_spawn_with_ui(
        args, root, ArtifactStore(), None, config, ui=ui,
    ))
    stored, rendered = _run_and_reload(
        tmp_path, "spawn", {"children": [{"label": "child", "task": "work"}]},
        config=config, hooks=hooks,
    )
    assert stored["outcome"]["status"] == "failed"
    assert "1/1 failed" in live[-1]
    assert "\u2717 failed" in rendered
    assert "\u2717 child" in rendered


def test_explicit_success_overrides_original_arguments_that_were_salvaged():
    item = {"name": "run_command", "arguments": '```json\n{"command":"test"}\n```'}
    rendered = _render(tool_call_card(item, with_tool_outcome("[error: printed data]", "success")))
    assert "\u2713 done" in rendered
