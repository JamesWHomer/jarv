"""An agent's shell directory must also scope files and project context."""

import json
from pathlib import Path

import pytest

from jarv.agent import _build_tool_hooks, build_instructions
from jarv.artifacts import ArtifactStore
from jarv.cancellation import CancellationToken
from jarv.config import DEFAULT_CONFIG
from jarv.edit_tool import classify_edit
from jarv.orchestrator import AgentNode, dispatch_tool, execute_tool_calls, run_subagent_loop
from jarv.provider import StreamDone, ToolCallDone
from jarv.read_tool import dispatch_read_batch
from jarv.retained_outputs import RetainedOutputStore
from jarv.shell import ShellState, execute_command


@pytest.fixture
def projects(tmp_path, monkeypatch):
    original = tmp_path / "original"
    target = tmp_path / "target"
    for directory in (original, target):
        directory.mkdir()
        (directory / "same.txt").write_text(directory.name, encoding="utf-8")
        (directory / "AGENTS.md").write_text(
            f"Instructions for {directory.name}", encoding="utf-8"
        )
    monkeypatch.chdir(original)
    return original, target


def test_cd_then_parallel_reads_and_edit_use_agent_directory(projects):
    original, target = projects
    node = AgentNode("root", 0, None, "test", False)
    config = {**DEFAULT_CONFIG, "command_safety": "none"}
    store = ArtifactStore()
    outputs = []
    calls = [
        ("run_command", {"command": f'cd "{target}"'}),
        ("read", {"input": "same.txt"}),
        ("read", {"input": "AGENTS.md"}),
        ("edit", {"path": "same.txt", "old_text": "target", "new_text": "edited"}),
    ]
    execute_tool_calls(
        [ToolCallDone(id=str(i), call_id=str(i), name=name, arguments=json.dumps(args))
         for i, (name, args) in enumerate(calls)],
        node=node, store=store, client=None, config=config,
        append_tool_result=lambda _call, output: outputs.append(output),
    )
    assert Path(node.shell_state.cwd) == target
    assert "\n\ntarget" in outputs[1]
    assert "Instructions for target" in outputs[2]
    assert "[EDIT RESULT]" in outputs[3]
    assert (target / "same.txt").read_text() == "edited"
    assert (original / "same.txt").read_text() == "original"
    assert Path.cwd() == original


@pytest.mark.parametrize("count", [1, 2])
def test_read_batch_forwards_explicit_directory(projects, count):
    original, target = projects
    results = dispatch_read_batch(
        [{"input": "same.txt"}] * count,
        cwd=target, visible_labels=set(), artifact_store=ArtifactStore(),
        retained_store=RetainedOutputStore(), config=DEFAULT_CONFIG,
    )
    assert len(results) == count
    assert all("\n\ntarget" in output for output in results)
    assert Path.cwd() == original


def test_edit_approval_boundary_uses_agent_directory(projects):
    original, target = projects
    assert classify_edit(target / "same.txt", cwd=target) == (False, "")
    assert classify_edit(original / "same.txt", cwd=target) == (
        True, "file outside the current working directory"
    )
    assert classify_edit(target / ".hidden", cwd=target) == (
        True, "hidden file or directory"
    )


def test_instructions_and_git_use_explicit_directory(projects, monkeypatch):
    original, target = projects
    calls = []

    def git(args, cwd):
        calls.append(cwd)
        if args == ["rev-parse", "--show-toplevel"]:
            return str(target)
        if args == ["branch", "--show-current"]:
            return "target-branch"
        return ""

    monkeypatch.setattr("jarv.project_context._run_git", git)
    instructions = build_instructions(DEFAULT_CONFIG, cwd=str(target))
    assert f"CWD: {target}" in instructions
    assert "Instructions for target" in instructions
    assert "Instructions for original" not in instructions
    assert "branch: target-branch" in instructions
    assert calls and all(cwd == target for cwd in calls)
    assert Path.cwd() == original


def test_child_directory_changes_do_not_redirect_parent_files(projects):
    original, target = projects
    parent = AgentNode("root", 0, None, "test", False,
                       shell_state=ShellState(str(original)))
    child = AgentNode("child", 1, "root", "test", False,
                      shell_state=parent.shell_state.copy())
    execute_command(f'cd "{target}"', shell_state=child.shell_state)
    store = ArtifactStore()
    assert "\n\ntarget" in dispatch_tool(
        "read", {"input": "same.txt"}, child, store, None, DEFAULT_CONFIG
    )
    assert "\n\noriginal" in dispatch_tool(
        "read", {"input": "same.txt"}, parent, store, None, DEFAULT_CONFIG
    )
    assert Path.cwd() == original


def test_root_edit_hook_uses_latest_shell_directory(projects, monkeypatch):
    original, target = projects
    node = AgentNode("root", 0, None, "test", False)
    monkeypatch.setattr("jarv.agent._print_tool_card", lambda *a, **kw: None)
    hooks = _build_tool_hooks(
        config={**DEFAULT_CONFIG, "command_safety": "none"}, history=[],
        usage_path=None, session_id="test", cancellation_token=CancellationToken(),
        retained_store=RetainedOutputStore(), ui=None, interactive_help={},
        root_node=node, artifact_store=ArtifactStore(), client=None,
    )
    # The closure must consult live state, not capture the initial directory.
    execute_command(f'cd "{target}"', shell_state=node.shell_state)
    output = hooks.run_edit(
        {"path": "same.txt", "old_text": "target", "new_text": "edited"}
    )
    assert "[EDIT RESULT]" in output
    assert (target / "same.txt").read_text() == "edited"
    assert (original / "same.txt").read_text() == "original"


def test_subagent_refreshes_project_instructions_after_cd(projects, monkeypatch):
    original, target = projects
    prompts = []

    def stream(*args, **kwargs):
        prompts.append(args[3])
        if len(prompts) == 1:
            name, arguments = "run_command", {"command": f'cd "{target}"'}
        else:
            name, arguments = "finish", {"longform": "done", "tldr": "done"}
        yield ToolCallDone(id=str(len(prompts)), call_id=str(len(prompts)),
                           name=name, arguments=json.dumps(arguments))
        yield StreamDone(response=None)

    monkeypatch.setattr("jarv.orchestrator.stream_response", stream)
    monkeypatch.setattr("jarv.project_context._run_git", lambda *args: None)
    result = run_subagent_loop(
        AgentNode("child", 1, "root", "test", False, incognito=True),
        ArtifactStore(), None, {**DEFAULT_CONFIG, "command_safety": "none"},
    )
    assert result == ("done", "done")
    assert len(prompts) == 2
    assert "Instructions for original" in prompts[0]
    assert f"CWD: {target}" in prompts[1]
    assert "Instructions for target" in prompts[1]
    assert "Instructions for original" not in prompts[1]
    assert Path.cwd() == original
