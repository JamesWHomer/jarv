"""CLI contracts exercised through real agent/tool loops with fake providers."""

import copy
import io
import json
import os
import sys
import threading
import time
from unittest.mock import Mock

import pytest

from jarv import agent, cli, history, orchestrator, safety
from jarv.config import DEFAULT_CONFIG
from jarv.provider import ProviderError, RetryableStreamError, StreamDone, TextDelta, ToolCallDone
from jarv.run_control import RunControl, RunStopped
from jarv.cancellation import CancellationToken


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(check_updates=False, project_context=False, audit=False)
    monkeypatch.setattr(cli, "load_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr("jarv.config.is_setup_complete", lambda *_: True)
    monkeypatch.setattr("jarv.provider_auth.resolve_api_key", lambda *_: "test-key")
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(history, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(history, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(agent, "_prepare_client_and_instructions", lambda *a, **kw: (object(), "system"))
    monkeypatch.setattr(agent, "record_response_usage", lambda *a, **kw: None)
    monkeypatch.setattr(agent, "stream_response", lambda *a, **kw: iter([TextDelta("answer"), StreamDone(None)]))
    return config


def invoke(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["jarv", *arguments])
    cli.main()


def call(name, arguments, ident="1"):
    return ToolCallDone(id="fc_" + ident, call_id="call_" + ident,
                        name=name, arguments=json.dumps(arguments))


def test_overrides_are_typed_temporary_and_explicit_flags_win():
    original = copy.deepcopy(DEFAULT_CONFIG)
    args = cli._build_parser().parse_args([
        "-c", "command_timeout=10", "--timeout", "20",
        "-c", "project_context=false", "-c", 'disabled_tools=["read"]',
        "--tools", "read,web_search", "--service-tier", "priority",
        "--system", "", "--base-url", "http://localhost:1234/v1",
        "--no-color", "--no-update-check", "--command-safety", "all",
    ])
    config = cli._apply_cli_overrides(original, args)
    assert config["command_timeout"] == 20
    assert config["project_context"] is False
    assert config["service_tiers"] == {"openai": "priority"}
    assert config["system_prompt"] == ""
    assert config["base_url"] == "http://localhost:1234/v1"
    assert config["colour"] is config["check_updates"] is False
    assert config["command_safety"] == "all"
    assert set(config["disabled_tools"]) == {"run_command", "edit", "spawn", "ask_user"}
    assert original == DEFAULT_CONFIG


@pytest.mark.parametrize("args", [
    ["--max-turns", "0"], ["--run-timeout", "-1"], ["--timeout", "no"],
    ["-c", "unknown=1"], ["-c", "audit=yes"], ["-c", "max_tool_output_chars=-1"],
    ["-c", 'disabled_tools=["unknown"]'], ["--tools", "read,unknown"],
    ["--tools", "read", "--no-tools"], ["--new", "--session", "abc"],
    ["--incognito", "--session", "abc"], ["--system", "x", "--system-file", "y"],
    ["--quiet", "--verbose"], ["--pro", "openai"],
])
def test_invalid_arguments_rejected(args):
    with pytest.raises(SystemExit) as error:
        cli._build_parser().parse_args(args)
    assert error.value.code == 2


@pytest.mark.parametrize("flag", ["--model=x", "--no-tools", "--non-interactive", "--output-format=json"])
def test_slash_commands_reject_runtime_flags(monkeypatch, flag):
    dispatch = Mock()
    monkeypatch.setattr(cli, "_run_slash_command", dispatch)
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "/help", flag)
    assert error.value.code == 2
    dispatch.assert_not_called()


def test_prompt_and_system_files_are_resolved_before_cwd(runtime, monkeypatch, tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    system = tmp_path / "system.txt"
    target = tmp_path / "project"
    target.mkdir()
    prompt.write_text("/help\nThis is a prompt, not a command.", encoding="utf-8")
    system.write_text("custom system", encoding="utf-8")
    captured = []
    original = agent.run_agent

    def capture(query, config, **kwargs):
        captured.append((query, config["system_prompt"], os.getcwd(), agent.get_session_shell_state().cwd))
        return original(query, config, **kwargs)

    monkeypatch.setattr(agent, "run_agent", capture)
    previous = os.getcwd()
    previous_shell = agent.get_session_shell_state().cwd
    invoke(monkeypatch, "--prompt-file", str(prompt), "--system-file", str(system),
           "-C", str(target), "--incognito", "--output-format", "text")
    assert capsys.readouterr().out == "answer\n"
    assert captured == [(prompt.read_text(), "custom system", str(target), str(target))]
    assert os.getcwd() == previous
    assert agent.get_session_shell_state().cwd == previous_shell


@pytest.mark.parametrize("output_format", ["text", "json", "jsonl"])
def test_clean_output_with_real_tool_loop(runtime, monkeypatch, capsys, tmp_path, output_format):
    file = tmp_path / "data.txt"
    file.write_text("file contents", encoding="utf-8")
    calls = iter([
        [TextDelta("Checking the file."), call("read", {"input": str(file)}), StreamDone(None)],
        [TextDelta("final answer"), StreamDone(None)],
    ])
    monkeypatch.setattr(agent, "stream_response", lambda *a, **kw: iter(next(calls)))
    invoke(monkeypatch, "--incognito", "--output-format", output_format, "read it")
    out, err = capsys.readouterr()
    if output_format == "text":
        assert out == "final answer\n"
    else:
        events = [json.loads(line) for line in out.splitlines()]
        result = events[-1]
        assert result["text"] == "final answer"
        assert result["status"] == "success"
        assert result["turns"] == 2
        assert result["exit_code"] == 0
        if output_format == "json":
            assert len(events) == 1
        else:
            assert {"start", "turn_start", "text_delta", "tool_call", "tool_result", "result"} <= {e["type"] for e in events}
    assert "Read" in err and "13 chars" in err


def test_quiet_preserves_answer_and_suppresses_progress(runtime, monkeypatch, capsys):
    invoke(monkeypatch, "--incognito", "--quiet", "hi")
    assert capsys.readouterr() == ("answer\n", "")


def test_verbose_uses_stderr_and_omits_secrets(runtime, monkeypatch, capsys):
    invoke(monkeypatch, "--incognito", "--verbose", "hi")
    out, err = capsys.readouterr()
    assert out == "answer\n"
    assert "Provider: openai" in err and "agent turns: 1" in err
    assert "test-key" not in err


@pytest.mark.parametrize("failure,code,status", [
    (ProviderError("broken provider"), 1, "error"), (KeyboardInterrupt(), 130, "cancelled"),
])
def test_machine_output_reports_failures(runtime, monkeypatch, capsys, failure, code, status):
    def fail(*a, **kw):
        raise failure
    monkeypatch.setattr(agent, "stream_response", fail)
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "--incognito", "--quiet", "--output-format", "json", "hi")
    out, err = capsys.readouterr()
    assert error.value.code == code
    assert json.loads(out)["status"] == status
    assert json.loads(out)["exit_code"] == code
    if status == "error":
        assert "broken provider" in err


def test_noninteractive_setup_never_prompts(runtime, monkeypatch, capsys):
    monkeypatch.setattr("jarv.config.is_setup_complete", lambda *_: False)
    monkeypatch.setattr(cli, "cmd_setup", lambda: pytest.fail("setup prompted"))
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "--non-interactive", "--output-format", "json", "hello")
    assert error.value.code == 3
    assert json.loads(capsys.readouterr().out)["status"] == "input_required"


@pytest.mark.parametrize("tool,args", [
    ("ask_user", {"question": "What next?"}),
    ("run_command", {"command": "echo harmless"}),
])
def test_noninteractive_stops_for_required_input(runtime, monkeypatch, capsys, tool, args):
    monkeypatch.setattr(agent, "stream_response", lambda *a, **kw: iter([call(tool, args), StreamDone(None)]))
    monkeypatch.setattr(safety, "prompt_confirmation", lambda *a, **kw: pytest.fail("approval prompted"))
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "--incognito", "--non-interactive", "--command-safety", "all",
               "--output-format", "json", "do it")
    result = json.loads(capsys.readouterr().out)
    assert error.value.code == 3
    assert result["status"] == "input_required"


def test_max_turns_stops_repeating_tool_loop(runtime, monkeypatch, capsys):
    monkeypatch.setattr(agent, "stream_response", lambda *a, **kw: iter([
        call("run_command", {"command": "must not execute"}), StreamDone(None),
    ]))
    monkeypatch.setattr(agent, "_dispatch_run_command_with_ui", lambda *a, **kw: pytest.fail("disabled tool ran"))
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "--incognito", "--no-tools", "--max-turns", "2", "--output-format", "json", "hello")
    result = json.loads(capsys.readouterr().out)
    assert error.value.code == 1
    assert result["turns"] == 2
    assert result["status"] == "limit"


def test_deadline_cancels_active_stream(runtime, monkeypatch):
    closed = threading.Event()
    def stream(*a, cancellation_token, **kw):
        unregister = cancellation_token.register(closed.set)
        try:
            assert closed.wait(2), "deadline never cancelled the transport"
            cancellation_token.throw_if_cancelled()
            yield StreamDone(None)
        finally:
            unregister()
    monkeypatch.setattr(agent, "stream_response", stream)
    start = time.monotonic()
    result = agent.run_agent("hello", {**runtime, "_run_timeout": 0.05}, client=object(), incognito=True)
    assert time.monotonic() - start < 1
    assert result.status == "limit"
    assert result.cancelled is False
    assert "timeout" in result.error


def test_named_sessions_resume_without_rebinding_terminal(runtime, monkeypatch, capsys):
    monkeypatch.setattr(history, "detect_terminal", lambda: ("terminal", "Terminal"))
    history.save_sessions({"terminals": {"terminal": "old"}, "sessions": {}})
    for prompt in ("first", "second"):
        invoke(monkeypatch, "--session", "job-42", "--output-format", "json", prompt)
        assert json.loads(capsys.readouterr().out)["session_id"] == "job-42"
    data = history.load_sessions()
    assert data["terminals"] == {"terminal": "old"}
    saved = history.load_history(history.history_file_for_session("job-42"))
    assert [x["content"] for x in saved if x.get("role") == "user"] == ["first", "second"]
    assert history._session_override is None


def test_archived_session_is_not_silently_replaced(runtime, monkeypatch, capsys):
    history.save_sessions({"terminals": {}, "sessions": {"old": {"archived": True}}})
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "--session", "old", "--output-format", "json", "hello")
    assert error.value.code == 2
    assert "archived" in json.loads(capsys.readouterr().out)["error"]


def test_jsonl_retry_marks_discarded_stream(runtime, monkeypatch, capsys):
    attempts = 0
    def stream(*a, **kw):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield TextDelta("discard this")
            raise RetryableStreamError("retry")
        yield TextDelta("answer")
        yield StreamDone(None)
    monkeypatch.setattr(agent, "stream_response", stream)
    invoke(monkeypatch, "--incognito", "--output-format", "jsonl", "hello")
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(event["type"] == "retry" for event in events)
    assert events[-1]["text"] == "answer"
    assert events[-1]["turns"] == 1


def test_subagents_share_turn_budget(runtime, monkeypatch):
    control = RunControl(CancellationToken(), max_turns=1)
    control.begin_turn()  # Parent spent the run's only turn.
    node = orchestrator.AgentNode(label="child", task="work", depth=1, parent_label="root", sterile=True)
    monkeypatch.setattr(orchestrator, "stream_response", lambda *a, **kw: pytest.fail("child exceeded budget"))
    with pytest.raises(RunStopped):
        orchestrator.run_subagent_loop(node, orchestrator.ArtifactStore(), object(), {**runtime, "_run_control": control})
    assert control.token.cancelled
    assert control.turns == 1


def test_quiet_tool_run_has_no_diagnostics(runtime, monkeypatch, capsys, tmp_path):
    file = tmp_path / "example.txt"
    file.write_text("hello", encoding="utf-8")
    responses = iter([
        [call("read", {"input": str(file)}), StreamDone(None)],
        [TextDelta("answer"), StreamDone(None)],
    ])
    monkeypatch.setattr(agent, "stream_response", lambda *a, **kw: iter(next(responses)))
    invoke(monkeypatch, "--quiet", "--incognito", "read it")
    assert capsys.readouterr() == ("answer\n", "")


def test_noninteractive_edit_cannot_write_without_approval(runtime, monkeypatch, tmp_path, capsys):
    file = tmp_path / "example.txt"
    file.write_text("before", encoding="utf-8")
    monkeypatch.setattr(agent, "stream_response", lambda *a, **kw: iter([
        call("edit", {"path": str(file), "old_text": "before", "new_text": "after"}), StreamDone(None),
    ]))
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "--incognito", "--non-interactive", "--command-safety", "all",
               "--output-format", "json", "edit it")
    assert error.value.code == 3
    assert json.loads(capsys.readouterr().out)["status"] == "input_required"
    assert file.read_text() == "before"


def test_noninteractive_auditor_policy_is_preserved(runtime, monkeypatch):
    config = {**runtime, "_non_interactive": True, "auditor_auto_approve": True}
    monkeypatch.setattr("jarv.auditor.audit_command", lambda *a, **kw: (True, "safe"))
    monkeypatch.setattr(safety, "prompt_confirmation", lambda *a, **kw: pytest.fail("prompted"))
    assert safety.check_command("rm -rf test", "risky", audit=True, config=config)[0]
    with pytest.raises(RunStopped, match="Manual approval"):
        safety.check_command("rm -rf test", "risky", audit=True,
                             config={**config, "auditor_auto_approve": False})


def test_deadline_interrupts_keyboard_wait():
    from jarv.command_input import read_editable_line
    token = CancellationToken()
    control = RunControl(token, timeout=0.05)
    try:
        with pytest.raises(Exception) as error:
            read_editable_line("Question: ", write=lambda _: None,
                               read_key=lambda: pytest.fail("blocking read"),
                               key_available=lambda: False, cancellation_token=token)
        from jarv.cancellation import TurnCancelled
        assert isinstance(error.value, TurnCancelled)
    finally:
        control.close()


def test_run_deadline_cleans_up_sleeping_command(runtime, monkeypatch):
    import shlex
    executable = sys.executable
    command = ("& '" + executable.replace("'", "''") + "' -c 'import time; time.sleep(30)'"
               if os.name == "nt" else shlex.quote(executable) + " -c 'import time; time.sleep(30)'")
    monkeypatch.setattr(agent, "stream_response", lambda *a, **kw: iter([
        call("run_command", {"command": command}), StreamDone(None),
    ]))
    started = time.monotonic()
    result = agent.run_agent("wait", {**runtime, "_run_timeout": 1, "command_safety": "none"},
                             client=object(), incognito=True)
    assert result.status == "limit"
    assert time.monotonic() - started < 5


@pytest.mark.parametrize("args", [
    ["--output-format", "json"], ["--non-interactive"], ["--quiet"],
])
def test_empty_oneshot_does_not_open_headsup(runtime, monkeypatch, args):
    monkeypatch.setattr(cli, "run_heads_up_mode", lambda *a, **kw: pytest.fail("opened heads-up"))
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, *args)
    assert error.value.code == 2


def test_generic_provider_and_tier_validation():
    for argv in (["-c", "provider=unknown"], ["--provider", "ollama", "--service-tier", "priority"]):
        args = cli._build_parser().parse_args(argv)
        with pytest.raises(ValueError):
            cli._apply_cli_overrides(DEFAULT_CONFIG, args)


def test_prompt_file_help_is_not_command(runtime, monkeypatch, tmp_path, capsys):
    file = tmp_path / "prompt.txt"
    file.write_text("help", encoding="utf-8")
    monkeypatch.setattr("jarv.commands.print_help", lambda **kw: pytest.fail("ran command"))
    invoke(monkeypatch, "--prompt-file", str(file), "--incognito", "--quiet")
    assert capsys.readouterr().out == "answer\n"


def test_late_worker_events_do_not_follow_terminal_result(capsys):
    from jarv.cli_output import CliOutput
    output = CliOutput("jsonl")
    output.event("start")
    output.finish(status="limit", error="limit", exit_code=1)
    output.event("tool_result", agent="late child", output="cancelled")
    assert [json.loads(line)["type"] for line in capsys.readouterr().out.splitlines()] == ["start", "result"]


def test_run_limit_is_shown_in_headsup(runtime, monkeypatch):
    monkeypatch.setattr(agent, "stream_response", lambda *a, **kw: iter([
        call("run_command", {"command": "must not execute"}), StreamDone(None),
    ]))
    ui = Mock()
    result = agent.run_agent("hello", {**runtime, "_max_turns": 1,
                                     "disabled_tools": ["run_command"]},
                             client=object(), incognito=True, heads_up=True, ui=ui)
    assert result.status == "limit"
    ui.show_error.assert_called_once_with("Maximum agent turns reached.")
