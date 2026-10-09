"""Usage diagnostics follow both interactive and one-shot output contracts."""

import copy
import io
import json
import sys
from types import SimpleNamespace

import pytest
from rich.text import Text

from jarv import agent, agent_ui, cli, history, usage
from jarv.cli_output import CliOutput
from jarv.config import DEFAULT_CONFIG
from jarv.provider import ProviderError, StreamDone, TextDelta, ToolCallDone


@pytest.fixture
def usage_runtime(monkeypatch, tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(check_updates=False, project_context=False, audit=False)
    monkeypatch.setattr(cli, "load_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr("jarv.config.is_setup_complete", lambda *_: True)
    monkeypatch.setattr("jarv.provider_auth.resolve_api_key", lambda *_: "test-key")
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(history, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.setattr(history, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(usage, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        agent, "_prepare_client_and_instructions", lambda *a, **kw: (object(), "system")
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=1200,
            cached_input_tokens=200,
            output_tokens=300,
            total_tokens=1500,
        )
    )
    monkeypatch.setattr(
        agent,
        "stream_response",
        lambda *a, **kw: iter([TextDelta("answer"), StreamDone(response)]),
    )
    return config


@pytest.mark.parametrize("output_format", [None, "text", "json", "jsonl"])
@pytest.mark.parametrize("include_session", [False, True])
@pytest.mark.parametrize("quiet", [False, True])
def test_usage_routing_through_real_cli(
    usage_runtime, monkeypatch, capsys, output_format, include_session, quiet
):
    usage_runtime.update(turn_summary=True, turn_summary_session=include_session)
    arguments = ["jarv"]
    if output_format:
        arguments.extend(["--output-format", output_format])
    if quiet:
        arguments.append("--quiet")
    arguments.append("hello")
    monkeypatch.setattr(sys, "argv", arguments)

    cli.main()

    stdout, stderr = capsys.readouterr()
    if output_format in ("json", "jsonl"):
        events = [json.loads(line) for line in stdout.splitlines()]
        assert events[-1]["type"] == "result"
        assert events[-1]["text"] == "answer"
        assert events[-1]["status"] == "success"
        if output_format == "json":
            assert len(events) == 1
        assert "Turn:" not in stdout
    elif output_format or quiet:
        assert stdout == "answer\n"
    else:
        assert "answer" in stdout

    if quiet:
        assert "Turn:" not in stdout
        assert stderr == ""
    else:
        diagnostic = " ".join((stderr if output_format else stdout).split())
        assert diagnostic.count("Turn:") == 1
        assert "1,200 in" in diagnostic
        assert "200 cached" in diagnostic
        assert "300 out" in diagnostic
        assert "1,500 total" in diagnostic
        assert "tok/s" in diagnostic
        assert ("1,500 session tokens" in diagnostic) == include_session
        assert "Usage:" not in diagnostic
        assert "Model:" not in diagnostic


@pytest.mark.parametrize("output_format", ["text", "json", "jsonl"])
@pytest.mark.parametrize("quiet", [False, True])
def test_usage_callback_keeps_protocol_clean_without_diagnostic_context(
    capsys, output_format, quiet
):
    output = CliOutput(output_format, quiet=quiet)
    output.show_usage_line(Text("Turn: 300 out"))
    output.finish(text="answer")

    stdout, stderr = capsys.readouterr()
    assert "Turn:" not in stdout
    assert stderr == ("" if quiet else "Turn: 300 out\n")
    if output_format == "text":
        assert stdout == "answer\n"
    else:
        assert json.loads(stdout)["text"] == "answer"


@pytest.fixture
def capture_output_trace(monkeypatch):
    writes = []

    class RecordingStream(io.StringIO):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def write(self, value):
            writes.append((self.name, value))
            return super().write(value)

    def capture():
        monkeypatch.setattr(sys, "stdout", RecordingStream("stdout"))
        monkeypatch.setattr(sys, "stderr", RecordingStream("stderr"))
        return writes

    return capture


@pytest.mark.parametrize("flags", [["--output-format", "text"], ["--verbose"]])
@pytest.mark.parametrize("answer", ["answer", ""])
def test_text_usage_follows_final_answer(
    usage_runtime, monkeypatch, capture_output_trace, flags, answer
):
    output_trace = capture_output_trace()
    usage_runtime.update(turn_summary=True, turn_summary_session=True)
    monkeypatch.setattr(
        agent, "stream_response",
        lambda *a, **kw: iter([
            *([TextDelta(answer)] if answer else []),
            StreamDone({"usage": {"input_tokens": 100, "output_tokens": 20}}),
        ]),
    )
    monkeypatch.setattr(sys, "argv", ["jarv", *flags, "hello"])

    cli.main()

    combined = "".join(value for _, value in output_trace)
    stdout = "".join(value for stream, value in output_trace if stream == "stdout")
    assert stdout == ("answer\n" if answer else "")
    assert combined.count("Turn:") == 1
    assert "120 session tokens" in " ".join(combined.split())
    if answer:
        assert combined.index(answer) < combined.index("Turn:")


@pytest.mark.parametrize("quiet", [False, True])
@pytest.mark.parametrize("output_format", ["text", "json", "jsonl"])
def test_tool_rounds_have_one_combined_summary_after_completion(
    usage_runtime, monkeypatch, capture_output_trace, quiet, output_format
):
    output_trace = capture_output_trace()
    usage_runtime.update(turn_summary=True, turn_summary_session=True)
    responses = iter([
        [
            ToolCallDone("fc_1", "call_1", "run_command", '{"command":"echo ok"}'),
            StreamDone({"usage": {"input_tokens": 100, "output_tokens": 20}}),
        ],
        [TextDelta("answer"), StreamDone({"usage": {"input_tokens": 140, "output_tokens": 40}})],
    ])
    monkeypatch.setattr(agent, "stream_response", lambda *a, **kw: iter(next(responses)))

    def execute(*_args, **_kwargs):
        diagnostic = "".join(value for stream, value in output_trace if stream == "stderr")
        assert "Turn:" not in diagnostic
        output_trace.append(("tool", "EXECUTED"))
        return "ok"

    monkeypatch.setattr(agent, "_dispatch_run_command_with_ui", execute)
    arguments = ["jarv", "--output-format", output_format]
    if quiet:
        arguments.append("--quiet")
    monkeypatch.setattr(sys, "argv", [*arguments, "run it"])

    cli.main()

    combined = "".join(value for _, value in output_trace)
    stdout = "".join(value for stream, value in output_trace if stream == "stdout")
    assert "Turn:" not in stdout
    if output_format == "text":
        assert stdout == "answer\n"
    else:
        events = [json.loads(line) for line in stdout.splitlines()]
        assert events[-1]["type"] == "result"
        assert events[-1]["text"] == "answer"
        assert events[-1]["status"] == "success"
        assert events[-1]["turns"] == 2
        if output_format == "json":
            assert len(events) == 1
    assert combined.index("EXECUTED") < combined.index("answer")
    if quiet:
        assert "Turn:" not in combined
    else:
        assert combined.count("Turn:") == 1
        assert combined.index("EXECUTED") < combined.index("Turn:")
        if output_format == "text":
            assert combined.index("answer") < combined.index("Turn:")
        else:
            assert combined.index('"type": "result"') < combined.index("Turn:")
        diagnostic = " ".join(combined.split())
        assert "240 in" in diagnostic
        assert "60 out" in diagnostic
        assert "300 total" in diagnostic
        assert "300 session tokens" in diagnostic
        assert "Usage:" not in combined and "Model:" not in combined


@pytest.mark.parametrize("output_format", ["text", "json", "jsonl"])
def test_failed_turn_after_successful_tool_response_has_no_summary(
    usage_runtime, monkeypatch, capsys, output_format
):
    usage_runtime["turn_summary"] = True
    responses = []

    def stream(*_args, **_kwargs):
        responses.append(True)
        if len(responses) > 1:
            raise ProviderError("Request failed")
        yield ToolCallDone("fc_1", "call_1", "run_command", '{"command":"echo ok"}')
        yield StreamDone({"usage": {"input_tokens": 100, "output_tokens": 20}})

    monkeypatch.setattr(agent, "stream_response", stream)
    monkeypatch.setattr(agent, "_dispatch_run_command_with_ui", lambda *a, **kw: "ok")
    monkeypatch.setattr(sys, "argv", ["jarv", "--output-format", output_format, "run it"])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 1
    stdout, stderr = capsys.readouterr()
    assert "Turn:" not in stdout + stderr
    assert "Request failed" in stderr
    if output_format == "text":
        assert stdout == ""
    else:
        events = [json.loads(line) for line in stdout.splitlines()]
        assert events[-1]["type"] == "result"
        assert events[-1]["status"] == "error"
        assert events[-1]["error"] == "Request failed"


def test_error_finish_drains_pending_text_usage_once(capsys):
    output = CliOutput("text")
    output.show_usage_line(Text("Turn: 20 out · 120 session tokens"))
    assert capsys.readouterr() == ("", "")

    output.finish(error="Request failed", status="error")
    output.flush_usage_lines()
    output.finish(error="Request failed", status="error")

    assert capsys.readouterr() == ("", "Turn: 20 out · 120 session tokens\nRequest failed\n")


@pytest.mark.parametrize("output_format", ["json", "jsonl"])
def test_json_usage_diagnostics_follow_final_result(capture_output_trace, output_format):
    writes = capture_output_trace()
    output = CliOutput(output_format)
    output.show_usage_line(Text("Turn: 20 out"))
    assert writes == []

    output.finish(text="answer")

    stdout = "".join(value for stream, value in writes if stream == "stdout")
    stderr = "".join(value for stream, value in writes if stream == "stderr")
    assert json.loads(stdout)["text"] == "answer"
    assert stderr == "Turn: 20 out\n"
    assert writes[0][0] == "stdout"


def test_turn_summary_helper_routes_measured_rate_to_ui():
    lines = []
    ui = SimpleNamespace(show_usage_line=lines.append)
    response = {"usage": {
        "input_tokens": 1200, "completion_tokens": 300, "completion_time": 2.0,
    }}
    agent_ui._print_turn_summary_if_enabled(
        {"turn_summary": True, "provider": "groq"},
        response,
        model="test-model",
        elapsed_seconds=30.0,
        ui=ui,
    )

    assert len(lines) == 1
    assert "1,200 in" in lines[0].plain
    assert "300 out" in lines[0].plain
    assert "150.0 tok/s (server)" in lines[0].plain


def test_legacy_usage_toggles_produce_one_combined_summary():
    lines = []
    config = {"print_usage_after_model": True, "print_usage_after_agent": True}
    agent_ui._print_turn_summary_if_enabled(
        config,
        {"usage": {"input_tokens": 1200, "output_tokens": 300}},
        model="test-model",
        elapsed_seconds=2.0,
        ui=SimpleNamespace(show_usage_line=lines.append),
    )

    assert len(lines) == 1
    assert lines[0].plain.count("Turn:") == 1
    assert "300 out" in lines[0].plain
    assert "tok/s unavailable" in lines[0].plain
    assert "session usage unavailable" in lines[0].plain
    assert "session cost unavailable" in lines[0].plain
    assert config == {"print_usage_after_model": True, "print_usage_after_agent": True}


@pytest.mark.parametrize("session,cost,incognito", [
    (False, False, False), (True, False, False),
    (False, True, False), (True, True, True),
])
def test_summary_loads_saved_usage_only_for_requested_session_fields(
    monkeypatch, tmp_path, session, cost, incognito
):
    reads = []
    lines = []
    usage_path = None if incognito else tmp_path / "usage.json"

    def load(path, session_id, *, warn):
        reads.append((path, session_id, warn))
        return {"totals": {"total_tokens": 2500, "provider_cost_usd": 0.5}}

    monkeypatch.setattr(agent_ui, "load_usage", load)
    agent_ui._print_turn_summary_if_enabled(
        {"turn_summary": True, "turn_summary_session": session, "turn_summary_cost": cost},
        {"usage": {"input_tokens": 1200, "output_tokens": 300}},
        model="test-model",
        elapsed_seconds=2.0,
        usage_path=usage_path,
        session_id="test-session",
        ui=SimpleNamespace(show_usage_line=lines.append),
    )

    assert reads == ([(usage_path, "test-session", False)] if (session or cost) and not incognito else [])
    assert len(lines) == 1
    if session:
        assert ("session usage unavailable" if incognito else "2,500 session tokens") in lines[0].plain
    if cost:
        assert ("session cost unavailable" if incognito else "session cost $0.50") in lines[0].plain


@pytest.mark.parametrize("config", [{}, {"turn_summary": True, "_quiet": True}])
def test_turn_summary_helper_is_silent_by_default_and_when_quiet(config, capsys):
    agent_ui._print_turn_summary_if_enabled(
        config,
        {"usage": {"input_tokens": 1200, "output_tokens": 300}},
        model="test-model",
        elapsed_seconds=2.0,
    )
    assert capsys.readouterr() == ("", "")
