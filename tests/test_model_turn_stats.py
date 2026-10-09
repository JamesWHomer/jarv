"""Exercise selected turn-summary fields across real tool rounds and both UIs."""

import io
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from rich.console import Console

from jarv import agent, agent_ui, turn_loop, usage
from jarv.config import DEFAULT_CONFIG
from jarv.headsup import HeadsupAgentUI
from jarv.history import SessionContext
from jarv.provider import (
    ProviderError, ReasoningStarted, RetryableStreamError,
    StreamDone, TextDelta, ToolCallDone,
)


SUMMARY_FIELDS = ("tokens", "cache", "reasoning", "speed", "time", "session", "cost")


def summary_settings(*fields):
    return {
        **DEFAULT_CONFIG,
        "turn_summary": True,
        **{f"turn_summary_{field}": field in fields for field in SUMMARY_FIELDS},
    }


@pytest.fixture
def summary_output(monkeypatch, headsup_app_factory):
    def create(heads_up):
        app = headsup_app_factory()
        output = io.StringIO()
        console = Console(file=output, width=240, color_system=None)
        monkeypatch.setattr(agent, "console", console)
        monkeypatch.setattr(agent_ui, "console", console)
        monkeypatch.setattr(agent.sys, "stdout", output)

        def lines():
            if heads_up:
                return [entry.renderable.plain for entry in app.entries if entry.kind == "usage"]
            return [line for line in output.getvalue().splitlines() if line.startswith("Turn:")]

        return SimpleNamespace(
            app=app,
            output=output,
            ui=HeadsupAgentUI(app) if heads_up else None,
            lines=lines,
        )

    return create


@pytest.mark.parametrize("heads_up", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_full_turn_has_one_combined_summary_after_all_tools(
    monkeypatch, summary_output, heads_up, enabled,
):
    display = summary_output(heads_up)
    stream_calls = []
    now = [0.0]
    monkeypatch.setattr(turn_loop, "perf_counter", lambda: now[0])

    def stream(*_args, **_kwargs):
        assert display.lines() == []
        stream_calls.append(True)
        if len(stream_calls) == 1:
            yield ToolCallDone("fc_1", "call_1", "run_command", '{"command":"echo ok"}')
            counts = {
                "input_tokens": 100, "completion_tokens": 20,
                "total_tokens": 120, "completion_time": 2.0,
                "cached_input_tokens": 5, "reasoning_output_tokens": 1,
            }
            now[0] += 3.0
        elif len(stream_calls) == 2:
            yield TextDelta("One more check.")
            yield ToolCallDone("fc_2", "call_2", "run_command", '{"command":"echo again"}')
            counts = {
                "input_tokens": 140, "completion_tokens": 40,
                "total_tokens": 180, "completion_time": 4.0,
                "cached_input_tokens": 10, "reasoning_output_tokens": 2,
            }
            now[0] += 5.0
        else:
            yield TextDelta("Finished.")
            counts = {
                "input_tokens": 160, "completion_tokens": 60,
                "total_tokens": 220, "completion_time": 3.0,
                "cached_input_tokens": 20, "reasoning_output_tokens": 3,
            }
            now[0] += 4.0
        yield StreamDone({"usage": counts})

    def execute(*_args, **_kwargs):
        assert display.lines() == []
        now[0] += 100.0
        return "ok"

    monkeypatch.setattr(agent, "stream_response", stream)
    monkeypatch.setattr(agent, "_dispatch_run_command_with_ui", execute)
    result = agent.run_agent(
        "Run it",
        {**DEFAULT_CONFIG, "provider": "groq", "turn_summary": enabled},
        client=object(), incognito=True, heads_up=heads_up, ui=display.ui,
    )

    assert result.error is None
    assert result.turns == 3
    lines = display.lines()
    assert len(lines) == int(enabled)
    if enabled:
        assert "400 in" in lines[0] and "120 out" in lines[0] and "520 total" in lines[0]
        assert "35 cached" in lines[0] and "6 reasoning" in lines[0]
        assert "20.0 tok/s" in lines[0] and "final response" in lines[0]
        assert "12.00s" in lines[0]
        if heads_up:
            assert display.app.entries[-2].kind == "assistant"
            assert display.app.entries[-1].kind == "usage"
        else:
            assert display.output.getvalue().index("Finished.") < display.output.getvalue().index(lines[0])


@pytest.mark.parametrize("heads_up", [False, True])
@pytest.mark.parametrize("selected", [("tokens",), ()])
def test_selected_fields_control_summary_and_all_off_is_silent(
    monkeypatch, summary_output, heads_up, selected,
):
    display = summary_output(heads_up)
    response = {"usage": {
        "input_tokens": 100,
        "cached_input_tokens": 30,
        "output_tokens": 20,
        "reasoning_output_tokens": 5,
        "total_tokens": 120,
    }}
    monkeypatch.setattr(
        agent, "stream_response",
        lambda *_args, **_kwargs: iter([TextDelta("Finished."), StreamDone(response)]),
    )
    result = agent.run_agent(
        "Hello", summary_settings(*selected),
        client=object(), incognito=True, heads_up=heads_up, ui=display.ui,
    )

    assert result.error is None
    assert display.lines() == (["Turn: 100 in · 20 out · 120 total"] if selected else [])


@pytest.mark.parametrize("heads_up", [False, True])
def test_incognito_session_field_is_unavailable_without_loading_saved_usage(
    monkeypatch, summary_output, heads_up,
):
    display = summary_output(heads_up)
    monkeypatch.setattr(
        agent, "stream_response",
        lambda *_args, **_kwargs: iter([
            TextDelta("Finished."),
            StreamDone({"usage": {"input_tokens": 100, "output_tokens": 20}}),
        ]),
    )

    def unexpected_usage_load(*_args, **_kwargs):
        pytest.fail("Incognito summaries must not load persisted session usage")

    monkeypatch.setattr(agent_ui, "load_usage", unexpected_usage_load)
    result = agent.run_agent(
        "Hello", summary_settings("session"),
        client=object(), incognito=True, heads_up=heads_up, ui=display.ui,
    )

    assert result.error is None
    assert display.lines() == ["Turn: session usage unavailable"]


@pytest.mark.parametrize("heads_up", [False, True])
def test_final_session_totals_include_all_model_responses(
    monkeypatch, tmp_path, summary_output, heads_up,
):
    display = summary_output(heads_up)
    context = SessionContext(
        session_id="summary-session", session_label="Summary test",
        history_file=tmp_path / "history.json",
        now=datetime(2026, 5, 21, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(agent, "prepare_session_context", lambda **_kwargs: context)
    monkeypatch.setattr(usage, "CONFIG_DIR", tmp_path)
    stream_calls = []

    def stream(*_args, **_kwargs):
        stream_calls.append(True)
        if len(stream_calls) == 1:
            yield ToolCallDone("fc_1", "call_1", "run_command", '{"command":"echo ok"}')
            counts = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        else:
            yield TextDelta("Finished.")
            counts = {"input_tokens": 140, "output_tokens": 40, "total_tokens": 180}
        yield StreamDone({"usage": counts})

    def execute(*_args, **_kwargs):
        assert display.lines() == []
        return "ok"

    monkeypatch.setattr(agent, "stream_response", stream)
    monkeypatch.setattr(agent, "_dispatch_run_command_with_ui", execute)
    result = agent.run_agent(
        "Run it", summary_settings("session"),
        client=object(), heads_up=heads_up, ui=display.ui,
    )

    assert result.error is None
    assert result.turns == 2
    assert display.lines() == ["Turn: 300 session tokens"]
    saved = usage.load_usage(usage.usage_file_for(context.history_file), context.session_id)
    assert saved["totals"]["total_tokens"] == 300


def test_retried_response_prints_only_successful_model_stats(monkeypatch, headsup_app_factory):
    app = headsup_app_factory()
    attempts = []
    now = [0.0]
    monkeypatch.setattr(turn_loop, "perf_counter", lambda: now[0])

    def stream(*_args, **_kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            now[0] = 50.0
            yield TextDelta("Partial")
            now[0] = 100.0
            raise RetryableStreamError("Disconnected")
        now[0] = 102.0
        yield TextDelta("Reco")
        now[0] = 104.0
        yield TextDelta("very")
        now[0] = 200.0
        yield StreamDone({"usage": {"input_tokens": 100, "output_tokens": 40}})

    monkeypatch.setattr(agent, "stream_response", stream)
    result = agent.run_agent(
        "Hello", {**DEFAULT_CONFIG, "turn_summary": True},
        client=object(), incognito=True, heads_up=True, ui=HeadsupAgentUI(app),
    )

    assert result.error is None
    lines = [entry.renderable.plain for entry in app.entries if entry.kind == "usage"]
    assert len(lines) == 1
    assert "~10.0 tok/s (stream)" in lines[0]


@pytest.mark.parametrize("heads_up", [False, True])
def test_stream_speed_uses_chunk_arrival_times_and_excludes_reasoning_and_metadata_wait(
    monkeypatch, summary_output, heads_up,
):
    display = summary_output(heads_up)
    now = [0.0]
    monkeypatch.setattr(turn_loop, "perf_counter", lambda: now[0])

    def stream(*_args, **_kwargs):
        now[0] = 1.0
        yield ReasoningStarted("reasoning_1")
        # Transport timestamps precede renderer handling. Neither the long
        # reasoning wait nor delayed handling belongs in text generation time.
        now[0] = 70.0
        yield TextDelta("Hello ", received_at=60.0)
        now[0] = 80.0
        yield TextDelta("there!", received_at=60.2)
        now[0] = 90.0
        yield StreamDone({"usage": {
            "input_tokens": 100,
            "output_tokens": 120,
            "reasoning_output_tokens": 100,
        }})

    monkeypatch.setattr(agent, "stream_response", stream)
    result = agent.run_agent(
        "Say hello", summary_settings("tokens", "reasoning", "speed", "time"),
        client=object(), incognito=True, heads_up=heads_up, ui=display.ui,
    )

    assert result.error is None
    assert result.text == "Hello there!"
    assert len(display.lines()) == 1
    summary = display.lines()[0]
    # 20 visible output tokens, half in the initial chunk: 10 / 0.2 seconds.
    assert "~50.0 tok/s (stream)" in summary
    assert "100 reasoning" in summary
    assert "90.00s" in summary


def test_failed_response_does_not_print_completed_model_stats(monkeypatch, headsup_app_factory):
    app = headsup_app_factory()

    def stream(*_args, **_kwargs):
        yield TextDelta("Partial")
        raise ProviderError("Disconnected")

    monkeypatch.setattr(agent, "stream_response", stream)
    result = agent.run_agent(
        "Hello", {**DEFAULT_CONFIG, "turn_summary": True},
        client=object(), incognito=True, heads_up=True, ui=HeadsupAgentUI(app),
    )

    assert result.error is not None
    assert not any(entry.kind == "usage" for entry in app.entries)


@pytest.mark.parametrize("heads_up", [False, True])
def test_failed_final_response_does_not_leave_intermediate_summaries(
    monkeypatch, summary_output, heads_up,
):
    display = summary_output(heads_up)
    attempts = []

    def stream(*_args, **_kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            yield ToolCallDone("fc_1", "call_1", "run_command", '{"command":"echo ok"}')
            yield StreamDone({"usage": {"input_tokens": 100, "output_tokens": 20}})
        else:
            yield TextDelta("Partial")
            raise ProviderError("Disconnected")

    def execute(*_args, **_kwargs):
        assert display.lines() == []
        return "ok"

    monkeypatch.setattr(agent, "stream_response", stream)
    monkeypatch.setattr(agent, "_dispatch_run_command_with_ui", execute)
    result = agent.run_agent(
        "Run it", summary_settings("tokens"),
        client=object(), incognito=True, heads_up=heads_up, ui=display.ui,
    )

    assert result.error == "Disconnected"
    assert result.turns == 2
    assert display.lines() == []


@pytest.mark.parametrize("heads_up", [False, True])
def test_new_user_turn_starts_fresh_summary_totals(monkeypatch, summary_output, heads_up):
    display = summary_output(heads_up)
    responses = iter([
        [
            ToolCallDone("fc_1", "call_1", "run_command", '{"command":"echo ok"}'),
            StreamDone({"usage": {"input_tokens": 100, "output_tokens": 20}}),
        ],
        [TextDelta("Finished."), StreamDone({"usage": {"input_tokens": 140, "output_tokens": 40}})],
        [TextDelta("Next answer."), StreamDone({"usage": {"input_tokens": 200, "output_tokens": 10}})],
    ])
    monkeypatch.setattr(agent, "stream_response", lambda *_args, **_kwargs: iter(next(responses)))
    monkeypatch.setattr(agent, "_dispatch_run_command_with_ui", lambda *_args, **_kwargs: "ok")

    for query in ("Run it", "Next question"):
        result = agent.run_agent(
            query, summary_settings("tokens"),
            client=object(), incognito=True, heads_up=heads_up, ui=display.ui,
        )
        assert result.error is None

    assert display.lines() == [
        "Turn: 240 in · 60 out · 300 total",
        "Turn: 200 in · 10 out · 210 total",
    ]
