import io
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from rich.console import Console

from jarv import agent, display, orchestrator, tool_progress
from jarv.artifacts import ArtifactStore
from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.cli_output import CliOutput
from jarv.config import DEFAULT_CONFIG
from jarv.headsup import HeadsupAgentUI, HeadsupApp
from jarv.orchestrator import AgentNode
from jarv.retained_outputs import RetainedOutputStore
from jarv.tool_outputs import tool_outcome, with_tool_outcome


def call(ident, name="read", arguments=None):
    return SimpleNamespace(
        id=ident, call_id=ident, name=name,
        arguments=arguments if arguments is not None else json.dumps(
            {"input" if name == "read" else "query": ident}
        ),
    )


def render(card):
    output = io.StringIO()
    Console(file=output, width=100, color_system=None).print(card)
    return output.getvalue()


class ProbeUI:
    def __init__(self):
        self.cards = {}
        self.events = []
        self.changed = threading.Condition()

    def show_tool_card(self, card):
        with self.changed:
            self.cards[card.item["call_id"]] = card
            self.events.append((card.item["call_id"], card.state, threading.get_ident()))
            self.changed.notify_all()

    def wait(self, ident, state):
        with self.changed:
            assert self.changed.wait_for(
                lambda: ident in self.cards and self.cards[ident].state == state, timeout=3,
            ), self.events


def execution(calls, ui, *, config=None, token=None):
    config = {**DEFAULT_CONFIG, "tool_call_display": "fullscreen", **(config or {})}
    token = token or CancellationToken()
    root, store = AgentNode("root", 0, None, "test", False), ArtifactStore()
    retained = RetainedOutputStore()
    hooks = agent._build_tool_hooks(
        config=config, history=[], usage_path=None, session_id="test",
        cancellation_token=token, retained_store=retained, ui=ui,
        interactive_help={}, root_node=root, artifact_store=store, client=None,
    )
    recorded, started, checkpointed = [], [], []
    hooks.on_tool_start = lambda item: started.append(item.call_id)
    hooks.on_tool_recorded = lambda item: checkpointed.append(item.call_id)

    def run():
        return orchestrator.execute_tool_calls(
            calls, node=root, store=store, client=None, config=config, hooks=hooks,
            cancellation_token=token, retained_store=retained,
            append_tool_result=lambda item, output: recorded.append((item.call_id, output)),
        )

    return SimpleNamespace(run=run, token=token, recorded=recorded,
                           started=started, checkpointed=checkpointed)


@contextmanager
def background(job, release):
    errors = []

    def run():
        try:
            job.run()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        yield thread, errors
    finally:
        release.set()
        thread.join(3)
        assert not thread.is_alive()


@pytest.mark.parametrize("name", ["read", "web_search"])
def test_single_call_is_visible_before_io_completes(monkeypatch, name):
    release, entered = threading.Event(), threading.Event()
    ui = ProbeUI()

    def blocked(*_args, **_kwargs):
        entered.set()
        assert release.wait(3)
        return with_tool_outcome("result", "success")

    monkeypatch.setattr(orchestrator, "dispatch_tool", blocked)
    job = execution([call("https://example.test", name)], ui)
    with background(job, release) as (thread, errors):
        assert entered.wait(3)
        ui.wait("https://example.test", "running")
        card = ui.cards["https://example.test"]
        with monkeypatch.context() as clock:
            clock.setattr(tool_progress.time, "perf_counter", lambda: card.started_at + 2)
            assert "running 2s" in render(card)
        assert "https://example.test" in render(card)
        assert job.recorded == []
        assert job.started == ["https://example.test"]
        release.set()
        ui.wait("https://example.test", "finished")
    assert errors == []
    assert job.checkpointed == ["https://example.test"]
    assert "done" in render(ui.cards["https://example.test"])
    assert {ident for _, _, ident in ui.events} == {thread.ident}


def test_fast_search_finishes_while_slow_read_runs_and_input_order_is_preserved(monkeypatch):
    release = threading.Event()
    ui = ProbeUI()

    def read(*_args, **_kwargs):
        assert release.wait(3)
        return with_tool_outcome("read output", "success")

    monkeypatch.setattr(orchestrator, "dispatch_read_tool", read)
    monkeypatch.setattr(orchestrator, "dispatch_web_tool",
                        lambda *_a, **_k: with_tool_outcome("search output", "success"))
    job = execution([call("slow"), call("fast", "web_search")], ui)
    with background(job, release) as (thread, errors):
        ui.wait("slow", "running")
        ui.wait("fast", "finished")
        assert list(ui.cards) == ["slow", "fast"]
        assert ui.cards["slow"].state == "running"
        assert ui.cards["fast"].output == "search output"
        assert job.recorded == []
    assert errors == []
    assert [ident for ident, _ in job.recorded] == ["slow", "fast"]
    assert orchestrator.WEB_SEARCH_READ_NUDGE in job.recorded[1][1]
    assert job.checkpointed == ["slow", "fast"]
    assert {ident for _, _, ident in ui.events} == {thread.ident}


def test_queued_calls_and_cancellation_do_not_receive_late_worker_updates(monkeypatch):
    release = threading.Event()
    ui = ProbeUI()
    workers_finished = [threading.Event() for _ in range(8)]

    def read(args, **_kwargs):
        try:
            assert release.wait(3)
            return with_tool_outcome("late result", "success")
        finally:
            workers_finished[int(args["input"])].set()

    monkeypatch.setattr(orchestrator, "dispatch_read_tool", read)
    job = execution([call(str(i)) for i in range(9)], ui)
    with background(job, release) as (thread, errors):
        for i in range(8):
            ui.wait(str(i), "running")
        assert ui.cards["8"].state == "queued"
        assert "queued" in render(ui.cards["8"])
        assert "8" not in job.started
        job.token.cancel()
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], TurnCancelled)
        assert all(card.finished for card in ui.cards.values())
        assert all("cancelled" in render(card) for card in ui.cards.values())
        before = list(ui.events)
    # Executor workers can unwind after the coordinator returns; their late
    # results must not revive the finished UI or start the queued ninth call.
    for finished in workers_finished:
        assert finished.wait(3)
    assert ui.events == before


@pytest.mark.parametrize("kind", ["failed", "invalid", "disabled", "exception", "cancelled"])
def test_failures_finalize_cards(monkeypatch, kind):
    ui = ProbeUI()
    config = {"disabled_tools": ["read"]} if kind == "disabled" else {}
    item = call("failed", arguments="{" if kind == "invalid" else None)

    def fail(*_args, **_kwargs):
        if kind == "exception":
            raise RuntimeError("broken reader")
        if kind == "cancelled":
            raise TurnCancelled()
        return with_tool_outcome("[read error: unavailable]", "failed")

    monkeypatch.setattr(orchestrator, "dispatch_read_tool", fail)
    job = execution([item], ui, config=config)
    if kind in {"exception", "cancelled"}:
        with pytest.raises(RuntimeError if kind == "exception" else TurnCancelled):
            job.run()
    else:
        job.run()
    card = ui.cards["failed"]
    assert card.finished
    assert tool_outcome(card.output).status in {"failed", "denied", "cancelled"}
    assert "running" not in render(card)
    if kind in {"invalid", "disabled"}:
        assert job.started == []
        assert [state for _, state, _ in ui.events] == ["finished"]


def test_headsup_replaces_independent_cards_and_stops_animation(monkeypatch):
    ready = threading.Event()
    ready.set()
    app = HeadsupApp(
        {**DEFAULT_CONFIG, "headsup_intro_logo": False, "headsup_intro_stars": False},
        client=object(), args=None, agent_loader=({"module": SimpleNamespace()}, ready),
        handle_slash=lambda *args: None, maybe_command=lambda *args: None,
        render_console=Console(file=io.StringIO(), width=100),
    )
    ui = HeadsupAgentUI(app)
    progress = tool_progress.ParallelToolDisplay({**DEFAULT_CONFIG, "tool_call_display": "fullscreen"}, ui)
    first, second = call("first"), call("second")
    for item in (first, second):
        progress.update(item, {"input": item.call_id}, "queued", "")
        progress.update(item, {"input": item.call_id}, "running", "")
    slots = dict(app._live_tool_index)
    assert len(slots) == 2
    assert ui.has_active_animation()
    assert ui._refresh_wait_statuses()
    progress.update(second, {"input": "second"}, "finished", with_tool_outcome("ok", "success"))
    assert list(app._live_tool_index) == ["tool:first"]
    assert app.entries[slots["tool:second"]].renderable.finished
    assert ui.has_active_animation()
    progress.finish(TurnCancelled())
    assert not app._live_tool_index
    assert not ui.has_active_animation()
    cards = [entry.renderable for entry in app.entries if entry.kind == "tool"]
    assert [card.item["call_id"] for card in cards] == ["first", "second"]
    assert "cancelled" in render(cards[0])
    assert "done" in render(cards[1])


@pytest.mark.parametrize("mode", ["print", "fullscreen"])
@pytest.mark.parametrize("cancel", [False, True])
def test_terminal_uses_one_live_display_and_cleans_up(monkeypatch, mode, cancel):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=100, color_system=None)
    monkeypatch.setattr(tool_progress, "console", console)
    instances = []
    real_live = tool_progress.Live

    def live(*args, **kwargs):
        result = real_live(*args, **{**kwargs, "auto_refresh": False})
        instances.append(result)
        return result

    monkeypatch.setattr(tool_progress, "Live", live)
    progress = tool_progress.ParallelToolDisplay({**DEFAULT_CONFIG, "tool_call_display": mode})
    initial_depth = display._live_display_depth
    try:
        for item in (call("first"), call("second", "web_search")):
            args = json.loads(item.arguments)
            progress.update(item, args, "queued", "")
            progress.update(item, args, "running", "")
        assert len(instances) == 1
        assert display._live_display_depth == initial_depth + 1
        assert "running 0s" in output.getvalue()
        if not cancel:
            for item in (call("second", "web_search"), call("first")):
                progress.update(item, json.loads(item.arguments), "finished", with_tool_outcome("ok", "success"))
    finally:
        progress.finish(TurnCancelled() if cancel else None)
    assert display._live_display_depth == initial_depth
    assert not instances[0].is_started
    final_frame = render(instances[0].renderable)
    assert final_frame.count("first") == final_frame.count("second") == 1
    assert "running" not in final_frame and "queued" not in final_frame
    if cancel:
        assert "cancelled" in final_frame


@pytest.mark.parametrize("protocol", [False, True])
@pytest.mark.parametrize("quiet", [False, True])
def test_append_only_output_stays_ordered_and_protocol_stdout_stays_clean(monkeypatch, protocol, quiet):
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)
    console = Console(file=stderr if protocol else stdout, width=100, color_system=None)
    monkeypatch.setattr(tool_progress, "console", console)
    monkeypatch.setattr(display, "console", console)
    ui = CliOutput("jsonl", quiet=quiet) if protocol else None
    progress = tool_progress.ParallelToolDisplay({**DEFAULT_CONFIG, "_quiet": quiet}, ui)
    items = [call("first"), call("second")]
    for item in items:
        progress.update(item, {"input": item.call_id}, "queued", "")
        progress.update(item, {"input": item.call_id}, "running", "")
    for item in reversed(items):
        progress.update(item, {"input": item.call_id}, "finished", with_tool_outcome("ok", "success"))
    assert stdout.getvalue() == stderr.getvalue() == ""
    progress.finish()
    text = stderr.getvalue() if protocol else stdout.getvalue()
    if quiet:
        assert text == ""
    else:
        assert text.index("first") < text.index("second")
        assert text.count("first") == text.count("second") == 1
        assert "running" not in text and "queued" not in text and "\x1b" not in text
    if protocol:
        ui.finish(text="answer")
        assert json.loads(stdout.getvalue())["text"] == "answer"
