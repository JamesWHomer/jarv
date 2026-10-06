"""Smoke tests that the headless heads-up harness drives a real loop.

These keep the harness (and the symbols it patches) honest against the live
``HeadsupApp`` source.
"""

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
from rich.cells import cell_len

from conftest import wait_for
import headsup_harness
from headsup_harness import HeadsupHarness, strip_ansi
from jarv import headsup


def _fake_agent(query, config, client, *, ui=None, **kwargs):
    if ui is not None and hasattr(ui, "finish_assistant_message"):
        if hasattr(ui, "begin_assistant_message"):
            ui.begin_assistant_message()
        ui.finish_assistant_message(f"reply to {query}")
    return SimpleNamespace(cancelled=False, error=None)


def test_harness_paints_initial_frame():
    with HeadsupHarness(width=70, height=14, run_agent=_fake_agent) as h:
        st = h.state()
        assert st["running"] is True
        assert st["refreshes"] >= 1
        assert h.plain_frame  # a frame was captured


def test_harness_echoes_typed_text_into_frame():
    with HeadsupHarness(width=70, height=14, run_agent=_fake_agent) as h:
        h.feed_text("hello harness")
        assert h.wait_idle()
        assert "hello harness" in h.plain_frame
        assert h.prompt_buffer == "hello harness"


def test_harness_submits_query_and_renders_reply():
    with HeadsupHarness(width=72, height=16, run_agent=_fake_agent) as h:
        h.feed_text("ping")
        h.feed_key("enter")
        assert h.wait_idle()
        assert "ping" in h.transcript
        assert "reply to ping" in h.transcript
        # Submitting clears the prompt buffer.
        assert h.prompt_buffer == ""


def test_harness_resize_repaints_at_the_new_dimensions():
    def frame_size(harness):
        lines = harness.plain_frame.splitlines()
        return max(map(cell_len, lines), default=0), len(lines)

    with HeadsupHarness(width=70, height=14, run_agent=_fake_agent) as h:
        assert frame_size(h) == (70, 14)
        initial_frame = h.frame
        h.resize(110, 30)
        assert wait_for(lambda: frame_size(h) == (110, 30), timeout=3.0)
        assert h.frame != initial_frame
        # Shrinking must also discard rows and columns from the larger frame.
        h.resize(50, 10)
        assert wait_for(lambda: frame_size(h) == (50, 10), timeout=3.0)


def test_harness_frame_carries_stale_edge_erase():
    with HeadsupHarness(width=80, height=16, run_agent=_fake_agent) as h:
        h.feed_text("content")
        assert h.wait_idle()
        # Heads-up routes its frame through EraseTrailingColumns, so the captured
        # terminal frame includes the erase-to-end-of-line control.
        assert "\x1b[0K" in h.frame


def test_strip_ansi_removes_escape_sequences():
    assert strip_ansi("\x1b[31mred\x1b[0m\x1b[0K") == "red"


@pytest.mark.parametrize("stage", ["startup", "initial_render", "repaint"])
def test_harness_propagates_background_errors_and_restores_patches(monkeypatch, stage):
    original_live = headsup.Live
    error = RuntimeError(f"broken {stage}")

    def fail(*_args, **_kwargs):
        raise error

    if stage == "startup":
        monkeypatch.setattr(headsup.HeadsupApp, "run", fail)
    elif stage == "initial_render":
        monkeypatch.setattr(headsup.HeadsupApp, "render", fail)

    harness = HeadsupHarness(run_agent=_fake_agent)
    with pytest.raises(RuntimeError, match=f"broken {stage}") as raised:
        with harness:
            assert stage == "repaint", "startup errors must fail before entering the body"
            harness.live._get_renderable = fail
            harness.app.invalidate()
            assert harness._loop_finished.wait(3.0), "repaint did not reach the renderer"
            # The exception must surface on context exit even if no wait method
            # was called after the background loop failed.
    assert raised.value is error
    assert harness._thread is None
    assert harness._stack is None
    assert headsup.Live is original_live


def test_harness_wait_idle_propagates_background_error(monkeypatch):
    harness = HeadsupHarness(run_agent=_fake_agent)
    with pytest.raises(RuntimeError, match="broken input"):
        with harness:
            def fail(*_args):
                raise RuntimeError("broken input")

            monkeypatch.setattr(harness.app, "on_key", fail)
            harness.feed_text("trigger failure")
            assert harness._loop_finished.wait(3.0)
            with pytest.raises(RuntimeError, match="broken input"):
                harness.wait_idle()
    assert harness._thread is None
    assert harness._stack is None


def test_harness_requires_initial_paint_and_cleans_up_early_exit(monkeypatch):
    original_live = headsup.Live
    monkeypatch.setattr(headsup.HeadsupApp, "run", lambda _app: None)
    harness = HeadsupHarness()
    with pytest.raises(AssertionError, match="initial frame"):
        with harness:
            pytest.fail("harness entered without painting")
    assert harness._thread is None
    assert harness._stack is None
    assert headsup.Live is original_live


def test_harness_restores_partial_setup_when_terminal_context_fails(monkeypatch):
    original_live = headsup.Live

    @contextmanager
    def broken_terminal():
        raise RuntimeError("terminal setup failed")
        yield

    monkeypatch.setattr(headsup_harness, "neutral_terminal_modes", broken_terminal)
    harness = HeadsupHarness()
    with pytest.raises(RuntimeError, match="terminal setup failed"):
        harness.__enter__()
    assert harness._thread is None
    assert harness._stack is None
    assert headsup.Live is original_live


def test_harness_keeps_patches_until_stalled_thread_has_stopped():
    # Simulate a stalled thread without creating an unkillable real worker.
    class StalledThread:
        alive = True

        def join(self, timeout):
            assert timeout > 0

        def is_alive(self):
            return self.alive

    original_live = headsup.Live
    replacement = object()
    harness = HeadsupHarness()
    thread = StalledThread()
    harness._thread = thread
    stack = ExitStack()
    stack.enter_context(mock.patch.object(headsup, "Live", replacement))
    harness._stack = stack
    try:
        with pytest.raises(AssertionError, match="did not stop"):
            harness.stop()
        assert harness._thread is thread
        assert harness._stack is stack
        assert headsup.Live is replacement
    finally:
        thread.alive = False
        harness.stop()
    assert harness._thread is None
    assert harness._stack is None
    assert headsup.Live is original_live


def test_harness_safety_confirmation_end_to_end():
    """A risky command inside a real heads-up turn prompts via the app.

    The gate runs on the agent worker thread while the loop owns the alt
    screen — exactly the topology that used to crash (nested Live) or hang
    (console.input racing the loop's key reader).
    """
    from jarv.safety import check_command

    results = {}

    def risky_agent(query, config, client, *, ui=None, **kwargs):
        results["gate"] = check_command(
            "taskkill /f /im notepad.exe", "risky", audit=False
        )
        if ui is not None and hasattr(ui, "finish_assistant_message"):
            ui.finish_assistant_message("done")
        return SimpleNamespace(cancelled=False, error=None)

    with HeadsupHarness(width=90, height=24, run_agent=risky_agent) as h:
        h.feed_text("kill notepad")
        h.feed_key("enter")
        assert wait_for(lambda: h.answer_prompt is not None, timeout=3.0)
        assert "Allow this command?" in h.answer_prompt
        assert "taskkill" in h.transcript

        h.feed_text("y")
        h.feed_key("enter")
        assert h.wait_idle(timeout=3.0)

    assert results["gate"] == (True, "")
    assert "approved" in h.transcript


def test_large_approval_script_remains_scrollable_while_prompting():
    from jarv.safety import check_command

    command = "\n".join(["rm -rf cache"] + [f"echo script-line-{n:03}" for n in range(80)])
    results = {}

    def agent(query, config, client, *, ui=None, **kwargs):
        results["gate"] = check_command(command, "all", audit=False)
        return SimpleNamespace(cancelled=False, error=None)

    with HeadsupHarness(width=90, height=24, run_agent=agent) as h:
        h.feed_text("review script")
        h.feed_key("enter")
        assert wait_for(lambda: h.answer_prompt is not None, timeout=3.0)
        assert "script-line-000" in h.transcript
        assert "script-line-040" in h.transcript
        assert "script-line-079" in h.transcript
        h.feed_key("pageup", repeat=100)
        assert wait_for(lambda: "rm -rf cache" in h.plain_frame, timeout=3.0)
        h.feed_text("n")
        h.feed_key("enter")
        assert h.wait_idle(timeout=3.0)
    assert results["gate"][0] is False
