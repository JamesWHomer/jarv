"""Completed turn summaries reach the terminal without further input."""

import threading
from types import SimpleNamespace

from conftest import wait_for
from headsup_harness import HeadsupHarness
from jarv import agent, headsup
from jarv.config import DEFAULT_CONFIG
from jarv.provider import StreamDone, TextDelta


def test_turn_summary_arriving_during_paint_is_shown_without_keypress(monkeypatch):
    previous_frame_ready = threading.Event()
    original_render = headsup.HeadsupApp.render

    def render_while_turn_finishes(app):
        frame = original_render(app)
        if not previous_frame_ready.is_set() and any(
            entry.kind == "assistant" for entry in app.entries
        ):
            # Hold the frame that predates the summary until the worker has
            # finished. Its invalidation must survive this in-flight paint.
            previous_frame_ready.set()
            assert wait_for(lambda: not app._agent_busy, timeout=3.0)
        return frame

    def stream(*_args, **_kwargs):
        yield TextDelta("Finished response.")
        assert previous_frame_ready.wait(3.0)
        yield StreamDone({"usage": {
            "input_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "completion_time": 0.5,
        }})

    monkeypatch.setattr(headsup.HeadsupApp, "render", render_while_turn_finishes)
    monkeypatch.setattr(agent, "stream_response", stream)
    config = {
        **DEFAULT_CONFIG,
        "provider": "groq",
        "turn_summary": True,
        "turn_summary_time": False,
        "headsup_intro_logo": False,
        "headsup_intro_stars": False,
    }
    with HeadsupHarness(
        width=120, height=20, config=config,
        args=SimpleNamespace(incognito=True), run_agent=agent.run_agent,
    ) as harness:
        harness.feed_text("Say hello")
        harness.feed_key("enter")
        # No input, resize, or animation tick may be needed to reveal the stats.
        assert wait_for(
            lambda: "Turn: 100 in" in harness.plain_frame, timeout=3.0,
        )
        assert "Finished response." in harness.plain_frame
        assert "20 out" in harness.plain_frame
        assert "120 total" in harness.plain_frame
        assert "40.0 tok/s (server)" in harness.plain_frame
        assert not harness.app._agent_busy
        assert harness.app._active_ui is None
