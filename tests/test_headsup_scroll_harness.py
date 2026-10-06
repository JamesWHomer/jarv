"""Streaming scroll behavior through the real heads-up input/render loop."""

import re
from unittest.mock import patch

import pytest
from rich.text import Text

from conftest import wait_for
from headsup_harness import HeadsupHarness


_CONFIG = {
    "provider": "openai",
    "model": "test-model",
    "headsup_intro_logo": False,
    "headsup_intro_stars": False,
}
_LINE_MARKER = re.compile(r"(?:assistant|history|tool|live)-\d{3}")


def _lines(prefix, count):
    return "\n".join(f"{prefix}-{number:03}" for number in range(count))


def _assistant_text(count):
    # A code block gives each streamed line a stable visible row.
    return "```text\n" + _lines("assistant", count) + "\n```"


def _paint_after(harness, change):
    """Wait for a loop paint after a producer update or queued input.

    A queued barrier records the refresh count on the loop thread, so an older
    frame completing concurrently with the update cannot satisfy the wait.
    """
    marker = object()
    observed = []
    original = harness.app.on_app_event

    def on_event(event):
        if event is marker:
            observed.append(harness.live.refresh_count)
        else:
            original(event)

    with patch.object(harness.app, "on_app_event", on_event):
        result = change()
        harness.app.post(marker)
        assert wait_for(
            lambda: observed and harness.live.refresh_count > observed[0],
            timeout=3.0,
        ), "heads-up loop did not paint the updated transcript"
    return result


def _visible_markers(harness):
    return [
        (row, marker.group())
        for row, line in enumerate(harness.plain_frame.splitlines())
        for marker in _LINE_MARKER.finditer(line)
    ]


@pytest.mark.parametrize("resume_key", ["CTRL_END", "MOUSE_WHEEL_DOWN"])
def test_streaming_holds_visible_rows_until_user_resumes_following(resume_key):
    with HeadsupHarness(width=100, height=24, config=dict(_CONFIG)) as h:
        index = _paint_after(
            h, lambda: h.app.upsert_assistant_message(None, _assistant_text(40))
        )
        assert "assistant-039" in h.plain_frame

        _paint_after(
            h, lambda: h.app.upsert_assistant_message(index, _assistant_text(45))
        )
        assert "assistant-044" in h.plain_frame

        _paint_after(h, lambda: h.feed_key("MOUSE_WHEEL_UP", repeat=4))
        reading = _visible_markers(h)
        assert reading
        assert "assistant-044" not in h.plain_frame
        assert "Ctrl+End jump to latest" in h.plain_frame

        # Grow the very response containing the anchor, then append and replace
        # tools beneath it. None of these updates should move the visible rows.
        for count in (50, 62, 75):
            _paint_after(
                h,
                lambda count=count: h.app.upsert_assistant_message(
                    index, _assistant_text(count)
                ),
            )
            assert _visible_markers(h) == reading

        _paint_after(h, lambda: h.app.add_tool(Text(_lines("tool", 12))))
        assert _visible_markers(h) == reading

        _paint_after(
            h, lambda: h.app.upsert_live_tool("running", Text(_lines("live", 4)))
        )
        assert _visible_markers(h) == reading

        _paint_after(
            h,
            lambda: h.app.replace_live_tool("running", Text(_lines("live", 30))),
        )
        assert _visible_markers(h) == reading

        repeat = 1000 if resume_key == "MOUSE_WHEEL_DOWN" else 1
        _paint_after(h, lambda: h.feed_key(resume_key, repeat=repeat))
        assert "live-029" in h.plain_frame
        assert "Auto-scroll paused" not in h.plain_frame

        _paint_after(h, lambda: h.app.add_tool(Text("tool-999")))
        assert "tool-999" in h.plain_frame
        assert "Auto-scroll paused" not in h.plain_frame


def test_live_tool_invalidation_keeps_history_in_place():
    with HeadsupHarness(width=100, height=24, config=dict(_CONFIG)) as h:
        _paint_after(h, lambda: h.app.add_tool(Text(_lines("history", 60))))
        live_output = Text(_lines("live", 4))
        _paint_after(h, lambda: h.app.upsert_live_tool("running", live_output))
        assert "live-003" in h.plain_frame

        _paint_after(h, lambda: h.feed_key("PAGEUP", repeat=3))
        reading = _visible_markers(h)
        assert reading
        assert all(marker.startswith("history-") for _, marker in reading)

        def append_live_output():
            with h.app.lock:
                live_output.append("\n" + _lines("live", 35))
                h.app.invalidate_live_tool("running")
            h.app.refresh()

        _paint_after(h, append_live_output)
        assert _visible_markers(h) == reading

        _paint_after(h, lambda: h.feed_key("CTRL_END"))
        assert "live-034" in h.plain_frame
