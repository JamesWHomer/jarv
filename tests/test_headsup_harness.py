"""Smoke tests that the headless heads-up harness drives a real loop.

These keep the harness (and the symbols it patches) honest against the live
``HeadsupApp`` source.
"""

from types import SimpleNamespace

from headsup_harness import HeadsupHarness, strip_ansi


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
        h.wait_idle()
        assert "hello harness" in h.plain_frame
        assert h.prompt_buffer == "hello harness"


def test_harness_submits_query_and_renders_reply():
    with HeadsupHarness(width=72, height=16, run_agent=_fake_agent) as h:
        h.feed_text("ping")
        h.feed_key("enter")
        h.wait_idle()
        assert "ping" in h.transcript
        assert "reply to ping" in h.transcript
        # Submitting clears the prompt buffer.
        assert h.prompt_buffer == ""


def test_harness_resize_changes_reported_size():
    with HeadsupHarness(width=70, height=14, run_agent=_fake_agent) as h:
        assert h.state()["size"] == [70, 14]
        h.resize(110, 30)
        h.wait_idle()
        assert h.state()["size"] == [110, 30]


def test_harness_frame_carries_stale_edge_erase():
    with HeadsupHarness(width=80, height=16, run_agent=_fake_agent) as h:
        h.feed_text("content")
        h.wait_idle()
        # Heads-up routes its frame through EraseTrailingColumns, so the captured
        # terminal frame includes the erase-to-end-of-line control.
        assert "\x1b[0K" in h.frame


def test_strip_ansi_removes_escape_sequences():
    assert strip_ansi("\x1b[31mred\x1b[0m\x1b[0K") == "red"
