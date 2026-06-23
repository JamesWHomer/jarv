"""Unit tests for the pure heads-up frame-composition core (jarv.tui_frame).

These cover the geometry/layout math that historically caused WSL/ConPTY
wrap-guard and stale-edge visual bugs. Because the functions are side-effect
free, the regressions are now caught here directly instead of only through a
live render.
"""

from rich.cells import cell_len
from rich.text import Text

from jarv import tui_frame
from jarv.tui_capabilities import wrap_guard_columns


def test_panel_width_reserves_wrap_guard_columns():
    assert tui_frame.panel_width(80) == 80 - wrap_guard_columns()
    assert tui_frame.panel_width(120) == 120 - wrap_guard_columns()


def test_panel_width_never_below_one():
    assert tui_frame.panel_width(1) == 1
    assert tui_frame.panel_width(0) == 1


def test_compute_layout_clamps_tiny_terminals():
    layout = tui_frame.compute_layout(5, 2)
    assert layout.term_w == 20
    assert layout.term_h == 8
    assert layout.panel_width == 20 - wrap_guard_columns()
    assert layout.inner_width == layout.panel_width - 4
    assert layout.body_height == max(3, layout.term_h - 2)


def test_compute_layout_matches_historical_math():
    term_w, term_h = 100, 30
    layout = tui_frame.compute_layout(term_w, term_h)
    assert layout.panel_width == term_w - wrap_guard_columns()
    assert layout.inner_width == layout.panel_width - 4
    assert layout.body_height == term_h - 2
    assert layout.max_prompt_rows == min(8, max(1, layout.body_height - 2), max(3, term_h // 3))


def test_transcript_rows_leaves_room_for_footer_and_prompt():
    assert tui_frame.transcript_rows(body_height=20, prompt_row_count=3) == 16
    # never collapses below a single row
    assert tui_frame.transcript_rows(body_height=3, prompt_row_count=8) == 1


def test_window_transcript_pins_newest_to_bottom_by_default():
    lines = [Text(str(i)) for i in range(10)]
    visible, clamped = tui_frame.window_transcript(lines, rows=4, scroll_offset=0)
    assert [t.plain for t in visible] == ["6", "7", "8", "9"]
    assert clamped == 0


def test_window_transcript_scrolls_up_and_clamps_offset():
    lines = [Text(str(i)) for i in range(10)]
    visible, clamped = tui_frame.window_transcript(lines, rows=4, scroll_offset=2)
    assert [t.plain for t in visible] == ["4", "5", "6", "7"]
    assert clamped == 2

    # Offset beyond the top is clamped to the maximum scroll.
    visible, clamped = tui_frame.window_transcript(lines, rows=4, scroll_offset=999)
    assert [t.plain for t in visible] == ["0", "1", "2", "3"]
    assert clamped == 6


def test_window_transcript_handles_short_transcript():
    lines = [Text("only")]
    visible, clamped = tui_frame.window_transcript(lines, rows=4, scroll_offset=5)
    assert [t.plain for t in visible] == ["only"]
    assert clamped == 0


def test_compose_title_includes_label_and_status_within_budget():
    title = tui_frame.compose_title("openai / gpt-x / high", panel_width=80)
    assert "jarv" in title.plain
    assert "openai / gpt-x / high" in title.plain
    # Title must stay within Rich's six-cell budget so it cannot push past the guard.
    assert cell_len(title.plain) <= 80 - 6


def test_compose_title_drops_status_when_no_room():
    title = tui_frame.compose_title("a very long model status string", panel_width=14)
    assert title.plain.startswith("jarv")
    assert cell_len(title.plain) <= max(1, 14 - 6)


def test_assemble_body_pads_to_full_height_and_appends_footer_and_prompt():
    footer = Text("FOOTER")
    prompt = [Text("PROMPT")]
    parts = tui_frame.assemble_body(
        visible=[Text("a"), Text("b")],
        footer=footer,
        prompt_lines=prompt,
        body_height=8,
        rows=5,
    )
    plains = [p.plain for p in parts]
    assert plains[-1] == "PROMPT"
    assert plains[-2] == "FOOTER"
    # The transcript region fills (body_height - 1 - len(prompt)) rows and keeps content.
    transcript_region = plains[:-2]
    assert len(transcript_region) == 8 - 1 - 1
    assert "a" in transcript_region and "b" in transcript_region


def test_assemble_body_trims_overflowing_transcript():
    visible = [Text(str(i)) for i in range(20)]
    parts = tui_frame.assemble_body(
        visible=visible,
        footer=Text("F"),
        prompt_lines=[Text("P")],
        body_height=6,
        rows=4,
    )
    # footer + prompt are always present; transcript is trimmed to fit body_height
    assert parts[-1].plain == "P"
    assert any(p.plain == "F" for p in parts)
