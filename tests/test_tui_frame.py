"""Unit tests for the pure heads-up frame-composition core (jarv.tui_frame).

These cover the geometry/layout math that historically caused WSL/ConPTY
wrap-guard and stale-edge visual bugs. Because the functions are side-effect
free, the regressions are now caught here directly instead of only through a
live render.
"""

import io

from rich.cells import cell_len
from rich.console import Console
from rich.text import Text

from jarv import tui_frame


def _emulate_terminal(stream: str, width: int, height: int) -> list[str]:
    """Replay a control stream onto a grid, modelling the pending-wrap latch.

    Just enough VT to catch the stale-edge regression: printable cells, CR/LF,
    cursor-home (``ESC[H``) and erase-to-end-of-line (``ESC[0K``). Auto-wrap is
    deferred -- writing the last column arms a latch instead of moving the cursor
    -- which is the behaviour that made an erase emitted *after* a flush
    right-border wipe that border (see ``EraseTrailingColumns``).
    """
    grid = [[" "] * width for _ in range(height)]
    row = col = 0
    pending_wrap = False
    i = 0
    while i < len(stream):
        ch = stream[i]
        if ch == "\x1b" and stream[i + 1] == "[":
            j = i + 2
            params = ""
            while j < len(stream) and not stream[j].isalpha():
                params += stream[j]
                j += 1
            final = stream[j]
            if final == "H":
                row = col = 0
                pending_wrap = False
            elif final == "K" and params in ("", "0"):
                for c in range(col, width):
                    grid[row][c] = " "
                pending_wrap = False
            i = j + 1
            continue
        if ch in "\r\n":
            if ch == "\n":
                row = min(height - 1, row + 1)
            col = 0
            pending_wrap = False
            i += 1
            continue
        if pending_wrap:
            row = min(height - 1, row + 1)
            col = 0
            pending_wrap = False
        if row < height and col < width:
            grid[row][col] = ch
        if col == width - 1:
            pending_wrap = True
        else:
            col += 1
        i += 1
    return ["".join(r) for r in grid]


def _render_to_terminal(width: int, height: int) -> list[str]:
    layout = tui_frame.compute_layout(width, height)
    frame = tui_frame.build_frame(
        [Text("hello"), Text("world")],
        title=tui_frame.compose_title("openai / gpt-x / high", layout.panel_width),
        subtitle=Text("$0.00 - 0% full"),
        panel_width=layout.panel_width,
        term_h=layout.term_h,
    )
    console = Console(
        file=io.StringIO(),
        width=width,
        height=height,
        force_terminal=True,
        color_system=None,
        legacy_windows=False,
    )
    console.print(frame)
    return _emulate_terminal(console.file.getvalue(), width, height)


def test_frame_keeps_flush_right_border_on_a_real_terminal():
    # Regression: on a terminal the erase-to-end-of-line used to fire after each
    # full-width row and wipe the border sitting in the last (auto-wrap) column,
    # leaving the heads-up panel with no right edge.
    width, height = 40, 12
    rows = _render_to_terminal(width, height)
    framed = [r for r in rows if r[0] == "│" or r[0] in "╭╰"]
    assert framed, "expected bordered rows to be drawn"
    # Top-right and bottom-right corners are present and flush to the edge.
    assert rows[0].rstrip(" ")[-1] == "╮"
    last_framed = max(idx for idx, r in enumerate(rows) if r.strip())
    assert rows[last_framed].rstrip(" ")[-1] == "╯"
    # Every interior bordered row carries the right edge in the final column.
    for r in framed:
        if r[0] == "│":
            assert r[width - 1] == "│", repr(r)


def test_panel_width_spans_full_terminal_width():
    # The panel border sits flush against the terminal edge -- no reserved gap.
    assert tui_frame.panel_width(80) == 80
    assert tui_frame.panel_width(120) == 120


def test_panel_width_never_below_one():
    assert tui_frame.panel_width(1) == 1
    assert tui_frame.panel_width(0) == 1


def test_compute_layout_clamps_tiny_terminals():
    layout = tui_frame.compute_layout(5, 2)
    assert layout.term_w == 20
    assert layout.term_h == 8
    assert layout.panel_width == 20
    assert layout.inner_width == layout.panel_width - 4
    assert layout.body_height == max(3, layout.term_h - 2)


def test_compute_layout_matches_historical_math():
    term_w, term_h = 100, 30
    layout = tui_frame.compute_layout(term_w, term_h)
    assert layout.panel_width == term_w
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


def _menu_console() -> Console:
    return Console(file=io.StringIO(), width=40, color_system=None)


def test_overlay_menu_anchors_block_to_bottom_and_keeps_background_to_the_right():
    background = [Text("S" * 30) for _ in range(5)]
    menu = [Text("/setup"), Text("/settings")]
    out = tui_frame.overlay_menu(
        background, menu, rows=5, width=30, console=_menu_console()
    )

    assert len(out) == 5
    # Rows above the menu are untouched -- the starfield/logo never move.
    assert [line.plain for line in out[:3]] == ["S" * 30] * 3
    # The menu is painted flush against the bottom of the body, left-aligned.
    assert out[-2].plain.startswith("/setup")
    assert out[-1].plain.startswith("/settings")
    # Stars show through to the right of the menu block; the block itself is a
    # clean rectangle (no stars bleed into the gap before them).
    block_width = len("/settings") + tui_frame._MENU_OVERLAY_GUTTER
    assert out[-1].plain == "/settings".ljust(block_width) + "S" * (30 - block_width)
    assert out[-2].plain == "/setup".ljust(block_width) + "S" * (30 - block_width)


def test_overlay_menu_is_noop_without_menu_lines():
    background = [Text("a"), Text("b")]
    assert tui_frame.overlay_menu(
        background, [], rows=4, width=10, console=_menu_console()
    ) is background


def test_overlay_menu_clips_when_menu_taller_than_body():
    background = [Text("S" * 12) for _ in range(2)]
    menu = [Text("one"), Text("two"), Text("three")]
    out = tui_frame.overlay_menu(
        background, menu, rows=2, width=12, console=_menu_console()
    )
    # Only the rows that fit are kept; the menu stays anchored to the bottom so
    # its tail rows win over the clipped head.
    assert len(out) == 2
    assert out[-1].plain.startswith("three")
    assert out[-2].plain.startswith("two")
