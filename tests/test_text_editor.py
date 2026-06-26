from jarv.command_input import TextInput
from jarv.text_editor import (
    apply_text_editor_key,
    initialize_text_editor,
    render_single_line,
    render_visual_line_window,
    render_visual_lines,
    selection_bounds,
)


def test_single_line_editor_inserts_and_deletes_at_cursor():
    state = {}
    initialize_text_editor(state, "abcd")

    apply_text_editor_key(state, "LEFT")
    apply_text_editor_key(state, "LEFT")
    apply_text_editor_key(state, "X")
    apply_text_editor_key(state, "DELETE")

    assert state["buffer"] == "abXd"
    assert state["cursor"] == 3


def test_single_line_editor_inserts_batched_text_at_cursor():
    state = {}
    initialize_text_editor(state, "abcd")
    state["cursor"] = 2

    assert apply_text_editor_key(state, TextInput("custom/model"))
    assert state["buffer"] == "abcustom/modelcd"
    assert state["cursor"] == 14


def test_single_line_editor_flattens_multiline_paste():
    state = {}
    initialize_text_editor(state, "abcd")
    state["cursor"] = 2

    assert apply_text_editor_key(state, TextInput("x\ny"))
    assert state["buffer"] == "abx ycd"
    assert state["cursor"] == 5


def test_single_line_editor_ignores_enter_and_vertical_movement():
    state = {}
    initialize_text_editor(state, "abcd")
    state["cursor"] = 2

    assert not apply_text_editor_key(state, "ENTER")
    assert not apply_text_editor_key(state, "UP")
    assert state["buffer"] == "abcd"
    assert state["cursor"] == 2


def test_single_line_renderer_masks_value_and_keeps_cursor_visible():
    state = {}
    initialize_text_editor(state, "secret-value")

    rendered = render_single_line(state, 6, masked=True)

    assert rendered.plain == "***** "
    assert "reverse" in str(rendered.spans[-1].style)
    assert "secret" not in rendered.plain


def test_visual_lines_highlight_spans_get_distinct_style():
    state = {}
    initialize_text_editor(state, "ab[chip]cd", multiline=True)
    state["cursor"] = 0  # cursor on 'a', clear of the highlighted span

    lines, _ = render_visual_lines(
        state,
        40,
        text_style="white",
        highlight_spans=[(2, 8)],
        highlight_style="dim cyan",
    )

    line = lines[0]
    assert line.plain == "ab[chip]cd"
    chip = "".join(line.plain[s.start:s.end] for s in line.spans if str(s.style) == "dim cyan")
    assert chip == "[chip]"


def test_visual_lines_without_highlight_render_one_run_per_segment():
    state = {}
    initialize_text_editor(state, "plain text", multiline=True)
    state["cursor"] = len("plain text")

    lines, _ = render_visual_lines(state, 40, text_style="white")

    # No highlight spans -> the body is a single styled run (plus the cursor).
    body = [s for s in lines[0].spans if str(s.style) == "white"]
    assert len(body) == 1
    assert lines[0].plain == "plain text "  # trailing cursor cell


def test_multiline_renderer_and_navigation_share_visual_wraps():
    state = {}
    initialize_text_editor(state, "abcdefghijkl", multiline=True)
    state["cursor"] = 2

    apply_text_editor_key(
        state,
        "DOWN",
        content_width=4,
        allow_newlines=True,
    )
    lines, cursor_line = render_visual_lines(state, 4)

    assert state["cursor"] == 6
    assert cursor_line == 1
    assert lines[1].plain == "efgh"


def test_multiline_editor_inserts_pasted_newlines_and_tabs():
    state = {}
    initialize_text_editor(state, "ab", multiline=True)
    state["cursor"] = 1

    assert apply_text_editor_key(
        state,
        TextInput("x\r\n\ty"),
        allow_newlines=True,
    )

    assert state["buffer"] == "ax\n\tyb"
    assert state["cursor"] == 5


def test_ctrl_arrow_jumps_by_word():
    state = {}
    initialize_text_editor(state, "hello world foo")
    state["cursor"] = len("hello world foo")

    assert not apply_text_editor_key(state, "CTRL_LEFT")
    assert state["cursor"] == len("hello world ")  # start of "foo"
    apply_text_editor_key(state, "CTRL_LEFT")
    assert state["cursor"] == len("hello ")  # start of "world"
    apply_text_editor_key(state, "CTRL_RIGHT")
    assert state["cursor"] == len("hello world")  # end of "world"


def test_shift_arrow_extends_and_collapses_selection():
    state = {}
    initialize_text_editor(state, "abcdef")
    state["cursor"] = 3

    apply_text_editor_key(state, "SHIFT_RIGHT")
    apply_text_editor_key(state, "SHIFT_RIGHT")
    assert state["selection_anchor"] == 3
    assert state["cursor"] == 5
    assert selection_bounds(state) == (3, 5)

    # Shrinking back to the anchor clears the selection entirely.
    apply_text_editor_key(state, "SHIFT_LEFT")
    apply_text_editor_key(state, "SHIFT_LEFT")
    assert state["selection_anchor"] is None
    assert selection_bounds(state) is None
    assert state["cursor"] == 3


def test_ctrl_shift_arrow_selects_by_word():
    state = {}
    initialize_text_editor(state, "hello world")
    state["cursor"] = 0

    apply_text_editor_key(state, "CTRL_SHIFT_RIGHT")
    assert selection_bounds(state) == (0, len("hello"))
    apply_text_editor_key(state, "CTRL_SHIFT_RIGHT")
    assert selection_bounds(state) == (0, len("hello world"))


def test_typing_replaces_selection():
    state = {}
    initialize_text_editor(state, "hello world")
    state["cursor"] = 6  # before "world"
    apply_text_editor_key(state, "CTRL_SHIFT_RIGHT")  # select "world"

    assert apply_text_editor_key(state, "X")
    assert state["buffer"] == "hello X"
    assert state["cursor"] == 7
    assert state["selection_anchor"] is None


def test_backspace_deletes_selection():
    state = {}
    initialize_text_editor(state, "hello world")
    state["cursor"] = 5  # after "hello"
    apply_text_editor_key(state, "SHIFT_LEFT")
    apply_text_editor_key(state, "SHIFT_LEFT")  # select "lo"

    assert apply_text_editor_key(state, "BACKSPACE")
    assert state["buffer"] == "hel world"
    assert state["cursor"] == 3
    assert state["selection_anchor"] is None


def test_pasted_text_replaces_selection():
    state = {}
    initialize_text_editor(state, "hello world", multiline=True)
    state["cursor"] = 0
    apply_text_editor_key(state, "CTRL_SHIFT_RIGHT")  # select "hello"

    assert apply_text_editor_key(state, TextInput("hi"), allow_newlines=True)
    assert state["buffer"] == "hi world"
    assert state["cursor"] == 2


def test_plain_left_collapses_selection_to_left_edge():
    state = {}
    initialize_text_editor(state, "abcdef")
    state["cursor"] = 2
    apply_text_editor_key(state, "SHIFT_RIGHT")
    apply_text_editor_key(state, "SHIFT_RIGHT")  # select [2, 4)

    apply_text_editor_key(state, "LEFT")
    assert state["cursor"] == 2  # collapsed to the left edge, no extra move
    assert state["selection_anchor"] is None


def test_render_visual_lines_paints_selection_span():
    state = {}
    initialize_text_editor(state, "hello world", multiline=True)
    state["cursor"] = 0  # keep the cursor off the selected run

    lines, _ = render_visual_lines(
        state,
        40,
        text_style="white",
        selection_span=(6, 11),
        selection_style="black on cyan",
    )
    line = lines[0]
    selected = "".join(
        line.plain[s.start : s.end]
        for s in line.spans
        if str(s.style) == "black on cyan"
    )
    assert selected == "world"


def test_visual_line_window_keeps_cursor_visible():
    state = {}
    initialize_text_editor(state, "a\nb\nc\nd", multiline=True)
    state["cursor"] = len("a\nb\nc")

    lines, cursor_line, start = render_visual_line_window(
        state,
        10,
        max_lines=2,
    )

    assert [line.plain for line in lines] == ["b", "c "]
    assert cursor_line == 1
    assert start == 1
