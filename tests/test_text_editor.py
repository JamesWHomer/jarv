from jarv.command_input import TextInput
from jarv.text_editor import (
    apply_text_editor_key,
    initialize_text_editor,
    render_single_line,
    render_visual_line_window,
    render_visual_lines,
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
