"""Output contracts and work bounds for rendering hot paths."""

import pytest
from rich.text import Span

from jarv import session_render, text_editor
from jarv.terminal_text import safe_terminal_link, safe_terminal_text
from jarv.tool_outputs import tool_outcome


@pytest.mark.parametrize("code", range(160))
def test_terminal_controls_are_escaped_with_unicode_surroundings(code):
    char = chr(code)
    escaped = "\\r" if code == 13 else f"\\x{code:02x}"
    expected = escaped if code not in (9, 10) and (code < 32 or code >= 127) else char

    rendered = safe_terminal_text(f"界{char}é")

    assert rendered == f"界{expected}é"
    assert safe_terminal_text(rendered) == rendered
    assert safe_terminal_link(f"https://example.test/界{char}é") == (32 <= code < 127)


def test_unicode_terminal_text_preserves_line_endings_and_noncontrols():
    assert safe_terminal_text("界\r\né\r\t😀\n") == "界\né\\r\t😀\n"
    assert safe_terminal_text("e\u0301界\u2028😀\u2029") == "e\u0301界\u2028😀\u2029"
    assert safe_terminal_link("https://example.test/界é😀")


def test_sanitized_text_keeps_plain_str_result_for_text_input_subclasses():
    from jarv.command_input import TextInput

    assert type(safe_terminal_text(TextInput("界é😀"))) is str


@pytest.mark.parametrize(
    ("cursor", "plain", "spans"),
    [
        (0, ">界e\u0301ab", [Span(1, 2, "reverse"), Span(2, 6, "white")]),
        (3, ">界e\u0301ab", [Span(1, 4, "white"), Span(4, 5, "reverse"), Span(5, 6, "white")]),
        (5, ">界e\u0301ab ", [Span(1, 6, "white"), Span(6, 7, "reverse")]),
    ],
)
def test_plain_draft_keeps_cursor_spans_without_character_style_checks(monkeypatch, cursor, plain, spans):
    def unexpected_check(*args):
        raise AssertionError("plain drafts do not need marker membership checks")

    monkeypatch.setattr(text_editor, "_index_in_spans", unexpected_check)
    lines, active = text_editor.render_visual_lines(
        {"buffer": "界e\u0301ab", "cursor": cursor}, 20, indent=">", text_style="white"
    )

    assert active == 0
    assert lines[0].plain == plain
    assert lines[0].spans == spans


def test_cursor_row_lookup_uses_last_shared_endpoint_without_scanning_previous_rows():
    class ObservedRows(list):
        reads = 0

        def __getitem__(self, index):
            self.reads += 1
            return super().__getitem__(index)

    rows = ObservedRows((index, index + 1) for index in range(10_000))
    assert text_editor.cursor_row_index(rows, 9999) == 9999
    assert rows.reads == 1
    assert text_editor.cursor_row_index([(0, 4), (4, 4)], 4) == 1
    assert text_editor.cursor_row_index([(0, 4)], -1) == 0
    assert text_editor.cursor_row_index([], 0) == 0


class _HistoryWithoutSlices(list):
    def __getitem__(self, index):
        assert not isinstance(index, slice), "lookahead must not copy unrelated history"
        return super().__getitem__(index)


def test_history_lookahead_stops_at_nearby_output_and_visible_item():
    output = {"type": "function_call_output", "call_id": "tool", "output": "result"}
    visible = {"type": "function_call", "call_id": "next"}
    history = _HistoryWithoutSlices([None, output, visible] + [None] * 10_000)

    assert session_render._tool_call_output(history, 0, "tool") == "result"
    assert session_render._next_visible_history_item(history, 1) is visible


def test_history_lookahead_retains_slice_bounds_user_boundary_and_output_outcome():
    visible = {"role": "assistant", "content": "reply"}
    history = _HistoryWithoutSlices([
        {"role": "user", "content": "new turn"},
        {"type": "function_call_output", "call_id": "tool", "output": "result",
         "outcome": {"status": "success"}},
        {"role": "system", "content": "hidden"},
        visible,
    ])

    assert session_render._next_visible_history_item(history, -2) is visible
    assert session_render._next_visible_history_item(history, 100) is None
    assert session_render._next_visible_history_item(history, -100) is history[0]
    assert tool_outcome(session_render._tool_call_output(history, -3, "tool")).status == "unknown"
    assert tool_outcome(session_render._tool_call_output(history, -100, "tool")).status == "unknown"
    assert tool_outcome(session_render._tool_call_output(history, 0, "tool")).status == "success"
