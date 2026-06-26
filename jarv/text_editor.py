"""Reusable cursor-aware text editing primitives for terminal views."""

from rich.text import Text

from .command_input import TextInput


def initialize_text_editor(
    state: dict,
    value: str,
    *,
    multiline: bool = False,
) -> None:
    state.update(
        buffer=value,
        cursor=len(value),
        original=value,
        multiline=multiline,
        preferred_visual_column=None,
        # Start of an active Shift-selection (None when nothing is selected). The
        # selected range is the span between this anchor and the cursor.
        selection_anchor=None,
    )


def visual_rows(value: str, content_width: int) -> list[tuple[int, int]]:
    content_width = max(1, content_width)
    rows: list[tuple[int, int]] = []
    absolute_start = 0
    logical_lines = value.split("\n")

    for logical_idx, logical_line in enumerate(logical_lines):
        line_length = len(logical_line)
        segment_starts = list(range(0, max(1, line_length), content_width))
        if line_length and line_length % content_width == 0:
            segment_starts.append(line_length)

        for segment_start in segment_starts:
            row_start = absolute_start + segment_start
            row_end = min(row_start + content_width, absolute_start + line_length)
            rows.append((row_start, row_end))

        if logical_idx < len(logical_lines) - 1:
            absolute_start += line_length + 1

    return rows


def cursor_row_index(rows: list[tuple[int, int]], cursor: int) -> int:
    matches = [
        idx
        for idx, (row_start, row_end) in enumerate(rows)
        if row_start <= cursor <= row_end
    ]
    return matches[-1] if matches else max(0, len(rows) - 1)


def _display_value(value: str, *, masked: bool) -> str:
    if not masked:
        return value
    return "".join("\n" if char == "\n" else "*" for char in value)


def _index_in_spans(index: int, spans) -> bool:
    return any(start <= index < end for start, end in spans)


def _append_segment(
    line: Text,
    segment: str,
    abs_start: int,
    local_cursor: int | None,
    *,
    spans,
    base_style: str,
    highlight_style: str,
    cursor_style: str,
    selection: tuple[int, int] | None = None,
    selection_style: str = "",
) -> None:
    """Append ``segment`` to ``line``, styling marker spans (and the cursor).

    With no spans this collapses to one styled run (plus the cursor split), so
    the rendered output is identical to plain text styling. An active
    ``selection`` span (highest priority after the cursor) paints the selected
    range with ``selection_style``.
    """
    def style_at(offset: int) -> str:
        idx = abs_start + offset
        if selection is not None and selection_style and selection[0] <= idx < selection[1]:
            return selection_style
        return highlight_style if _index_in_spans(idx, spans) else base_style

    length = len(segment)
    offset = 0
    while offset < length:
        if local_cursor is not None and offset == local_cursor:
            line.append(segment[offset], style=cursor_style)
            offset += 1
            continue
        run_style = style_at(offset)
        end = offset + 1
        while (
            end < length
            and not (local_cursor is not None and end == local_cursor)
            and style_at(end) == run_style
        ):
            end += 1
        line.append(segment[offset:end], style=run_style)
        offset = end

    if local_cursor is not None and local_cursor >= length:
        line.append(" ", style=cursor_style)


def render_visual_lines(
    state: dict,
    content_width: int,
    *,
    indent: str = "",
    masked: bool = False,
    text_style: str = "green",
    cursor_style: str = "reverse",
    highlight_spans=None,
    highlight_style: str = "cyan",
    selection_span: tuple[int, int] | None = None,
    selection_style: str = "",
) -> tuple[list[Text], int]:
    value = str(state.get("buffer", ""))
    display = _display_value(value, masked=masked)
    cursor = max(0, min(int(state.get("cursor", len(value))), len(value)))
    rows = visual_rows(value, content_width)
    active_row = cursor_row_index(rows, cursor)
    spans = highlight_spans or ()
    rendered: list[Text] = []

    for idx, (row_start, row_end) in enumerate(rows):
        segment = display[row_start:row_end]
        line = Text(indent)
        _append_segment(
            line,
            segment,
            row_start,
            (cursor - row_start) if idx == active_row else None,
            spans=spans,
            base_style=text_style,
            highlight_style=highlight_style,
            cursor_style=cursor_style,
            selection=selection_span,
            selection_style=selection_style,
        )
        rendered.append(line)

    return rendered, active_row


def render_visual_line_window(
    state: dict,
    content_width: int,
    *,
    max_lines: int | None = None,
    indent: str = "",
    masked: bool = False,
    text_style: str = "green",
    cursor_style: str = "reverse",
    highlight_spans=None,
    highlight_style: str = "cyan",
    selection_span: tuple[int, int] | None = None,
    selection_style: str = "",
) -> tuple[list[Text], int, int]:
    lines, cursor_idx = render_visual_lines(
        state,
        content_width,
        indent=indent,
        masked=masked,
        text_style=text_style,
        cursor_style=cursor_style,
        highlight_spans=highlight_spans,
        highlight_style=highlight_style,
        selection_span=selection_span,
        selection_style=selection_style,
    )
    if max_lines is None:
        return lines, cursor_idx, 0
    visible_count = max(1, max_lines)
    start = max(0, min(cursor_idx - visible_count + 1, len(lines) - visible_count))
    return lines[start : start + visible_count], cursor_idx - start, start


def render_single_line(
    state: dict,
    width: int,
    *,
    masked: bool = False,
    text_style: str = "green",
    cursor_style: str = "reverse",
    cursor_visible: bool = True,
) -> Text:
    value = str(state.get("buffer", ""))
    display = _display_value(value, masked=masked).replace("\n", " ")
    cursor = max(0, min(int(state.get("cursor", len(value))), len(value)))
    width = max(1, width)

    if cursor == len(display):
        start = max(0, cursor - max(0, width - 1))
        end = cursor
    else:
        start = max(0, cursor - width + 1)
        end = min(len(display), start + width)
    segment = display[start:end]
    local_cursor = cursor - start

    line = Text()
    if not cursor_visible:
        line.append(segment, style=text_style)
        return line
    line.append(segment[:local_cursor], style=text_style)
    if local_cursor < len(segment):
        line.append(segment[local_cursor], style=cursor_style)
        line.append(segment[local_cursor + 1 :], style=text_style)
    else:
        line.append(" ", style=cursor_style)
    return line


def _move_vertical(
    state: dict,
    direction: int,
    *,
    content_width: int,
    count: int,
) -> None:
    value = str(state.get("buffer", ""))
    cursor = max(0, min(int(state.get("cursor", len(value))), len(value)))
    rows = visual_rows(value, content_width)
    current_idx = cursor_row_index(rows, cursor)
    current_start, _current_end = rows[current_idx]
    preferred_column = state.get("preferred_visual_column")
    if preferred_column is None:
        preferred_column = cursor - current_start
    target_idx = max(0, min(len(rows) - 1, current_idx + direction * max(1, count)))
    target_start, target_end = rows[target_idx]
    state["cursor"] = min(target_start + int(preferred_column), target_end)
    state["preferred_visual_column"] = preferred_column


# Shift / Ctrl+Shift + arrow tokens that extend the active selection.
_SELECT_EXTEND_KEYS = frozenset(
    {"SHIFT_LEFT", "SHIFT_RIGHT", "CTRL_SHIFT_LEFT", "CTRL_SHIFT_RIGHT"}
)


def _next_word_boundary(value: str, cursor: int) -> int:
    """Index at the end of the word at/after ``cursor`` (whitespace-delimited)."""
    length = len(value)
    index = cursor
    while index < length and value[index].isspace():
        index += 1
    while index < length and not value[index].isspace():
        index += 1
    return index


def _prev_word_boundary(value: str, cursor: int) -> int:
    """Index at the start of the word at/before ``cursor`` (whitespace-delimited)."""
    index = cursor
    while index > 0 and value[index - 1].isspace():
        index -= 1
    while index > 0 and not value[index - 1].isspace():
        index -= 1
    return index


def selection_bounds(state: dict) -> tuple[int, int] | None:
    """Return the ``(start, end)`` of the active selection, or ``None``.

    The selection spans from ``selection_anchor`` to ``cursor``; a collapsed span
    (anchor == cursor) counts as no selection.
    """
    anchor = state.get("selection_anchor")
    if anchor is None:
        return None
    value = str(state.get("buffer", ""))
    cursor = max(0, min(int(state.get("cursor", len(value))), len(value)))
    anchor = max(0, min(int(anchor), len(value)))
    lo, hi = (anchor, cursor) if anchor <= cursor else (cursor, anchor)
    if lo == hi:
        return None
    return lo, hi


def _word_jump(value: str, cursor: int, *, forward: bool, count: int) -> int:
    pos = cursor
    for _ in range(max(1, count)):
        pos = _next_word_boundary(value, pos) if forward else _prev_word_boundary(value, pos)
    return pos


def apply_text_editor_key(
    state: dict,
    key: str,
    repeat_count: int = 1,
    *,
    content_width: int = 80,
    allow_newlines: bool = False,
) -> bool:
    value = str(state.get("buffer", ""))
    cursor = max(0, min(int(state.get("cursor", len(value))), len(value)))
    repeat = max(1, repeat_count)
    selection = selection_bounds(state)
    changed = False

    # --- Selection-extending motion: keep the anchor, move the cursor. ---
    if key in _SELECT_EXTEND_KEYS:
        if state.get("selection_anchor") is None:
            state["selection_anchor"] = cursor
        if key == "SHIFT_LEFT":
            new_cursor = max(0, cursor - repeat)
        elif key == "SHIFT_RIGHT":
            new_cursor = min(len(value), cursor + repeat)
        elif key == "CTRL_SHIFT_LEFT":
            new_cursor = _word_jump(value, cursor, forward=False, count=repeat)
        else:  # CTRL_SHIFT_RIGHT
            new_cursor = _word_jump(value, cursor, forward=True, count=repeat)
        state["cursor"] = new_cursor
        state["preferred_visual_column"] = None
        if state.get("selection_anchor") == new_cursor:
            # Collapsed back onto the anchor -> nothing selected.
            state["selection_anchor"] = None
        return False

    # --- Word-wise motion (Ctrl + arrow): move and drop any selection. ---
    if key == "CTRL_LEFT":
        state["cursor"] = _word_jump(value, cursor, forward=False, count=repeat)
        state["selection_anchor"] = None
        state["preferred_visual_column"] = None
        return False
    if key == "CTRL_RIGHT":
        state["cursor"] = _word_jump(value, cursor, forward=True, count=repeat)
        state["selection_anchor"] = None
        state["preferred_visual_column"] = None
        return False

    if key == "LEFT":
        # With a selection, plain Left collapses to its left edge.
        state["cursor"] = selection[0] if selection else max(0, cursor - repeat)
        state["selection_anchor"] = None
        state["preferred_visual_column"] = None
    elif key == "RIGHT":
        state["cursor"] = selection[1] if selection else min(len(value), cursor + repeat)
        state["selection_anchor"] = None
        state["preferred_visual_column"] = None
    elif key == "HOME":
        if allow_newlines:
            rows = visual_rows(value, content_width)
            state["cursor"] = rows[cursor_row_index(rows, cursor)][0]
        else:
            state["cursor"] = 0
        state["selection_anchor"] = None
        state["preferred_visual_column"] = None
    elif key == "END":
        if allow_newlines:
            rows = visual_rows(value, content_width)
            state["cursor"] = rows[cursor_row_index(rows, cursor)][1]
        else:
            state["cursor"] = len(value)
        state["selection_anchor"] = None
        state["preferred_visual_column"] = None
    elif allow_newlines and key == "UP":
        _move_vertical(
            state,
            -1,
            content_width=content_width,
            count=repeat,
        )
        state["selection_anchor"] = None
    elif allow_newlines and key == "DOWN":
        _move_vertical(
            state,
            1,
            content_width=content_width,
            count=repeat,
        )
        state["selection_anchor"] = None
    elif key == "BACKSPACE":
        if selection is not None:
            lo, hi = selection
            state["buffer"] = value[:lo] + value[hi:]
            state["cursor"] = lo
            state["selection_anchor"] = None
            changed = True
        elif cursor:
            state["buffer"] = value[: cursor - 1] + value[cursor:]
            state["cursor"] = cursor - 1
            changed = True
    elif key == "DELETE":
        if selection is not None:
            lo, hi = selection
            state["buffer"] = value[:lo] + value[hi:]
            state["cursor"] = lo
            state["selection_anchor"] = None
            changed = True
        elif cursor < len(value):
            state["buffer"] = value[:cursor] + value[cursor + 1 :]
            changed = True
    elif allow_newlines and key == "ENTER":
        lo, hi = selection if selection is not None else (cursor, cursor)
        state["buffer"] = value[:lo] + "\n" + value[hi:]
        state["cursor"] = lo + 1
        state["selection_anchor"] = None
        changed = True
    elif isinstance(key, TextInput):
        inserted = str(key).replace("\r\n", "\n").replace("\r", "\n")
        if not allow_newlines:
            inserted = inserted.replace("\n", " ")
        if inserted and all(
            char == "\n" or char == "\t" or char.isprintable()
            for char in inserted
        ):
            lo, hi = selection if selection is not None else (cursor, cursor)
            state["buffer"] = value[:lo] + inserted + value[hi:]
            state["cursor"] = lo + len(inserted)
            state["selection_anchor"] = None
            changed = True
    elif (
        isinstance(key, str)
        and key
        and len(key) == 1
        and key.isprintable()
    ):
        lo, hi = selection if selection is not None else (cursor, cursor)
        state["buffer"] = value[:lo] + key + value[hi:]
        state["cursor"] = lo + len(key)
        state["selection_anchor"] = None
        changed = True

    if changed:
        state["preferred_visual_column"] = None
    return changed
