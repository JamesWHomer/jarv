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


def render_visual_lines(
    state: dict,
    content_width: int,
    *,
    indent: str = "",
    masked: bool = False,
    text_style: str = "green",
    cursor_style: str = "reverse",
) -> tuple[list[Text], int]:
    value = str(state.get("buffer", ""))
    display = _display_value(value, masked=masked)
    cursor = max(0, min(int(state.get("cursor", len(value))), len(value)))
    rows = visual_rows(value, content_width)
    active_row = cursor_row_index(rows, cursor)
    rendered: list[Text] = []

    for idx, (row_start, row_end) in enumerate(rows):
        segment = display[row_start:row_end]
        line = Text(indent)
        if idx == active_row:
            local_cursor = cursor - row_start
            line.append(segment[:local_cursor], style=text_style)
            if local_cursor < len(segment):
                line.append(segment[local_cursor], style=cursor_style)
                line.append(segment[local_cursor + 1 :], style=text_style)
            else:
                line.append(" ", style=cursor_style)
        else:
            line.append(segment, style=text_style)
        rendered.append(line)

    return rendered, active_row


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
    changed = False

    if key == "LEFT":
        state["cursor"] = max(0, cursor - repeat_count)
        state["preferred_visual_column"] = None
    elif key == "RIGHT":
        state["cursor"] = min(len(value), cursor + repeat_count)
        state["preferred_visual_column"] = None
    elif key == "HOME":
        if allow_newlines:
            rows = visual_rows(value, content_width)
            state["cursor"] = rows[cursor_row_index(rows, cursor)][0]
        else:
            state["cursor"] = 0
        state["preferred_visual_column"] = None
    elif key == "END":
        if allow_newlines:
            rows = visual_rows(value, content_width)
            state["cursor"] = rows[cursor_row_index(rows, cursor)][1]
        else:
            state["cursor"] = len(value)
        state["preferred_visual_column"] = None
    elif allow_newlines and key == "UP":
        _move_vertical(
            state,
            -1,
            content_width=content_width,
            count=repeat_count,
        )
    elif allow_newlines and key == "DOWN":
        _move_vertical(
            state,
            1,
            content_width=content_width,
            count=repeat_count,
        )
    elif key == "BACKSPACE" and cursor:
        state["buffer"] = value[: cursor - 1] + value[cursor:]
        state["cursor"] = cursor - 1
        changed = True
    elif key == "DELETE" and cursor < len(value):
        state["buffer"] = value[:cursor] + value[cursor + 1 :]
        changed = True
    elif allow_newlines and key == "ENTER":
        state["buffer"] = value[:cursor] + "\n" + value[cursor:]
        state["cursor"] = cursor + 1
        changed = True
    elif (
        isinstance(key, str)
        and key
        and (len(key) == 1 or isinstance(key, TextInput))
        and all(char.isprintable() for char in key)
    ):
        state["buffer"] = value[:cursor] + key + value[cursor:]
        state["cursor"] = cursor + len(key)
        changed = True

    if changed:
        state["preferred_visual_column"] = None
    return changed
