"""Text and row layout for the session picker (no storage or terminal I/O)."""

import re
from bisect import bisect_right

from rich.text import Text

from .tool_outputs import flatten_content_text


def one_line(value: str) -> str:
    return " ".join(value.split())


def fitted(value: str | Text, width: int, *, pad: bool = False) -> Text:
    """Clip in terminal cells, including wide and combining characters."""
    text = value.copy() if isinstance(value, Text) else Text(value)
    text.no_wrap = True
    text.overflow = "crop"
    text.truncate(max(0, width), overflow="ellipsis", pad=pad)
    return text


def highlighted(value: str, query: str, style: str = "") -> Text:
    text = Text(value, style=style)
    if query.strip():
        text.highlight_regex(re.compile(re.escape(query.strip()), re.IGNORECASE), "bold black on bright_cyan")
    return text


def conversation_title(meta: dict, snippet: str = "") -> str:
    saved = meta.get("title")
    if isinstance(saved, str) and saved.strip():
        return one_line(saved)
    return one_line(snippet) or "Untitled conversation"


def first_prompt(history: list) -> str:
    for item in history:
        if isinstance(item, dict) and item.get("role") == "user":
            text = one_line(flatten_content_text(item.get("content", "")))
            if text:
                return text
    return ""


def match_excerpt(text: str, query: str, width: int) -> str:
    match = re.search(re.escape(query.strip()), text, re.IGNORECASE) if query.strip() else None
    if match is None:
        return ""
    # Keep a little leading context, but put the match within the viewport.
    start = max(0, match.start() - min(20, max(0, width // 4)))
    # Only normalize/highlight the small visible excerpt, never the remainder
    # of a potentially megabyte-sized transcript on every frame.
    end = max(match.end(), start + max(1, width) * 3)
    return ("…" if start else "") + one_line(text[start:end])


def highlighted_transcript(lines: list[Text], query: str) -> tuple[list[Text], int | None]:
    """Highlight literal searches even when words wrap onto the next line."""
    if not query.strip():
        return lines, None
    starts = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line.plain) + 1
    text = "\n".join(line.plain for line in lines)
    pattern = re.compile(r"\s+".join(re.escape(word) for word in query.split()), re.IGNORECASE)
    result = [line.copy() for line in lines]
    first = None
    for match in pattern.finditer(text):
        index = max(0, bisect_right(starts, match.start()) - 1)
        if first is None:
            first = index
        while index < len(lines) and starts[index] < match.end():
            start = max(0, match.start() - starts[index])
            end = min(len(lines[index].plain), match.end() - starts[index])
            if start < end:
                result[index].stylize("bold black on bright_cyan", start, end)
            index += 1
    return result, first


def reflow_position(source: list[Text], offset: int, target: list[Text]) -> int:
    """Keep the same passage visible when a transcript changes width.

    Wrapping and table borders change the physical rows but usually leave the
    letters/numbers in the transcript intact. Use that stream as the anchor.
    """
    def stream(lines):
        starts = []
        chunks = []
        position = 0
        for line in lines:
            starts.append(position)
            chunk = "".join(char for char in line.plain if char.isalnum())
            chunks.append(chunk)
            position += len(chunk)
        return "".join(chunks), starts

    if not source or not target:
        return 0
    original, source_starts = stream(source)
    changed, target_starts = stream(target)
    if not original or not changed:
        return min(len(target) - 1, offset * len(target) // len(source))
    anchor = source_starts[min(offset, len(source) - 1)]
    if original != changed:
        # Width-dependent tool cards may truncate some source. Find the next
        # passage where possible; otherwise retain proportional progress.
        excerpt = original[anchor:anchor + 64]
        match = changed.find(excerpt) if excerpt else -1
        anchor = match if match >= 0 else int(len(changed) * anchor / max(1, len(original)))
    return max(0, min(len(target) - 1, bisect_right(target_starts, anchor) - 1))


def pane_heading(label: str, width: int, *, active: bool, count: str = "") -> Text:
    heading = Text()
    heading.append("› " if active else "  ", style="cyan" if active else "dim")
    heading.append(label, style="bold underline" if active else "dim")
    heading.append(" ")
    if count and heading.cell_len + len(count) + 1 <= width:
        heading.append(" " * (width - heading.cell_len - len(count)) + count, style="dim")
    return fitted(heading, width)


def session_row(row: dict, title: str, width: int, *, selected: bool, focused: bool,
                marked: bool, selecting: bool, query: str, armed: bool) -> Text:
    background = "on #17333b" if focused else "on #15252b" if selected or marked else ""
    line = Text(style=background)
    line.append("› " if selected else "  ", style="bold cyan" if focused else "dim cyan")
    if selecting:
        line.append("[x] " if marked else "[ ] ", style="bold cyan" if marked else "dim")
    state = "[current]" if row.get("is_current") else "[archived]" if row.get("archived") else ""
    timestamp = row.get("time_str", "—")
    # At small widths, preserve the title and explicit state before the date.
    show_time = width >= 52
    tail = "  ".join(part for part in (state, timestamp if show_time else "") if part)
    tail_width = Text(tail).cell_len + (2 if tail else 0)
    title_width = max(1, width - line.cell_len - tail_width)
    line.append_text(fitted(highlighted(title, query, "bold red" if armed else "bold" if selected else ""), title_width, pad=True))
    if tail:
        line.append("  ")
        if state:
            line.append(state, style="green" if row.get("is_current") else "dim")
            if show_time:
                line.append("  ")
        if show_time:
            line.append(timestamp, style="dim")
    return fitted(line, width, pad=True)


def beside(left: list[Text], right: list[Text], left_width: int,
           right_width: int, height: int) -> list[Text]:
    result = []
    for index in range(height):
        # A neutral parent keeps each pane's base style local to its span.
        # Appending to the left Text itself would extend its background across
        # the separator, right pane, and terminal padding.
        line = Text(no_wrap=True, overflow="crop")
        line.append_text(fitted(left[index] if index < len(left) else "", left_width, pad=True))
        line.append(" │ ", style="dim")
        line.append_text(fitted(right[index] if index < len(right) else "", right_width, pad=True))
        result.append(line)
    return result
