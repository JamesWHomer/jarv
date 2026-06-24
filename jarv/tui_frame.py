"""Pure frame-composition helpers for the heads-up TUI.

This module owns the geometry and layout math for the heads-up panel -- the part
that historically caused the WSL/ConPTY stale-edge visual bugs.
Every function here is side-effect free and depends only on its arguments, so the
behaviour can be unit-tested at fixed terminal sizes (see tests/test_tui_frame.py
and the golden-frame tests in tests/test_headsup.py).

The stateful event loop, locking, and animation timing remain in headsup.py; this
module never reads global state or the clock.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich import box
from rich.cells import cell_len
from rich.console import Console, Group, RenderableType
from rich.control import Control, ControlType
from rich.panel import Panel
from rich.segment import Segment
from rich.text import Text

from .display import TITLE_STYLE
from .tui_capabilities import supports_erase_eol
from .tui_layout import clip_text

_ERASE_TO_END_OF_LINE = Control((ControlType.ERASE_IN_LINE, 0)).segment

HEADSUP_TITLE_LABEL = "jarv ▸ heads-up"

# Rich reserves six cells around panel titles. Keeping the title inside that
# budget prevents Rich from expanding past our wrap guard.
_TITLE_CELL_BUDGET = 6


class EraseTrailingColumns:
    """Render a frame, clearing each row's stale right edge before drawing it.

    The frame spans the full terminal width, so its right border sits flush
    against the screen edge -- in the very last column. Before drawing each row
    we emit an erase-to-end-of-line control while the cursor is still at column
    0, which clears the entire physical row (including any stale right border a
    previous, wider frame left behind). Drawing the row afterwards repaints the
    border, so it survives.

    The erase must come *before* the row content. Emitted after a full-width
    line, the cursor sits in the last column in the terminal's pending-wrap
    state, and the erase would wipe the right border we just drew there -- the
    WSL/ConPTY redraw quirk this used to trip over. The pending-wrap latch
    itself needs no special handling: Rich's per-frame cursor-home and the
    newline between rows already reset it.
    """

    def __init__(self, renderable: RenderableType):
        self.renderable = renderable

    def __rich_console__(self, console: Console, options):
        lines = console.render_lines(
            self.renderable,
            options,
            pad=False,
            new_lines=False,
        )
        newline = Segment.line()
        erase_row = console.is_terminal and supports_erase_eol()
        for index, line in enumerate(lines):
            if index:
                yield newline
            if erase_row:
                yield _ERASE_TO_END_OF_LINE
            yield from line


def panel_width(terminal_width: int) -> int:
    """Width of a fullscreen panel: the full terminal width, flush to the edge.

    Shared by every fullscreen view (heads-up, settings, setup, the session
    browser) so their borders line up identically against the screen edge. The
    stale right-edge artifact this used to leave columns spare for is handled by
    :class:`EraseTrailingColumns` instead (see the module docstring).
    """
    return max(1, terminal_width)


@dataclass(frozen=True)
class FrameLayout:
    """Resolved geometry for one rendered frame."""

    term_w: int
    term_h: int
    panel_width: int
    inner_width: int
    body_height: int
    max_prompt_rows: int


def compute_layout(term_w: int, term_h: int) -> FrameLayout:
    """Derive all per-frame dimensions from the terminal size.

    Mirrors the historical inline math in HeadsupApp.render so behaviour is
    preserved exactly; the value is that it is now pure and testable.
    """
    term_w = max(20, term_w)
    term_h = max(8, term_h)
    pw = panel_width(term_w)
    inner_width = max(1, pw - 4)
    body_height = max(3, term_h - 2)
    max_prompt_rows = min(8, max(1, body_height - 2), max(3, term_h // 3))
    return FrameLayout(
        term_w=term_w,
        term_h=term_h,
        panel_width=pw,
        inner_width=inner_width,
        body_height=body_height,
        max_prompt_rows=max_prompt_rows,
    )


def transcript_rows(body_height: int, prompt_row_count: int) -> int:
    """Rows available for the transcript above the footer and prompt."""
    return max(1, body_height - prompt_row_count - 1)


def window_transcript(
    transcript: list[Text],
    rows: int,
    scroll_offset: int,
) -> tuple[list[Text], int]:
    """Clamp the scroll offset and slice the visible transcript window.

    Returns (visible_lines, clamped_scroll_offset). The offset counts lines from
    the bottom, so 0 pins the newest content to the prompt.
    """
    max_scroll = max(0, len(transcript) - rows)
    clamped = max(0, min(scroll_offset, max_scroll))
    end = len(transcript) - clamped
    start = max(0, end - rows)
    return list(transcript[start:end]), clamped


def compose_title(
    model_status: str,
    panel_width: int,
    *,
    left_label: str = HEADSUP_TITLE_LABEL,
) -> Text:
    """Build the panel title: left label, a rule, then the model status."""
    title_width = max(1, panel_width - _TITLE_CELL_BUDGET)
    title = Text(no_wrap=True, overflow="crop")
    left = clip_text(left_label, title_width)
    title.append(left, style=TITLE_STYLE)
    remaining = title_width - cell_len(left)
    if remaining <= 1:
        return title
    status = clip_text(model_status, remaining - 1)
    separator_width = title_width - cell_len(left) - cell_len(status) - 2
    if separator_width > 0:
        title.append(" ")
        title.append("─" * separator_width, style="cyan")
        title.append(" ")
    else:
        title.append(" ")
    title.append(status, style="dim")
    return title


def assemble_body(
    visible: list[RenderableType],
    footer: RenderableType,
    prompt_lines: list[RenderableType],
    body_height: int,
    rows: int,
) -> list[RenderableType]:
    """Pad/trim the visible transcript and stack footer + prompt beneath it."""
    parts: list[RenderableType] = list(visible)
    while len(parts) < rows:
        parts.insert(0, Text(""))
    target_rows_before_footer = max(0, body_height - 1 - len(prompt_lines))
    if len(parts) > target_rows_before_footer:
        del parts[target_rows_before_footer:]
    while len(parts) < target_rows_before_footer:
        parts.append(Text(""))
    parts.append(footer)
    parts.extend(prompt_lines)
    return parts


def build_frame(
    parts: list[RenderableType],
    *,
    title: Text,
    subtitle: Text,
    panel_width: int,
    term_h: int,
) -> RenderableType:
    """Wrap the composed body in the bordered panel and erase trailing columns."""
    panel = Panel(
        Group(*parts),
        title=title,
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(0, 1),
        width=panel_width,
        height=term_h,
    )
    return EraseTrailingColumns(panel)
