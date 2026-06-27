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


def _clip_head(value: str, width: int, *, ellipsis: str = "…") -> str:
    """Clip ``value`` to ``width`` cells by dropping its *head*.

    The opposite of :func:`clip_text`: the tail survives and a leading ellipsis
    marks the truncation. Used for the working-directory label so the most
    specific (rightmost) path segments stay readable when the panel is narrow.
    """
    if width <= 0:
        return ""
    if cell_len(value) <= width:
        return value
    if width <= cell_len(ellipsis):
        ellipsis = ""
    budget = width - cell_len(ellipsis)
    kept: list[str] = []
    used = 0
    for char in reversed(value):
        char_width = cell_len(char)
        if used + char_width > budget:
            break
        kept.append(char)
        used += char_width
    return ellipsis + "".join(reversed(kept))


def compose_subtitle(
    left_label: str,
    right: Text,
    panel_width: int,
    *,
    left_style: str = "dim",
) -> Text:
    """Build the panel's bottom bar: a left label, a rule, then a right status.

    The mirror of :func:`compose_title` for the bottom border. ``right`` (the
    usage status) is anchored to the right and preserved intact; ``left_label``
    (the working directory) is anchored to the left and is the first thing
    truncated -- from its head, so the trailing path segments survive -- as the
    panel narrows. The middle is filled with the same cyan rule as the title so
    the border reads as continuous.
    """
    bar_width = max(1, panel_width - _TITLE_CELL_BUDGET)
    bar = Text(no_wrap=True, overflow="crop")

    right = right.copy()
    right.truncate(bar_width, overflow="ellipsis")
    right_cells = cell_len(right.plain)

    # Reserve the status plus a one-cell gutter either side of the rule.
    left_budget = bar_width - right_cells - 3
    if not left_label or left_budget < 1:
        fill = bar_width - right_cells - 1
        if fill > 0:
            bar.append("─" * fill, style="cyan")
            bar.append(" ")
        bar.append_text(right)
        return bar

    left = _clip_head(left_label, left_budget)
    bar.append(left, style=left_style)
    separator_width = bar_width - cell_len(left) - right_cells - 2
    bar.append(" ")
    if separator_width > 0:
        bar.append("─" * separator_width, style="cyan")
    bar.append(" ")
    bar.append_text(right)
    return bar


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


# No gutter to the right of the menu block: the menu is a bordered box, so its
# own right edge already separates it from the starfield/transcript, which is
# free to show through immediately past the border.
_MENU_OVERLAY_GUTTER = 0


def overlay_menu(
    visible: list[Text],
    menu_lines: list[Text],
    *,
    rows: int,
    width: int,
    console: Console,
) -> list[Text]:
    """Composite the slash-command menu over the bottom of the body.

    The menu is painted as a left-aligned block flush above the prompt, so the
    transcript/starfield behind it keeps its full height instead of being pushed
    up when the menu opens. Cells under the block are replaced by the menu;
    cells to its right keep the background, so the starfield twinkles on beside
    the menu exactly as it does above it.

    ``visible`` is bottom-aligned into ``rows`` rows first (matching how
    :func:`assemble_body` pads it) so the menu anchors against the prompt.
    """
    if not menu_lines or rows <= 0:
        return visible
    body = [Text("") for _ in range(max(0, rows - len(visible)))] + list(visible)
    body = body[-rows:]
    block_width = max((cell_len(line.plain) for line in menu_lines), default=0)
    block_width = max(0, min(width, block_width + _MENU_OVERLAY_GUTTER))
    count = len(menu_lines)
    for offset, menu_line in enumerate(menu_lines):
        row_index = rows - count + offset
        if 0 <= row_index < len(body):
            body[row_index] = _overlay_line(
                body[row_index], menu_line, width, block_width, console
            )
    return body


def _overlay_line(
    background: Text,
    foreground: Text,
    width: int,
    block_width: int,
    console: Console,
) -> Text:
    """Replace the first ``block_width`` cells of ``background`` with ``foreground``.

    Cells from ``block_width`` to ``width`` keep the background, so whatever sits
    behind the menu (typically the starfield) shows through to its right.
    """
    merged = list(
        Segment.adjust_line_length(list(foreground.render(console)), block_width, pad=True)
    )
    if width > block_width:
        bg = Segment.adjust_line_length(list(background.render(console)), width, pad=True)
        merged.extend(list(Segment.divide(bg, [block_width, width]))[1])
    line = Text(no_wrap=True, overflow="crop")
    for segment in merged:
        if segment.text:
            line.append(segment.text, style=segment.style)
    return line


# The input field and its slash-command popup are drawn as one continuous box
# from these helpers. The border is a calm dark blue (dim cyan) so it frames the
# field without competing with the white draft text; every edge -- the popup top,
# the divider that joins the popup to the field, and the field's own sides and
# bottom -- shares this one style so the whole thing reads as a single seamless
# unit rather than two stacked boxes.
PROMPT_BOX_BORDER_STYLE = "dim cyan"


def box_top(width: int, *, style: str = PROMPT_BOX_BORDER_STYLE) -> Text:
    """The rounded top edge ``╭───╮`` of the box, spanning ``width`` cells."""
    return _box_edge(width, "╭", "╮", style)


def box_bottom(width: int, *, style: str = PROMPT_BOX_BORDER_STYLE) -> Text:
    """The rounded bottom edge ``╰───╯`` of the box, spanning ``width`` cells."""
    return _box_edge(width, "╰", "╯", style)


def box_divider(width: int, *, style: str = PROMPT_BOX_BORDER_STYLE) -> Text:
    """The interior rule ``├───┤`` that joins the popup to the input field.

    Its tees butt against the side borders above and below, so a popup (top edge
    + rows) and the field beneath it read as one box split by a single rule.
    """
    return _box_edge(width, "├", "┤", style)


def box_tab_top(width: int, tab_width: int, *, style: str = PROMPT_BOX_BORDER_STYLE) -> Text:
    """The input field's top edge with a narrower popup docked onto its left.

    Renders ``├────┴────────╮``: the left ``tab_width`` cells close the compact,
    left-aligned popup resting on top (``├`` down into the field on the left,
    ``┴`` under the popup's right border), and the remainder is the field's own
    rounded top edge. The two share this one line, so the popup reads as part of
    the same box rather than a separate panel floating above it. When the popup
    spans the full width this degrades to a plain ``├───┤`` divider.
    """
    tab_width = max(2, min(tab_width, width))
    if tab_width >= width:
        return box_divider(width, style=style)
    left = "├" + "─" * (tab_width - 2) + "┴"
    right = "─" * (width - tab_width - 1) + "╮"
    return _styled_line(left + right, style)


def _box_edge(width: int, left: str, right: str, style: str) -> Text:
    field_width = max(1, width - 2)
    return _styled_line(left + "─" * field_width + right, style)


def _styled_line(text: str, style: str) -> Text:
    # Carry the border colour as a span, not as the Text's base style: a base
    # style is applied by the parent renderable, but these edges are also painted
    # straight onto the body by the overlay (where ``Text.render`` drops a
    # span-less base style and the border would fall back to white). A span
    # survives both paths, so the edge keeps its colour wherever it's drawn.
    line = Text(text, no_wrap=True)
    line.stylize(style)
    return line


def box_row(content: Text, width: int, *, style: str = PROMPT_BOX_BORDER_STYLE) -> Text:
    """Frame one content line between the box's side borders.

    The content is padded with a one-cell gutter inside each border (dropped only
    when the box is too narrow for it) and clipped to fit, so every row in the
    box -- popup suggestion or input line -- lines its borders up exactly.
    """
    content_width = max(1, width - 4)
    has_gutter = width >= 4
    body = content.copy()
    body.truncate(content_width, overflow="crop")
    line = Text(no_wrap=True, overflow="crop")
    line.append("│", style=style)
    if has_gutter:
        line.append(" ")
    line.append_text(body)
    padding = max(0, content_width - cell_len(body.plain))
    if padding:
        line.append(" " * padding)
    if has_gutter:
        line.append(" ")
    line.append("│", style=style)
    return line


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
    return wrap_frame(panel)


def wrap_frame(renderable: RenderableType) -> RenderableType:
    """Wrap any full-screen renderable with the stale-edge erase.

    Shared by every alternate-screen view (heads-up, the tree/session browsers,
    settings, read-only scroll overlays) so they all clear a previous, wider
    frame's stale right border on WSL/ConPTY rather than only heads-up doing so.
    ``EraseTrailingColumns`` itself is a no-op on non-terminals and when
    ``supports_erase_eol()`` is disabled, so wrapping is always safe.
    """
    return EraseTrailingColumns(renderable)
