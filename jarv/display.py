import os
import re
import sys
import threading
import time
from contextlib import contextmanager

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

def _truecolor_color_system() -> str | None:
    """Return ``"truecolor"`` when the terminal renders 24-bit colour, else ``None``.

    Rich auto-detects truecolor from ``COLORTERM``, but several terminals that do
    render it never set that variable -- most visibly WSL2, where the distro's
    ``TERM`` is ``xterm-256color`` so Rich quantises to the 256-colour cube. That
    both shifts the heads-up intro's hand-tuned gradients off-hue and makes them
    shimmer, because each animation frame's slightly different RGB can snap to a
    different cube entry. Detect the known-good cases and ask for truecolor;
    return ``None`` everywhere else so Rich keeps its own auto-detection (and a
    genuine 256-colour terminal isn't handed sequences it can't render).
    """
    colorterm = os.environ.get("COLORTERM", "").lower()
    if "truecolor" in colorterm or "24bit" in colorterm:
        return "truecolor"
    if os.environ.get("TERM", "").endswith("-direct"):
        return "truecolor"
    # WSL's terminals (Windows Terminal, VS Code) all render truecolor even
    # though the distro advertises only 256 colours via TERM.
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return "truecolor"
    return None


def _make_console() -> Console:
    forced = _truecolor_color_system()
    return Console(color_system=forced) if forced else Console()


console = _make_console()

# Process-wide (not per-thread): a Rich Live owned by one thread must be
# visible to workers on other threads deciding whether they can start their
# own Live on the shared console (Rich allows only one per console).
_live_display_depth = 0
_live_display_depth_lock = threading.Lock()
_first_paint_marks: set[str] = set()

PANEL_BORDER_STYLE = "cyan"
ACCENT_STYLE = "bold cyan"
TITLE_STYLE = "bold bright_white"

TOOL_CARD_STYLES = {
    "run_command": (">", "Command", "yellow"),
    "web_search": ("\u2315", "Web search", "green"),
    "read": ("\u2261", "Read", "cyan"),
    "edit": ("\u00b1", "Edit", "bright_yellow"),
    "spawn": ("\u21b3", "Subagent", "magenta"),
    "ask_user": ("?", "Ask user", "blue"),
    "safety": ("\u26a0", "Safety", "yellow"),
}


class ToolCardHeader:
    """Render a tool label and status at opposite ends of one row."""

    def __init__(self, title: Text, metadata: Text, status: Text):
        self.title = title
        self.metadata = metadata
        self.status = status

    def __rich_console__(self, console, options):
        width = max(1, options.max_width)
        left = self.title.copy()
        if self.metadata:
            left.append("  ")
            left.append_text(self.metadata)
        status = self.status.copy()
        if not status:
            yield left
            return
        gap = max(1, width - left.cell_len - status.cell_len)
        if left.cell_len + status.cell_len + 1 > width:
            left.truncate(max(1, width - status.cell_len - 1), overflow="ellipsis")
            gap = 1
        line = Text(no_wrap=True, overflow="crop")
        line.append_text(left)
        line.append(" " * gap)
        line.append_text(status)
        yield line


class ToolCard:
    """Render a compact tool block with a quiet colored left rail."""

    def __init__(self, accent: str, content: RenderableType):
        self.accent = Style.parse(accent)
        self.content = content

    def __rich_console__(self, console, options):
        inner_options = options.update(width=max(1, options.max_width - 2))
        lines = console.render_lines(
            self.content,
            inner_options,
            pad=False,
            new_lines=False,
        )
        for line in lines:
            yield Segment("\u258e ", self.accent)
            yield from line
            yield Segment.line()

STEP_DOT_DONE = "\u25cf"
STEP_DOT_ACTIVE = "\u25cf"
STEP_DOT_PENDING = "\u25cb"


def terminal_size(*, console: Console = console) -> tuple[int, int]:
    """Return current terminal dimensions as (width, height)."""
    for stream in (1, 2, 0):
        try:
            size = os.get_terminal_size(stream)
        except OSError:
            continue
        return max(1, size.columns), max(1, size.lines)

    size = console.size
    return max(1, size.width), max(1, size.height)


def mark_first_paint(label: str) -> None:
    """Emit a benchmark timestamp when first-paint instrumentation is enabled."""
    if os.environ.get("JARV_BENCH_FIRST_PAINT") != "1":
        return
    if label in _first_paint_marks:
        return
    _first_paint_marks.add(label)
    print(f"JARV_FIRST_PAINT {label} {time.time_ns()}", file=sys.stderr, flush=True)


def jarv_panel(
    body: RenderableType,
    title: str,
    subtitle: str | None = None,
    *,
    padding: tuple[int, int] = (1, 2),
    width: int | None = None,
    height: int | None = None,
) -> Panel:
    """Return a Panel using the shared jarv aesthetic."""
    return Panel(
        body,
        title=f"[{TITLE_STYLE}]jarv \u25b8 {title}[/{TITLE_STYLE}]",
        title_align="left",
        subtitle=f"[dim]{subtitle}[/dim]" if subtitle else None,
        subtitle_align="right",
        border_style=PANEL_BORDER_STYLE,
        box=box.ROUNDED,
        padding=padding,
        width=width,
        height=height,
    )


def section_rule(label: str, step: int | None = None, total: int | None = None) -> Rule:
    if step is not None and total is not None:
        dots = []
        for i in range(1, total + 1):
            if i < step:
                dots.append(f"[green]{STEP_DOT_DONE}[/green]")
            elif i == step:
                dots.append(f"[bold cyan]{STEP_DOT_ACTIVE}[/bold cyan]")
            else:
                dots.append(f"[bright_black]{STEP_DOT_PENDING}[/bright_black]")
        progress = " ".join(dots)
        title_text = f"[{ACCENT_STYLE}]{label}[/{ACCENT_STYLE}]  {progress}"
    else:
        title_text = f"[{ACCENT_STYLE}]{label}[/{ACCENT_STYLE}]"
    return Rule(title=title_text, style="bright_black", align="left")


def live_display_depth() -> int:
    """Return how many tracked Rich Live displays are active process-wide."""
    with _live_display_depth_lock:
        return _live_display_depth


@contextmanager
def track_live_display():
    """Increment the process-wide live-display depth while the block runs."""
    global _live_display_depth
    with _live_display_depth_lock:
        _live_display_depth += 1
    try:
        yield
    finally:
        with _live_display_depth_lock:
            _live_display_depth -= 1


def rendered_text_lines(renderable: RenderableType, width: int) -> list[Text]:
    """Render a Rich object to wrapped Text lines at the given width."""
    options = console.options.update(width=max(1, width))
    rendered = console.render_lines(renderable, options, pad=False)
    lines: list[Text] = []
    for rendered_line in rendered:
        line = Text(no_wrap=True, overflow="crop")
        for segment in rendered_line:
            if segment.text:
                line.append(segment.text, style=segment.style)
        lines.append(line)
    return lines


def status_line(prefix: str, message: str, prefix_style: str = "bold cyan", message_style: str = "") -> str:
    """Format a single-line status message with a colored prefix glyph."""
    if message_style:
        return f"[{prefix_style}]{prefix}[/{prefix_style}] [{message_style}]{message}[/{message_style}]"
    return f"[{prefix_style}]{prefix}[/{prefix_style}] {message}"


def tool_card(
    tool_name: str,
    body: RenderableType,
    *,
    metadata: str = "",
    status: str = "done",
    status_style: str = "green",
    display_mode: str = "print",
) -> RenderableType:
    """Return the shared compact card used for root tool calls."""
    icon, label, accent = TOOL_CARD_STYLES.get(
        tool_name,
        ("\u2022", tool_name.replace("_", " ").title(), "cyan"),
    )
    title = Text()
    title.append(f"{icon} ", style=f"bold {accent}")
    title.append(label, style=f"bold {accent}")

    metadata_text = Text(metadata, style="dim")
    status_text = Text(justify="right", style="dim")
    if status:
        if status_style == "green":
            status_text.append("\u2713 ", style="bold green")
        elif status_style == "blue":
            status_text.append("\u25cf ", style="blue")
        elif status_style == "red":
            status_text.append("\u2717 ", style="bold red")
        else:
            status_text.append("\u25cf ", style=status_style)
        status_text.append(status)

    # Print mode stays quiet on success (the default state) but still signals
    # failures and in-flight work; fullscreen always shows the pill.
    show_status = bool(status) and (
        display_mode == "fullscreen" or status_style != "green"
    )
    header = ToolCardHeader(
        title,
        metadata_text,
        status_text if show_status else Text(),
    )
    content = Group(header, body)
    if display_mode == "fullscreen":
        return Panel(
            content,
            border_style="bright_black",
            box=box.ROUNDED,
            safe_box=False,
            padding=(0, 1),
        )
    return ToolCard(accent, content)


DISPLAY_HEIGHT_RATIO = 3
DISPLAY_MIN_LINE_LIMIT = 3


def flatten_headings(text: str) -> str:
    return re.sub(r"^#{1,6}\s+(.+)$", r"**\1**", text, flags=re.MULTILINE)


def output_display_line_limit(*, console: Console = console) -> int:
    _, terminal_height = terminal_size(console=console)
    return max(DISPLAY_MIN_LINE_LIMIT, terminal_height // DISPLAY_HEIGHT_RATIO)


def output_display_split(line_limit: int) -> tuple[int, int]:
    visible_lines = max(2, line_limit - 1)
    tail_lines = max(1, visible_lines // 3)
    head_lines = visible_lines - tail_lines
    return head_lines, tail_lines


def hidden_lines_hint(hidden: int, *, where: str = "middle", suffix: str = "") -> Text:
    """The one dim-italic hint used everywhere display output is cropped.

    ``where`` names which part of the content was dropped relative to what is
    shown: ``"middle"`` (head+tail kept), ``"above"`` (tail kept), or
    ``"below"`` (head kept).
    """
    plural = "s" if hidden != 1 else ""
    if where == "above":
        message = f"↑ {hidden} earlier line{plural} hidden"
    elif where == "below":
        message = f"… {hidden} more line{plural}"
    else:
        message = f"… {hidden} line{plural} hidden …"
    if suffix:
        message += f" — {suffix}"
    return Text(message, style="dim italic")


def clip_middle(lines: list, line_limit: int) -> tuple[list, list, int]:
    """Split ``lines`` into (head, tail, hidden) keeping both ends."""
    if len(lines) <= line_limit:
        return lines, [], 0
    head_count, tail_count = output_display_split(line_limit)
    hidden = len(lines) - head_count - tail_count
    return lines[:head_count], lines[-tail_count:], hidden


def clip_tail(lines: list, line_limit: int) -> tuple[list, int]:
    """Split ``lines`` into (tail, hidden) keeping only the newest lines."""
    if len(lines) <= line_limit:
        return lines, 0
    hidden = len(lines) - line_limit
    return lines[hidden:], hidden


def output_renderable(output: str, *, max_lines: int | None = None) -> RenderableType:
    lines = output.splitlines()
    line_limit = (
        output_display_line_limit(console=console)
        if max_lines is None
        else max(DISPLAY_MIN_LINE_LIMIT, int(max_lines))
    )
    head, tail, hidden = clip_middle(lines, line_limit)
    if hidden > 0:
        return Group(
            Text("\n".join(head), style="dim"),
            hidden_lines_hint(hidden),
            Text("\n".join(tail), style="dim"),
        )
    return Text(output, style="dim")


def display_output(output: str, *, max_lines: int | None = None) -> None:
    console.print(output_renderable(output, max_lines=max_lines))
