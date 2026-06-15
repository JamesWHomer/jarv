import os
import re
import signal
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

console = Console()

_live_display_depth = threading.local()

PANEL_BORDER_STYLE = "cyan"
ACCENT_STYLE = "bold cyan"
TITLE_STYLE = "bold bright_white"

TOOL_CARD_STYLES = {
    "run_command": (">", "Command", "yellow"),
    "web_search": ("\u2315", "Web search", "green"),
    "read": ("\u2261", "Read", "cyan"),
    "spawn": ("\u21b3", "Subagent", "magenta"),
    "ask_user": ("?", "Ask user", "blue"),
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

RESIZE_REFRESH_INTERVAL = 0.02
RESIZE_ACTIVE_INTERVAL = 0.1

STEP_DOT_DONE = "\u25cf"
STEP_DOT_ACTIVE = "\u25cf"
STEP_DOT_PENDING = "\u25cb"


def terminal_size(*, console: Console = console) -> tuple[int, int]:
    """Return current terminal dimensions as (width, height)."""
    for stream in (0, 1, 2):
        try:
            size = os.get_terminal_size(stream)
        except OSError:
            continue
        return max(1, size.columns), max(1, size.lines)

    size = console.size
    return max(1, size.width), max(1, size.height)


def jarv_panel(body: RenderableType, title: str, subtitle: str | None = None, padding: tuple = (1, 2)) -> Panel:
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
    """Return how many nested Rich Live displays are active on this thread."""
    return int(getattr(_live_display_depth, "depth", 0) or 0)


@contextmanager
def track_live_display():
    """Increment live-display depth for the current thread."""
    depth = live_display_depth()
    _live_display_depth.depth = depth + 1
    try:
        yield
    finally:
        _live_display_depth.depth = depth


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

    header = ToolCardHeader(
        title,
        metadata_text,
        status_text if display_mode == "fullscreen" else Text(),
    )
    content = Group(header, body)
    if display_mode == "fullscreen":
        return Panel(
            content,
            border_style="bright_black",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    return ToolCard(accent, content)


@contextmanager
def refresh_on_resize(
    live,
    *,
    console: Console = console,
    interval: float = RESIZE_REFRESH_INTERVAL,
    active_interval: float = RESIZE_ACTIVE_INTERVAL,
    on_change=None,
):
    """Refresh a Rich Live display when the terminal dimensions change.

    ``on_change`` lets callers substitute a custom repaint (e.g. a full-screen
    hard repaint for inline views) for the default ``live.refresh()``.
    """
    stop = threading.Event()
    changed = threading.Event()
    last_size = terminal_size(console=console)
    previous_sigwinch = None
    restore_sigwinch = False

    def _repaint() -> None:
        if on_change is not None:
            on_change()
        else:
            live.refresh()

    def _watch() -> None:
        nonlocal last_size
        next_interval = interval

        while not stop.is_set():
            deadline = time.monotonic() + next_interval
            while not stop.is_set():
                signaled = changed.wait(max(0.0, deadline - time.monotonic()))
                changed.clear()
                if not signaled or next_interval == interval:
                    break
            if stop.is_set():
                break

            current_size = terminal_size(console=console)
            if current_size != last_size:
                last_size = current_size
                _repaint()
                next_interval = active_interval
            else:
                next_interval = interval

    if hasattr(signal, "SIGWINCH"):
        try:
            previous_sigwinch = signal.getsignal(signal.SIGWINCH)

            def _handle_sigwinch(signum, frame):
                changed.set()
                if callable(previous_sigwinch):
                    previous_sigwinch(signum, frame)

            signal.signal(signal.SIGWINCH, _handle_sigwinch)
            restore_sigwinch = True
        except (ValueError, OSError):
            restore_sigwinch = False

    thread = threading.Thread(target=_watch, name="jarv-resize-refresh", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        changed.set()
        thread.join(timeout=max(0.1, interval * 2))
        if restore_sigwinch:
            try:
                signal.signal(signal.SIGWINCH, previous_sigwinch)
            except (ValueError, OSError):
                pass

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


def output_renderable(output: str, *, max_lines: int | None = None) -> RenderableType:
    lines = output.splitlines()
    line_limit = (
        output_display_line_limit(console=console)
        if max_lines is None
        else max(DISPLAY_MIN_LINE_LIMIT, int(max_lines))
    )
    if len(lines) > line_limit:
        head_lines, tail_lines = output_display_split(line_limit)
        hidden = len(lines) - head_lines - tail_lines
        return Group(
            Text("\n".join(lines[:head_lines]), style="dim"),
            Text(
                f"... {hidden} lines omitted from the middle ...",
                style="dim italic",
            ),
            Text("\n".join(lines[-tail_lines:]), style="dim"),
        )
    return Text(output, style="dim")


def display_output(output: str, *, max_lines: int | None = None) -> None:
    console.print(output_renderable(output, max_lines=max_lines))
