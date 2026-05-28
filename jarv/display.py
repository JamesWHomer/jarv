import os
import re
import signal
import threading
from contextlib import contextmanager

from rich import box
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.rule import Rule

console = Console()

PANEL_BORDER_STYLE = "cyan"
ACCENT_STYLE = "bold cyan"
TITLE_STYLE = "bold bright_white"

RESIZE_REFRESH_INTERVAL = 0.1

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


def status_line(prefix: str, message: str, prefix_style: str = "bold cyan", message_style: str = "") -> str:
    """Format a single-line status message with a colored prefix glyph."""
    if message_style:
        return f"[{prefix_style}]{prefix}[/{prefix_style}] [{message_style}]{message}[/{message_style}]"
    return f"[{prefix_style}]{prefix}[/{prefix_style}] {message}"


@contextmanager
def refresh_on_resize(live, *, console: Console = console, interval: float = RESIZE_REFRESH_INTERVAL):
    """Refresh a Rich Live display when the terminal dimensions change."""
    stop = threading.Event()
    changed = threading.Event()
    last_size = terminal_size(console=console)
    previous_sigwinch = None
    restore_sigwinch = False

    def _watch() -> None:
        nonlocal last_size
        while not stop.is_set():
            changed.wait(interval)
            changed.clear()
            if stop.is_set():
                break
            current_size = terminal_size(console=console)
            if current_size == last_size:
                continue
            last_size = current_size
            live.refresh()

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

DISPLAY_LINE_LIMIT = 30


def flatten_headings(text: str) -> str:
    return re.sub(r"^#{1,6}\s+(.+)$", r"**\1**", text, flags=re.MULTILINE)


def display_output(output: str) -> None:
    lines = output.splitlines()
    if len(lines) > DISPLAY_LINE_LIMIT:
        console.print("\n".join(lines[:DISPLAY_LINE_LIMIT]), style="dim", markup=False)
        hidden = len(lines) - DISPLAY_LINE_LIMIT
        console.print(f"[dim italic]... {hidden} more lines hidden[/dim italic]")
    else:
        console.print(output, style="dim", markup=False)
