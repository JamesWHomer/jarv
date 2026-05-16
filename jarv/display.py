import re

from rich import box
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.rule import Rule

console = Console()

PANEL_BORDER_STYLE = "cyan"
ACCENT_STYLE = "bold cyan"
TITLE_STYLE = "bold bright_white"

STEP_DOT_DONE = "\u25cf"
STEP_DOT_ACTIVE = "\u25cf"
STEP_DOT_PENDING = "\u25cb"


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

DISPLAY_LINE_LIElastic License 2.0 = 30


def flatten_headings(text: str) -> str:
    return re.sub(r"^#{1,6}\s+(.+)$", r"**\1**", text, flags=re.MULTILINE)


def display_output(output: str) -> None:
    lines = output.splitlines()
    if len(lines) > DISPLAY_LINE_LIElastic License 2.0:
        console.print("\n".join(lines[:DISPLAY_LINE_LIElastic License 2.0]), style="dim", markup=False)
        hidden = len(lines) - DISPLAY_LINE_LIElastic License 2.0
        console.print(f"[dim italic]... {hidden} more lines hidden (full output sent to model)[/dim italic]")
    else:
        console.print(output, style="dim", markup=False)
