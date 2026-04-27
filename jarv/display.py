import re

from rich.console import Console

console = Console()

DISPLAY_LINE_LIElastic License 2.0 = 30


def flatten_headings(text: str) -> str:
    return re.sub(r"^#{1,6}\s+(.+)$", r"**\1**", text, flags=re.MULTILINE)


def display_output(output: str) -> None:
    lines = output.splitlines()
    if len(lines) > DISPLAY_LINE_LIElastic License 2.0:
        console.print("\n".join(lines[:DISPLAY_LINE_LIElastic License 2.0]), style="dim")
        hidden = len(lines) - DISPLAY_LINE_LIElastic License 2.0
        console.print(f"[dim italic]... {hidden} more lines hidden (full output sent to model)[/dim italic]")
    else:
        console.print(output, style="dim")
