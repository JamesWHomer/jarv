"""Keep literal, single-paragraph replies off the full Markdown import path."""

import re

from rich.text import Text
from .terminal_text import safe_terminal_text


# Intentionally narrow: anything capable of introducing Markdown, entities,
# indentation, line breaks or Unicode normalization uses Rich's parser.
_LITERAL_PARAGRAPH = re.compile(r"[A-Za-z][A-Za-z0-9 ,.!?:;'\-]*\Z")


class _LiteralParagraph:
    def __init__(self, content):
        self.content = content

    def __rich_console__(self, console, options):
        style = console.get_style("none", default="none")
        style += console.get_style("markdown.paragraph", default="none")
        text = Text(justify="left")
        text.append(self.content, style)
        yield from console.render(text, options.update(height=None))


def markdown_renderable(content: str):
    """Render plain paragraphs exactly like Markdown; delegate everything else."""
    content = safe_terminal_text(content)
    if content and content == content.rstrip() and _LITERAL_PARAGRAPH.fullmatch(content):
        return _LiteralParagraph(content)
    from rich.markdown import Markdown

    return Markdown(content)
