"""Output and argument semantics must survive the cold-start shortcuts."""

import io
import sys

import pytest
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

from jarv import cli
from jarv.markdown_render import markdown_renderable


@pytest.mark.parametrize("width", [1, 8, 40, 100])
@pytest.mark.parametrize("content", [
    "Hello!", "A long ordinary sentence with words that need to wrap.",
    "Let's try commas, repeated   spaces; numbers 123 and hyphen-words.",
    "x", " leading space", "trailing space ", "one\ntwo", "one\n\ntwo",
    "1. list item", "- list item", "**bold**", "a_b", "a &amp; b",
    "a <b>HTML</b>", "a [link](https://example.com)", "`code`", "```py\nx\n```",
    "Unicode: café 日本語", "", "Hello\x00there", "a\\*b",
])
def test_markdown_shortcut_preserves_segments(content, width):
    console = Console(file=io.StringIO(), width=width, force_terminal=True,
                      theme=Theme({"markdown.paragraph": "bold cyan"}))
    options = console.options.update(height=3)
    # Compare rendered text, colour/style and line boundaries, not just strings.
    actual = list(console.render(markdown_renderable(content), options))
    # Unsafe controls are now visible escapes at the display boundary.
    from jarv.terminal_text import safe_terminal_text
    expected = list(console.render(Markdown(safe_terminal_text(content)), options))
    assert actual == expected


@pytest.mark.parametrize("arguments", [
    ["/help"], ["/HELP"], ["help"], ["/set", "model", "a model"],
    ["/unknown"], ["/history", "unexpected"],
])
def test_option_free_commands_skip_parser_but_keep_arguments(monkeypatch, arguments):
    seen = []
    monkeypatch.setattr(sys, "argv", ["jarv", *arguments])
    monkeypatch.setattr(cli, "_command_entry", seen.append)
    monkeypatch.setattr(cli, "_build_parser", lambda: pytest.fail("unneeded parser"))
    cli.main()
    assert seen == [arguments]


@pytest.mark.parametrize("arguments, status", [
    (["/help", "--quiet"], 2), (["/help", "--nonsense"], 2),
    (["--model", "--version"], 2), (["--help"], 0),
])
def test_options_preserve_argparse_semantics(monkeypatch, arguments, status):
    monkeypatch.setattr(sys, "argv", ["jarv", *arguments])
    monkeypatch.setattr(cli, "_command_entry", lambda _: pytest.fail("bypassed parser"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == status


def test_version_does_not_load_parser(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["jarv", "--version"])
    monkeypatch.setattr(cli, "_build_parser", lambda: pytest.fail("unneeded parser"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 0
    assert capsys.readouterr().out == f"jarv {cli.__version__}\n"
