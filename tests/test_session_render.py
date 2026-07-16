import io
import json

import pytest
from rich.console import Console

from jarv import display
from jarv.config import DEFAULT_CONFIG, validate_config
from jarv.session_render import ToolCallCard, tool_call_card, tool_call_card_from_args


@pytest.fixture(autouse=True)
def _reset_display_lines():
    yield
    display.configure_output_display_lines("auto")


def _render(renderable, width: int = 200) -> str:
    stream = io.StringIO()
    Console(file=stream, force_terminal=False, color_system=None, width=width).print(
        renderable
    )
    return stream.getvalue()


def _long_command_card() -> ToolCallCard:
    command = "\n".join(f"print({number})  # script line" for number in range(40))
    return tool_call_card_from_args("run_command", {"command": command})


def test_run_command_card_clips_long_inline_scripts():
    rendered = _render(_long_command_card())

    assert "> print(0)" in rendered
    assert "print(2)" in rendered
    assert "print(3)" not in rendered
    assert "… 37 more lines" in rendered


def test_run_command_card_clips_output_to_display_budget():
    display.configure_output_display_lines(6)
    output = "\n".join(f"output line {number}" for number in range(40))
    card = tool_call_card_from_args(
        "run_command", {"command": "python gen.py"}, output=output
    )

    rendered = _render(card)

    assert "output line 0" in rendered
    assert "output line 39" in rendered
    assert "output line 20" not in rendered
    assert "lines hidden" in rendered


def test_tool_call_card_expands_to_full_command_and_output():
    display.configure_output_display_lines(6)
    output = "\n".join(f"output line {number}" for number in range(40))
    card = tool_call_card_from_args(
        "run_command",
        {"command": "\n".join(f"print({n})" for n in range(40))},
        output=output,
    )
    assert card.expandable
    card.expanded = True

    rendered = _render(card)

    assert "print(39)" in rendered
    assert "output line 20" in rendered
    assert "lines hidden" not in rendered
    assert "more lines" not in rendered


def test_generic_tool_card_caps_raw_arguments():
    item = {
        "name": "mystery_tool",
        "arguments": json.dumps({"payload": "y" * 5000}, ensure_ascii=True),
    }
    card = tool_call_card(item)

    rendered = _render(card, width=6000)

    assert "y" * 5000 not in rendered
    assert "… +" in rendered and "chars" in rendered

    card.expanded = True
    assert "y" * 5000 in _render(card, width=6000)


def test_read_card_still_summarizes_without_leaking_content():
    output = (
        "[READ RESULT]\nReturned size: 120\nTotal size: 120\nEOF: true\n\nsecret body"
    )
    rendered = _render(tool_call_card_from_args("read", {"input": "notes.txt"}, output=output))

    assert "notes.txt" in rendered
    assert "120 chars" in rendered
    assert "secret body" not in rendered


def test_read_card_expanded_shows_the_read_content():
    output = (
        "[READ RESULT]\nReturned size: 120\nTotal size: 120\nEOF: true\n\n"
        "line one of the page\nline two of the page"
    )
    card = tool_call_card_from_args("read", {"input": "notes.txt"}, output=output)
    card.expanded = True

    rendered = _render(card)

    assert "120 chars" in rendered
    assert "line one of the page" in rendered
    assert "line two of the page" in rendered


def test_read_card_expanded_skips_image_content():
    output = (
        "[READ RESULT]\nImage media type: image/png\nImage bytes: 2048\n\n"
        "base64junk=="
    )
    card = tool_call_card_from_args("read", {"input": "logo.png"}, output=output)
    card.expanded = True

    rendered = _render(card)

    assert "image image/png" in rendered
    assert "base64junk" not in rendered


def test_web_search_card_expanded_shows_full_results():
    output = "\n".join(f"{n}. Result title {n}\n  https://example.com/{n}" for n in range(1, 8))
    card = tool_call_card_from_args("web_search", {"query": "jarv"}, output=output)

    collapsed = _render(card)
    assert "7 results" in collapsed
    assert "Result title 7" not in collapsed

    card.expanded = True
    expanded = _render(card)
    assert "Result title 7" in expanded
    assert "https://example.com/7" in expanded


def test_edit_card_expanded_shows_full_snippets():
    old_text = "\n".join(f"old line {n}" for n in range(10))
    new_text = "\n".join(f"new line {n}" for n in range(10))
    card = tool_call_card_from_args(
        "edit", {"path": "a.py", "old_text": old_text, "new_text": new_text}
    )

    collapsed = _render(card)
    assert "old line 9" not in collapsed
    assert "… 7 more lines" in collapsed

    card.expanded = True
    expanded = _render(card)
    assert "- old line 9" in expanded
    assert "+ new line 9" in expanded
    assert "more lines" not in expanded


def test_display_lines_config_validates_auto_and_integers():
    config = {**DEFAULT_CONFIG, "tool_output_display_lines": "auto"}
    assert validate_config(config)
    assert config["tool_output_display_lines"] == "auto"

    config = {**DEFAULT_CONFIG, "tool_output_display_lines": "15"}
    assert validate_config(config)
    assert config["tool_output_display_lines"] == 15

    for invalid in (0, 2, "abc", True):
        config = {**DEFAULT_CONFIG, "tool_output_display_lines": invalid}
        assert not validate_config(config)
