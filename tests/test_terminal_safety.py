"""Hostile content must stay data at every human terminal boundary."""

import io
import json
from unittest.mock import Mock

import pytest
from rich.console import Console
from rich.control import Control
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

from jarv import safety
from jarv.agent_ui import StreamingMarkdownPreview, TailMarkdown
from jarv.cli_output import CliOutput
from jarv.display import SafeConsole, command_line_renderable, output_renderable
from jarv.edit_tool import _diff_renderable
from jarv.headsup import _UserMessage, SafetyConfirmCard
from jarv.markdown_render import markdown_renderable
from jarv.session_render import _history_visual_lines, tool_call_card_from_args
from jarv.terminal_text import safe_terminal_text


PAYLOAD = "before\x1b]52;c;QUJD\x1b\\\x1b[2J\x1b[?1049l\x07\x08\x85\rafter"


def render(value, console_type=Console, color_system=None, **kwargs):
    stream = io.StringIO()
    console_type(file=stream, width=300, force_terminal=True,
                 color_system=color_system, legacy_windows=False, **kwargs).print(value)
    return stream.getvalue()


def assert_visible(text):
    assert "\x1b]52" not in text
    assert "\x1b[2J" not in text
    assert "\x1b[?1049l" not in text
    assert "\\x1b" in text
    assert "\\x07" in text
    assert "\\x08" in text
    assert "\\x85" in text
    assert "\\r" in text


def test_control_escaping_is_idempotent_and_preserves_ordinary_text():
    source = "hello 世界\r\n\t" + "".join(map(chr, range(256)))
    result = safe_terminal_text(source)
    assert result.startswith("hello 世界\n\t")
    assert safe_terminal_text(result) == result
    assert not any(ord(c) < 32 and c not in "\n\t" or 127 <= ord(c) < 160 for c in result)


@pytest.mark.parametrize("factory", [
    output_renderable,
    lambda text: command_line_renderable(text, expanded=True),
    markdown_renderable,
    lambda text: TailMarkdown(text, max_lines=10),
    _UserMessage,
    _diff_renderable,
    lambda text: safety._build_confirmation_body(text, "test reason"),
    lambda text: tool_call_card_from_args("run_command", {"command": text}, output=text),
])
def test_content_is_sanitized_before_rich_rendering(factory):
    assert_visible(render(factory(PAYLOAD)))


def test_saved_transcript_is_safe_without_modifying_records():
    history = [{"role": "user", "content": PAYLOAD},
               {"role": "assistant", "content": PAYLOAD}]
    before = json.dumps(history)
    assert_visible("\n".join(line.plain for line in _history_visual_lines(history, 300)))
    assert json.dumps(history) == before


@pytest.mark.parametrize("name,args,output", [
    ("read", {"input": PAYLOAD}, ""),
    ("edit", {"path": PAYLOAD, "old_text": "old", "new_text": "new"}, ""),
    ("web_search", {"query": PAYLOAD}, ""),
    ("web_search", {"query": "test"}, "1. " + PAYLOAD),
    ("spawn", {"children": [{"label": PAYLOAD}]}, ""),
    ("spawn", {"children": [{"label": "child"}]},
     json.dumps([{"label": "child", "status": "done", "tldr": PAYLOAD}])),
])
def test_tool_card_fields_do_not_lose_controls_before_rendering(name, args, output):
    before = json.dumps(args)
    assert_visible(render(tool_call_card_from_args(name, args, output=output)))
    assert json.dumps(args) == before


def test_auditor_reason_controls_are_visible_in_headsup_and_inline():
    state = {"done": True, "allow": False, "reason": PAYLOAD}
    card = SafetyConfirmCard(safety.ConfirmRequest(body=Text("script"), audit_state=state))
    assert_visible(render(card))
    assert_visible(render(safety._AuditPanel(Text("script"), 0, state, [])))


def test_streaming_preview_preserves_raw_text_and_escapes_split_controls():
    live = Mock()
    preview = StreamingMarkdownPreview(live, max_lines=10, refresh_interval=0)
    for part in ("before\x1b", PAYLOAD[len("before\x1b"):]):
        preview.append(part)
    preview.flush()
    assert preview.text == PAYLOAD
    assert_visible(render(live.update.call_args.args[0]))


def test_final_console_guard_sanitizes_segments_and_rejects_unsafe_links():
    class RawContent:
        def __rich_console__(self, console, options):
            yield Segment(PAYLOAD, Style(link="https://example.test/\x1b]52;bad\x07"))

    result = render(RawContent(), SafeConsole, color_system="standard")
    assert_visible(result)
    assert "https://example.test" not in result
    # Rich's UI controls and valid links still work.
    assert "\x1b[2J" in render(Control.clear(), SafeConsole)
    result = render(Text("link", style=Style(link="https://example.test/")), SafeConsole,
                    color_system="standard")
    assert "https://example.test/" in result


def test_console_string_diagnostics_escape_controls_before_rich_strips_them():
    assert_visible(render(PAYLOAD, SafeConsole))


@pytest.mark.parametrize("output_format", ["text", "json", "jsonl"])
def test_cli_human_output_safe_and_json_roundtrips(output_format):
    output = CliOutput(output_format, verbose=True)
    output.stdout = io.StringIO()
    output.stderr = io.StringIO()
    output.start_turn("query", {"provider": PAYLOAD, "model": "test"})
    output.event("delta", text=PAYLOAD)
    output.finish(text=PAYLOAD, error=PAYLOAD, session_id=PAYLOAD)
    assert_visible(output.stderr.getvalue())
    if output_format == "text":
        assert_visible(output.stdout.getvalue())
    else:
        records = list(map(json.loads, output.stdout.getvalue().splitlines()))
        assert records[-1]["text"] == PAYLOAD
        assert records[-1]["error"] == PAYLOAD
        if output_format == "jsonl":
            assert records[1]["text"] == PAYLOAD


def test_approval_preview_preserves_all_lines_and_execution_order():
    lines = ["", *[f"$value = {i}" for i in range(12)], "rm -rf cache", "",
             "Set-Location important", "$target = 'data'", "rm -rf $target", ""]
    command = "\n".join(lines)
    body = safety._build_confirmation_body(command, "deletion")
    shown = [part.plain for part in body.renderables][2:]
    assert shown == ["  " + line for line in lines]
    assert "hidden" not in render(body)


def test_inline_auditor_prints_full_script_before_live_status(monkeypatch):
    console = Mock()
    command = "\n".join(f"echo line-{i:03}" for i in range(100))
    body = safety._build_confirmation_body(command, "review")
    state = {"done": True, "allow": True, "reason": "safe"}

    class FakeLive:
        def __init__(self, value, **kwargs):
            self.value = value

        def __enter__(self):
            printed = "\n".join(render(call.args[0]) for call in console.print.call_args_list if call.args)
            assert "line-000" in printed and "line-099" in printed
            assert "line-000" not in render(self.value)
            return self

        def refresh(self):
            pass

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(safety, "console", console)
    monkeypatch.setattr(safety.sys, "platform", "win32")
    monkeypatch.setattr("rich.live.Live", FakeLive)
    assert safety._live_audit_poll(body, state)
