import json
from types import SimpleNamespace

from conftest import make_console

from jarv import session_commands, session_store


def test_history_visual_lines_include_tool_calls():
    history = [
        {"role": "user", "content": "Check the repository"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "run_command",
            "arguments": '{"command":"git status --short","timeout":10}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "working tree clean",
        },
        {"role": "assistant", "content": "The repository is clean."},
    ]

    lines, anchors = session_commands._history_visual_lines_and_anchors(history, 100)
    rendered = "\n".join(line.plain for line in lines)

    assert "> Command" in rendered
    assert "\u2713 done" in rendered
    assert "> git status --short" in rendered
    assert "working tree clean" in rendered
    assert "\u256d" in rendered
    assert rendered.index("jarv:") < rendered.index("> Command")
    assert rendered.count("jarv:") == 1
    assert len(anchors) == 3


def test_history_visual_lines_include_status_records():
    history = [
        {"role": "user", "content": "Hello"},
        {
            "type": "status",
            "phase": "response",
            "content": "Started responding in 1.0 second.",
        },
        {"role": "assistant", "content": "Hi."},
    ]

    lines, anchors = session_commands._history_visual_lines_and_anchors(history, 100)
    rendered = "\n".join(line.plain for line in lines)

    assert "Started responding in 1.0 second." in rendered
    assert rendered.index("jarv:") < rendered.index("Started responding")
    assert rendered.index("Started responding") < rendered.index("Hi.")
    assert len(anchors) == 2


def test_history_visual_lines_include_malformed_tool_arguments():
    history = [
        {
            "type": "function_call",
            "name": "run_command",
            "arguments": '{"command":"unfinished"',
        }
    ]

    lines = session_commands._history_visual_lines(history, 100)

    rendered = "\n".join(line.plain for line in lines)
    assert '{"command":"unfinished"' in rendered
    assert "\u2717 failed" in rendered


def test_history_visual_lines_render_read_like_fullscreen_tool_card():
    history = [
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "read",
            "arguments": '{"input":"README.md","offset":10,"size":50}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": "file contents are not shown in the live card",
        },
    ]

    rendered = "\n".join(
        line.plain for line in session_commands._history_visual_lines(history, 100)
    )

    assert "\u2261 Read" in rendered
    assert "README.md" in rendered
    assert "offset 10  \u2022  size 50" in rendered
    assert "file contents are not shown" not in rendered


def test_history_visual_lines_group_multiple_tool_calls_under_one_jarv_heading():
    history = [
        {"role": "user", "content": "Use two tools"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "web_search",
            "arguments": '{"query":"OpenAI"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "results",
        },
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "read",
            "arguments": '{"input":"README.md"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": "contents",
        },
        {"role": "assistant", "content": "Done."},
    ]

    rendered = "\n".join(
        line.plain for line in session_commands._history_visual_lines(history, 100)
    )

    assert rendered.count("jarv:") == 1
    assert rendered.index("jarv:") < rendered.index("\u2315 Web search")
    assert rendered.index("\u2315 Web search") < rendered.index("\u2261 Read")
    assert rendered.index("\u2261 Read") < rendered.index("Done.")
    web_search_bottom = next(
        index
        for index, line in enumerate(rendered.splitlines())
        if line.startswith("\u2570") and index > rendered.splitlines().index("jarv:")
    )
    assert rendered.splitlines()[web_search_bottom + 1].startswith("\u256d")


def test_history_visual_lines_ignore_hidden_records_between_tool_calls():
    history = [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "web_search",
            "arguments": '{"query":"OpenAI"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "results",
        },
        {"type": "reasoning", "summary": "Choose the next tool."},
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "read",
            "arguments": '{"input":"README.md"}',
        },
    ]

    rendered_lines = [
        line.plain for line in session_commands._history_visual_lines(history, 100)
    ]
    first_card_bottom = rendered_lines.index(
        next(line for line in rendered_lines if line.startswith("\u2570"))
    )

    assert rendered_lines[first_card_bottom + 1].startswith("\u256d")


def test_session_row_widths_preserve_message_space_on_small_screens():
    assert session_commands._session_row_widths(40) == (13, 7, 16)
    assert session_commands._session_row_widths(58) == (28, 7, 19)


def test_session_row_widths_give_extra_space_to_message_on_large_screens():
    assert session_commands._session_row_widths(110) == (28, 7, 71)


# --- command paths (cmd_archive / cmd_history) ----------------------------- #

def _setup_session(tmp_path, monkeypatch, history_items):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    history_file = sessions_dir / "history-abc.json"
    if history_items is not None:
        history_file.write_text(json.dumps(history_items), encoding="utf-8")
    monkeypatch.setattr(
        session_commands,
        "prepare_session_context",
        lambda: SimpleNamespace(history_file=history_file),
    )
    console, output = make_console()
    monkeypatch.setattr(session_commands, "console", console)
    return history_file, output


def test_cmd_archive_moves_history_and_sidecars(tmp_path, monkeypatch):
    history_file, output = _setup_session(
        tmp_path, monkeypatch, [{"role": "user", "content": "hello"}]
    )
    archive_dir = tmp_path / "archive"
    monkeypatch.setattr(session_store, "ARCHIVE_DIR", archive_dir)
    forgotten = []
    monkeypatch.setattr(
        session_commands, "forget_current_session", lambda: forgotten.append(True)
    )

    session_commands.cmd_archive()

    assert not history_file.exists()
    assert list(archive_dir.glob("history-*.json"))
    assert forgotten == [True]
    rendered = output.getvalue()
    assert "Session archived to" in rendered
    assert "New session starts" in rendered


def test_cmd_archive_without_history_prints_nothing_to_archive(tmp_path, monkeypatch):
    _, output = _setup_session(tmp_path, monkeypatch, None)
    monkeypatch.setattr(session_store, "ARCHIVE_DIR", tmp_path / "archive")
    monkeypatch.setattr(session_commands, "forget_current_session", lambda: None)

    session_commands.cmd_archive()

    assert "No history to archive" in output.getvalue()


def test_cmd_history_renders_plain_transcript_when_not_a_tty(tmp_path, monkeypatch):
    _, output = _setup_session(
        tmp_path,
        monkeypatch,
        [
            {"role": "user", "content": "check the repo"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "run_command",
                "arguments": '{"command":"git status"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "clean"},
            {"role": "assistant", "content": "All clean."},
        ],
    )

    session_commands.cmd_history()

    rendered = output.getvalue()
    assert "You" in rendered
    assert "check the repo" in rendered
    assert "git status" in rendered
    assert "All clean." in rendered
    assert "1 exchange(s)" in rendered


def test_cmd_history_without_history_prints_no_history(tmp_path, monkeypatch):
    _, output = _setup_session(tmp_path, monkeypatch, None)

    session_commands.cmd_history()

    assert "No history yet" in output.getvalue()
