from jarv import session_commands


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
    assert "\u250c" in rendered
    assert rendered.index("jarv:") < rendered.index("> Command")
    assert rendered.count("jarv:") == 1
    assert len(anchors) == 3


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
        if line.startswith("\u2514") and index > rendered.splitlines().index("jarv:")
    )
    assert rendered.splitlines()[web_search_bottom + 1].startswith("\u250c")


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
        next(line for line in rendered_lines if line.startswith("\u2514"))
    )

    assert rendered_lines[first_card_bottom + 1].startswith("\u250c")


def test_session_row_widths_preserve_message_space_on_small_screens():
    assert session_commands._session_row_widths(40) == (13, 7, 16)
    assert session_commands._session_row_widths(58) == (28, 7, 19)


def test_session_row_widths_give_extra_space_to_message_on_large_screens():
    assert session_commands._session_row_widths(110) == (28, 7, 71)
