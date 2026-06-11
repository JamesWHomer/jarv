import io

from rich.console import Console

from jarv import command_input, settings_command
from jarv.config import DEFAULT_CONFIG


def _system_prompt_row(config):
    return next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == "system_prompt"
    )


def test_system_prompt_row_shows_status_instead_of_contents():
    config = dict(DEFAULT_CONFIG)
    row = _system_prompt_row(config)

    assert settings_command._settings_value_text(row, config).plain == "default"

    config["system_prompt"] = "Custom instructions"
    assert settings_command._settings_value_text(row, config).plain == "custom \u00b7 19 chars"

    config["system_prompt"] = ""
    assert settings_command._settings_value_text(row, config).plain == "empty"


def test_system_prompt_editor_inserts_at_cursor_and_preserves_newlines():
    config = {**DEFAULT_CONFIG, "system_prompt": "one\ntwo"}
    edit = settings_command._settings_begin_edit(_system_prompt_row(config), config)

    settings_command._settings_multiline_apply_key(edit, "UP")
    settings_command._settings_multiline_apply_key(edit, "HOME")
    settings_command._settings_multiline_apply_key(edit, ">")
    settings_command._settings_multiline_apply_key(edit, "END")
    settings_command._settings_multiline_apply_key(edit, "ENTER")
    settings_command._settings_multiline_apply_key(edit, "x")

    assert edit["buffer"] == ">one\nx\ntwo"
    assert edit["cursor"] == 6


def test_system_prompt_up_down_moves_across_visually_wrapped_rows():
    config = {**DEFAULT_CONFIG, "system_prompt": "abcdefghijkl"}
    edit = settings_command._settings_begin_edit(_system_prompt_row(config), config)
    edit["cursor"] = 2

    settings_command._settings_multiline_apply_key(edit, "DOWN", inner_width=8)
    assert edit["cursor"] == 6

    settings_command._settings_multiline_apply_key(edit, "DOWN", inner_width=8)
    assert edit["cursor"] == 10

    settings_command._settings_multiline_apply_key(edit, "UP", inner_width=8)
    assert edit["cursor"] == 6


def test_system_prompt_commit_preserves_whitespace(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    row = _system_prompt_row(config)
    edit = settings_command._settings_begin_edit(row, config)
    edit["buffer"] = "  first line\nsecond line  "
    saved = []
    monkeypatch.setattr(settings_command, "save_config", lambda value: saved.append(dict(value)))

    updated, message, style, done = settings_command._settings_commit_edit(edit, config)

    assert done
    assert style == "green"
    assert "custom" in message
    assert updated["system_prompt"] == "  first line\nsecond line  "
    assert saved[-1]["system_prompt"] == "  first line\nsecond line  "


def test_system_prompt_editor_renders_save_and_discard_guidance():
    config = {**DEFAULT_CONFIG, "system_prompt": "line one\nline two"}
    edit = settings_command._settings_begin_edit(_system_prompt_row(config), config)
    edit["buffer"] += " changed"
    edit["discard_armed"] = True

    output = io.StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=80)
    console.print(
        *settings_command._settings_editor_lines(edit, config, 76, max_lines=10),
        sep="\n",
    )
    rendered = output.getvalue()

    assert "Ctrl+S save" in rendered
    assert "Enter newline" in rendered
    assert "modified" in rendered
    assert "Esc again to discard" in rendered


def test_system_prompt_controls_are_pinned_to_editor_bottom():
    config = {**DEFAULT_CONFIG, "system_prompt": "short"}
    edit = settings_command._settings_begin_edit(_system_prompt_row(config), config)

    lines = settings_command._settings_editor_lines(edit, config, 76, max_lines=10)

    assert len(lines) == 10
    assert "Ctrl+S save" in lines[-1].plain
    assert any(not line.plain for line in lines[3:-1])


def test_system_prompt_cursor_renders_once_at_wrap_boundary():
    config = {**DEFAULT_CONFIG, "system_prompt": "abcdefgh"}
    edit = settings_command._settings_begin_edit(_system_prompt_row(config), config)
    edit["cursor"] = 4

    lines, cursor_line = settings_command._settings_multiline_visual_lines(edit, 8)

    assert sum(line.plain.count("_") for line in lines) == 1
    assert cursor_line == 1


def test_read_key_maps_windows_ctrl_s(monkeypatch):
    class FakeMsvcrt:
        @staticmethod
        def getwch():
            return "\x13"

    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.setitem(command_input.sys.modules, "msvcrt", FakeMsvcrt)

    assert command_input._read_key(text_mode=True) == "CTRL_S"
