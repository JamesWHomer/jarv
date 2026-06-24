import io
from types import SimpleNamespace

from rich.console import Console

from jarv import command_input, settings_command, settings_editor
from jarv.config import DEFAULT_CONFIG


_DUMMY_CATALOG = SimpleNamespace(
    request=lambda *a, **k: None,
    cancel_pending=lambda: None,
)


def _api_key_row(config):
    return next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == "api_key"
    )


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


def test_short_system_prompt_editor_requests_compact_height():
    config = {**DEFAULT_CONFIG, "system_prompt": "short"}
    edit = settings_command._settings_begin_edit(_system_prompt_row(config), config)

    assert settings_command._settings_desired_editor_height(
        edit,
        config,
        76,
        32,
    ) == 7


def test_system_prompt_cursor_uses_reverse_video_at_wrap_boundary():
    config = {**DEFAULT_CONFIG, "system_prompt": "abcdefgh"}
    edit = settings_command._settings_begin_edit(_system_prompt_row(config), config)
    edit["cursor"] = 4

    lines, cursor_line = settings_command._settings_multiline_visual_lines(edit, 8)

    assert cursor_line == 1
    assert lines[1].plain == "  efgh"
    assert sum(
        "reverse" in str(span.style)
        for line in lines
        for span in line.spans
    ) == 1


def test_system_prompt_cursor_uses_reverse_space_at_end_of_line():
    config = {**DEFAULT_CONFIG, "system_prompt": "abc"}
    edit = settings_command._settings_begin_edit(_system_prompt_row(config), config)

    lines, cursor_line = settings_command._settings_multiline_visual_lines(edit, 20)

    assert cursor_line == 0
    assert lines[0].plain == "  abc "
    assert "reverse" in str(lines[0].spans[-1].style)


def test_compact_text_setting_reuses_cursor_aware_editor():
    config = {**DEFAULT_CONFIG, "base_url": "https://example.test"}
    row = next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == "base_url"
    )
    edit = settings_command._settings_begin_edit(row, config)

    assert edit["cursor"] == len(config["base_url"])
    settings_command.apply_text_editor_key(edit, "HOME")
    settings_command.apply_text_editor_key(edit, "X")

    assert edit["buffer"] == "Xhttps://example.test"


def test_compact_text_setting_renders_discard_guidance():
    config = {**DEFAULT_CONFIG, "base_url": "https://example.test"}
    row = next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == "base_url"
    )
    edit = settings_command._settings_begin_edit(row, config)
    edit["buffer"] += "/v1"
    edit["discard_armed"] = True

    lines = settings_command._settings_editor_lines(edit, config, 80)

    assert any("Esc again to discard" in line.plain for line in lines)


def test_compact_integer_editor_uses_one_value_row_and_minimal_height():
    config = dict(DEFAULT_CONFIG)
    row = next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == "max_stdin_chars"
    )
    edit = settings_command._settings_begin_edit(row, config)

    lines = settings_command._settings_editor_lines(edit, config, 80)

    assert len(lines) == 1
    assert lines[0].plain.startswith("  Value: 200000")
    assert "Enter save" not in lines[0].plain
    assert settings_command._settings_desired_editor_height(
        edit,
        config,
        80,
        24,
    ) == 3


def test_compact_integer_validation_adds_one_editor_row():
    config = dict(DEFAULT_CONFIG)
    row = next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == "max_stdin_chars"
    )
    edit = settings_command._settings_begin_edit(row, config)
    edit["buffer"] = "abc"
    edit["cursor"] = 3
    edit["error"] = "Enter a positive integer."

    lines = settings_command._settings_editor_lines(edit, config, 80)

    assert len(lines) == 2
    assert lines[1].plain == "  Enter a positive integer."
    assert settings_command._settings_desired_editor_height(
        edit,
        config,
        80,
        24,
    ) == 4


def test_api_key_compact_editor_masks_value_but_not_clear():
    config = dict(DEFAULT_CONFIG)
    row = next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == "api_key"
    )
    edit = settings_command._settings_begin_edit(row, config)
    edit["buffer"] = "secret"
    edit["cursor"] = len(edit["buffer"])

    masked = settings_command._settings_editor_lines(edit, config, 80)
    assert "secret" not in "\n".join(line.plain for line in masked)
    assert "******" in "\n".join(line.plain for line in masked)

    edit["buffer"] = "clear"
    edit["cursor"] = len(edit["buffer"])
    clear = settings_command._settings_editor_lines(edit, config, 80)
    assert "clear" in "\n".join(line.plain for line in clear)


def test_api_key_editor_masks_stored_config_key_as_placeholder():
    config = {**DEFAULT_CONFIG, "provider": "openai", "api_keys": {"openai": "sk-secretvalue"}}
    row = _api_key_row(config)
    edit = settings_command._settings_begin_edit(row, config)

    assert edit["key_source"] == "config"
    assert edit["placeholder_active"] is True

    text = "\n".join(
        line.plain for line in settings_command._settings_editor_lines(edit, config, 80)
    )
    assert "*****" in text
    assert "sk-secretvalue" not in text
    assert "already saved" in text


def test_api_key_backspace_clears_stored_config_key(monkeypatch):
    saved = []
    monkeypatch.setattr(settings_command, "save_config", lambda cfg: saved.append(dict(cfg)))
    config = {**DEFAULT_CONFIG, "provider": "openai", "api_keys": {"openai": "sk-secretvalue"}}
    row = _api_key_row(config)
    edit = settings_command._settings_begin_edit(row, config)

    config, outcome = settings_editor.apply_editor_key(
        edit, config, "BACKSPACE", 1, catalog=_DUMMY_CATALOG, inner_width=80
    )
    assert outcome.kind == "continue"
    assert edit["placeholder_active"] is False
    assert edit["cleared"] is True

    # The masked stand-in is gone; the field now reads as empty.
    text = "\n".join(
        line.plain for line in settings_command._settings_editor_lines(edit, config, 80)
    )
    assert "*****" not in text

    cfg, message, style, done = settings_command._settings_commit_edit(edit, config)
    assert done is True
    assert message == "cleared stored API key"
    assert "openai" not in cfg.get("api_keys", {})
    assert saved


def test_api_key_enter_without_touching_keeps_stored_key():
    config = {**DEFAULT_CONFIG, "provider": "openai", "api_keys": {"openai": "sk-secretvalue"}}
    row = _api_key_row(config)
    edit = settings_command._settings_begin_edit(row, config)

    cfg, message, _style, done = settings_command._settings_commit_edit(edit, config)
    assert done is True
    assert message == "API key unchanged"
    assert cfg["api_keys"]["openai"] == "sk-secretvalue"


def test_api_key_typing_replaces_stored_config_placeholder(monkeypatch):
    saved = []
    monkeypatch.setattr(settings_command, "save_config", lambda cfg: saved.append(dict(cfg)))
    config = {**DEFAULT_CONFIG, "provider": "openai", "api_keys": {"openai": "sk-oldvalue"}}
    row = _api_key_row(config)
    edit = settings_command._settings_begin_edit(row, config)

    config, _outcome = settings_editor.apply_editor_key(
        edit, config, "x", 1, catalog=_DUMMY_CATALOG, inner_width=80
    )
    assert edit["placeholder_active"] is False
    assert edit["buffer"] == "x"

    edit["buffer"] = "sk-brandnewreplacementkey00"
    edit["cursor"] = len(edit["buffer"])
    cfg, message, _style, done = settings_command._settings_commit_edit(edit, config)
    assert done is True
    assert message == "saved API key"
    assert cfg["api_keys"]["openai"] == "sk-brandnewreplacementkey00"


def test_api_key_editor_announces_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fromenv-1234567890")
    config = {**DEFAULT_CONFIG, "provider": "openai", "api_keys": {}, "api_key": ""}
    row = _api_key_row(config)
    edit = settings_command._settings_begin_edit(row, config)

    assert edit["key_source"] == "env"
    assert edit["placeholder_active"] is False

    text = "\n".join(
        line.plain for line in settings_command._settings_editor_lines(edit, config, 80)
    )
    assert "OPENAI_API_KEY" in text
    assert "override" in text
    assert "*****" not in text


def test_api_key_value_text_distinguishes_env_from_config(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fromenv-1234567890")
    env_config = {**DEFAULT_CONFIG, "provider": "openai", "api_keys": {}, "api_key": ""}
    row = _api_key_row(env_config)
    assert settings_command._settings_value_text(row, env_config).plain == "from OPENAI_API_KEY"

    config_config = {**DEFAULT_CONFIG, "provider": "openai", "api_keys": {"openai": "sk-stored"}}
    assert settings_command._settings_value_text(row, config_config).plain == "configured"


def test_read_key_maps_windows_ctrl_s(monkeypatch):
    class FakeMsvcrt:
        @staticmethod
        def getwch():
            return "\x13"

    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.setitem(command_input.sys.modules, "msvcrt", FakeMsvcrt)

    assert command_input._read_key(text_mode=True) == "CTRL_S"


def test_reset_requires_explicit_confirmation(monkeypatch):
    config = {**DEFAULT_CONFIG, "system_prompt": "Keep this prompt"}
    row = _system_prompt_row(config)
    saved = []
    monkeypatch.setattr(settings_command, "save_config", lambda value: saved.append(dict(value)))

    action_bar = settings_command._settings_reset_action_bar(row, config, 100)

    assert action_bar.plain.startswith(
        "Reset System prompt?   custom \u00b7 16 chars \u2192 default"
    )
    assert action_bar.plain.endswith("y reset   Esc back")
    assert len(action_bar.plain) == 100
    assert config["system_prompt"] == "Keep this prompt"
    assert saved == []

    updated, message, style = settings_command._settings_finish_reset(row, config, "y")

    assert updated["system_prompt"] == DEFAULT_CONFIG["system_prompt"]
    assert message == "reset System prompt"
    assert style == "cyan"
    assert saved[-1]["system_prompt"] == DEFAULT_CONFIG["system_prompt"]


def test_reset_confirmation_cancels_on_other_keys(monkeypatch):
    row = _system_prompt_row(DEFAULT_CONFIG)

    for key in ("r", "ESC", "DOWN"):
        config = {**DEFAULT_CONFIG, "system_prompt": "Keep this prompt"}
        saved = []
        monkeypatch.setattr(settings_command, "save_config", lambda value: saved.append(dict(value)))

        updated, message, style = settings_command._settings_finish_reset(row, config, key)

        assert updated["system_prompt"] == "Keep this prompt"
        assert message == "System prompt reset cancelled"
        assert style == "dim"
        assert saved == []


def test_api_key_reset_action_uses_clear_language():
    config = dict(DEFAULT_CONFIG)
    row = next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == "api_key"
    )

    action_bar = settings_command._settings_reset_action_bar(row, config, 80)

    assert action_bar.plain.startswith("Clear stored API key?")
    assert action_bar.plain.endswith("y clear   Esc back")
    assert len(action_bar.plain) == 80
