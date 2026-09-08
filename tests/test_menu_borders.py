import io
from types import SimpleNamespace

import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text

from jarv import paths, session_browser, settings_interactive, setup_interactive, tree_browser, tui_panel, usage_command
from jarv.config import DEFAULT_CONFIG
from jarv.settings_command import _settings_begin_edit
from jarv.settings_editor import render_editor_panel


def _render(renderable, width):
    output = io.StringIO()
    Console(file=output, width=width, color_system=None, legacy_windows=False).print(renderable)
    return output.getvalue().splitlines()


@pytest.mark.parametrize("width,height", [(42, 12), (80, 24)])
def test_all_menu_frames_are_borderless_and_use_full_width(monkeypatch, width, height):
    config = {**DEFAULT_CONFIG, "headsup_border": False}
    for module in (session_browser, settings_interactive, setup_interactive, tree_browser, usage_command):
        monkeypatch.setattr(module, "terminal_size", lambda *, console=None: (width, height))
    settings = settings_interactive.SettingsApp(config)
    setup = setup_interactive.SetupApp(config)
    sessions = session_browser.SessionBrowserScreen(
        data={}, sessions={}, terminals={}, rows=[], current_session_id=None,
    )
    tree = tree_browser.TreeBrowserScreen(model=SimpleNamespace(nodes=[], roots=[], active_path=[]))
    usage = usage_command.UsageScreen(initial_scope="week")
    frames = [
        settings.render(), setup.render(), sessions.render(), tree.render(), usage.render(),
    ]
    row = next(row for row in settings.rows if row["key"] == "base_url")
    settings.edit = _settings_begin_edit(row, config)
    frames.append(settings.render())
    setup.phase = "step"
    frames.append(setup.render())
    sessions.preview_sid = "test-session"
    monkeypatch.setattr(sessions, "_preview_lines", lambda sid, content_width: [Text("X" * content_width)])
    frames.append(sessions.render())
    for frame in frames:
        lines = _render(frame, width)
        assert len(lines) == height
        assert all(cell_len(line) == width for line in lines)
        assert lines[0].startswith("jarv ▸")
        assert not any(char in "\n".join(lines) for char in "╭╮╰╯│")
    assert "X" * width in _render(frames[-1], width)


@pytest.mark.parametrize("width,height", [(42, 12), (100, 24)])
def test_borderless_menus_end_with_controls(monkeypatch, width, height):
    config = {**DEFAULT_CONFIG, "headsup_border": False}
    for module in (session_browser, settings_interactive, setup_interactive, tree_browser, usage_command):
        monkeypatch.setattr(module, "terminal_size", lambda *, console=None: (width, height))
    settings = settings_interactive.SettingsApp(config)
    setup = setup_interactive.SetupApp(config)
    setup.phase = "ready"
    sessions = session_browser.SessionBrowserScreen(
        data={}, sessions={}, terminals={}, rows=[], current_session_id=None,
    )
    tree = tree_browser.TreeBrowserScreen(model=SimpleNamespace(nodes=[], roots=[], active_path=[]))
    usage = usage_command.UsageScreen(initial_scope="week")
    monkeypatch.setattr(usage, "_view", lambda: SimpleNamespace(window_label="week", source_path="usage.json"))
    monkeypatch.setattr(usage, "_body_lines", lambda view, width: [Text("usage row")] * 40)
    for app in (settings, setup, sessions, tree, usage):
        lines = _render(app.render(), width)
        assert len(lines) == height
        assert lines[-1].strip(), type(app).__name__
        assert any(hint in lines[-1].lower() for hint in ("enter", "esc", "scroll", "↑↓")), lines[-1]
    sessions.preview_sid = "test-session"
    monkeypatch.setattr(sessions, "_preview_lines", lambda sid, width: [Text("preview row")] * 40)
    assert "↑↓ scroll" in _render(sessions.render(), width)[-1]


def test_borderless_panel_omits_empty_bottom_row_and_clips_header():
    panel = tui_panel.MenuPanel(Text("body"), title="menu", subtitle="界" * 40, border=False, width=20)
    lines = _render(panel, 20)
    assert len(lines) == 2
    assert lines[0].startswith("menu  ")
    assert all(cell_len(line) == 20 for line in lines)
    assert lines[-1].strip() == "body"


@pytest.mark.parametrize("key", ["base_url", "system_prompt"])
def test_borderless_editor_controls_stay_on_last_row(key):
    config = {**DEFAULT_CONFIG, "headsup_border": False}
    settings = settings_interactive.SettingsApp(config)
    row = next(row for row in settings.rows if row["key"] == key)
    edit = _settings_begin_edit(row, config)
    panel = render_editor_panel(edit, config, panel_width=80, height=12, title="edit")
    lines = _render(panel, 80)
    assert len(lines) == 12
    assert "Esc" in lines[-1]


def test_settings_toggle_redraws_current_menu(monkeypatch):
    monkeypatch.setattr("jarv.settings_command.save_config", lambda config: None)
    monkeypatch.setattr(settings_interactive, "terminal_size", lambda *, console=None: (80, 24))
    app = settings_interactive.SettingsApp(dict(DEFAULT_CONFIG))
    app.selected = next(i for i, row in enumerate(app.rows) if row["key"] == "headsup_border")
    assert _render(app.render(), 80)[0].startswith("╭")
    app.on_key("ENTER", 1)
    assert _render(app.render(), 80)[0].startswith("jarv")
    assert not tui_panel.menu_border_enabled()
    app.on_key("ENTER", 1)
    assert _render(app.render(), 80)[0].startswith("╭")
    assert tui_panel.menu_border_enabled()


def test_standalone_menu_reads_saved_border_preference(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"headsup_border": false}', encoding="utf-8")
    monkeypatch.setattr(paths, "CONFIG_FILE", config_file)
    monkeypatch.setattr(tui_panel, "_menu_border", None)
    panel = tui_panel.MenuPanel(Text("X" * 42), title="jarv ▸ test", title_align="left", width=42, height=8)
    lines = _render(panel, 42)
    assert lines[0].startswith("jarv")
    assert lines[1] == "X" * 42
