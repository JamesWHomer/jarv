from jarv.settings_interactive import _settings_fullscreen_panel_width


def test_settings_fullscreen_panel_width_leaves_wrap_guard_column():
    assert _settings_fullscreen_panel_width(120) == 119
    assert _settings_fullscreen_panel_width(2) == 1
    assert _settings_fullscreen_panel_width(1) == 1
