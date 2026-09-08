import pytest

from jarv import settings_command
from jarv import commands, config as config_module
from jarv.config import DEFAULT_CONFIG


@pytest.mark.parametrize("key", ["headsup_border", "headsup_intro_logo", "headsup_intro_stars"])
def test_headsup_display_toggle_saves_and_resets(monkeypatch, key):
    config = dict(DEFAULT_CONFIG)
    saved = []
    monkeypatch.setattr(settings_command, "save_config", lambda value: saved.append(dict(value)))
    row = next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == key
    )
    assert row["section"] == "display"
    assert config[key] is True
    updated, _ = settings_command._settings_apply_quick(row, config)
    assert updated[key] is False
    assert saved[-1][key] is False
    reset, _ = settings_command._settings_reset_row(row, updated)
    assert reset[key] is True
    assert saved[-1][key] is True


@pytest.mark.parametrize("key", ["headsup_border", "headsup_intro_logo", "headsup_intro_stars"])
def test_set_and_unset_headsup_display_toggle(monkeypatch, key):
    config = dict(DEFAULT_CONFIG)
    saved = []
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(config_module, "save_config", lambda value: saved.append(dict(value)))
    commands.cmd_set([key, "false"])
    assert saved[-1][key] is False
    config.update(saved[-1])
    commands.cmd_unset([key])
    assert saved[-1][key] is True
