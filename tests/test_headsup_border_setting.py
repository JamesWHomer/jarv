from jarv import settings_command
from jarv import commands, config as config_module
from jarv.config import DEFAULT_CONFIG


def test_headsup_border_toggle_saves_and_resets(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    saved = []
    monkeypatch.setattr(settings_command, "save_config", lambda value: saved.append(dict(value)))
    row = next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == "headsup_border"
    )
    assert row["section"] == "display"
    assert config["headsup_border"] is True
    updated, _ = settings_command._settings_apply_quick(row, config)
    assert updated["headsup_border"] is False
    assert saved[-1]["headsup_border"] is False
    reset, _ = settings_command._settings_reset_row(row, updated)
    assert reset["headsup_border"] is True
    assert saved[-1]["headsup_border"] is True


def test_set_and_unset_headsup_border(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    saved = []
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(config_module, "save_config", lambda value: saved.append(dict(value)))
    commands.cmd_set(["headsup_border", "false"])
    assert saved[-1]["headsup_border"] is False
    config.update(saved[-1])
    commands.cmd_unset(["headsup_border"])
    assert saved[-1]["headsup_border"] is True
