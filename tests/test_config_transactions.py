import copy
import json

import pytest

from jarv import config as config_module, settings_command
from jarv.config_schema import build_default_config, validate_config_fields


@pytest.mark.parametrize("key,value", [
    ("interactive_commands", "false"), ("audit", 1), ("colour", None),
    ("api_keys", None), ("api_keys", []), ("api_keys", {"openai": 123}),
    ("command_timeout", True), ("command_timeout", 1.5),
])
def test_schema_rejects_malformed_types(key, value):
    config = build_default_config()
    config[key] = value
    errors = []
    assert not validate_config_fields(config, report=errors.append)
    assert any(key in error for error in errors)


@pytest.mark.parametrize("existing", [False, True])
def test_loaded_defaults_do_not_share_mutable_state(tmp_path, monkeypatch, existing):
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("jarv.history.migrate_flat_session_files", lambda: None)
    if existing:
        (tmp_path / "config.json").write_text(json.dumps({"model": "gpt-5.4-mini"}))
    before = copy.deepcopy(config_module.DEFAULT_CONFIG)
    loaded = config_module.load_config()
    loaded["api_keys"]["example"] = "secret"
    loaded["disabled_tools"].append("read")
    assert config_module.DEFAULT_CONFIG == before
    assert build_default_config()["api_keys"] == {}


def test_failed_setting_edit_preserves_live_config(monkeypatch):
    config = build_default_config()
    before = copy.deepcopy(config)
    saved = []
    monkeypatch.setattr(settings_command, "save_config", saved.append)
    row = next(row for row in settings_command._settings_rows(config)
               if row["key"] == "tool_output_display_lines")
    returned, _, _, done = settings_command._settings_commit_edit(
        {"row": row, "buffer": "invalid"}, config)
    assert not done
    assert returned is config
    assert config == before
    assert not saved


def test_failed_nested_setting_edit_preserves_live_config(monkeypatch):
    config = build_default_config()
    config["api_keys"] = {"openai": "old-secret"}
    before = copy.deepcopy(config)
    monkeypatch.setattr(settings_command, "validate_config", lambda candidate: False)
    row = next(row for row in settings_command._settings_rows(config) if row["key"] == "api_key")
    returned, _, _, done = settings_command._settings_commit_edit(
        {"row": row, "buffer": "clear"}, config)
    assert not done
    assert returned is config
    assert config == before


def test_settings_allows_zero_subagent_depth(monkeypatch):
    config = build_default_config()
    saved = []
    monkeypatch.setattr(settings_command, "save_config", saved.append)
    row = next(row for row in settings_command._settings_rows(config)
               if row["key"] == "max_subagent_depth")
    returned, _, _, done = settings_command._settings_commit_edit(
        {"row": row, "buffer": "0"}, config)
    assert done
    assert returned is config
    assert config["max_subagent_depth"] == 0
    assert saved[0]["max_subagent_depth"] == 0
