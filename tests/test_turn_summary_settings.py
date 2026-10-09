"""One registry drives turn-summary defaults, settings rows, and migration."""

import copy
import json

import pytest

from jarv import config as config_module, settings_command
from jarv.config_schema import CONFIG_FIELD_BY_KEY, build_default_config, parse_config_value, validate_config_fields
from jarv.turn_summary_settings import (
    TURN_SUMMARY_FIELDS,
    enabled_turn_summary_fields,
    migrate_turn_summary_settings,
)


CORE_FIELDS = frozenset({"tokens", "cache", "reasoning", "speed", "time"})
USAGE_FIELDS = frozenset({"tokens", "cache", "session", "cost"})


def test_turn_summary_starts_disabled_with_core_fields_selected():
    config = build_default_config()

    assert config["turn_summary"] is False
    assert enabled_turn_summary_fields(config) == frozenset()
    config["turn_summary"] = True
    assert enabled_turn_summary_fields(config) == CORE_FIELDS
    assert "print_usage_after_model" not in config
    assert "print_usage_after_agent" not in config


@pytest.mark.parametrize("field", TURN_SUMMARY_FIELDS, ids=lambda field: field.name)
def test_field_registry_drives_schema_and_independent_toggle_reset(field, monkeypatch):
    config = build_default_config()
    config["turn_summary"] = True
    before = dict(config)
    schema = CONFIG_FIELD_BY_KEY[field.key]
    assert schema.default is field.default
    assert schema.label == field.label
    assert parse_config_value(field.key, "false") is False
    row = next(row for row in settings_command._settings_rows(config) if row["key"] == field.key)
    assert row["section"] == "turn summary"
    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)

    updated, message = settings_command._settings_apply_quick(row, config)

    assert updated[field.key] is not field.default
    assert (field.name in enabled_turn_summary_fields(updated)) is (not field.default)
    assert {key for key in before if before[key] != updated[key]} == {field.key}
    assert message.startswith(f"saved {field.label}:")

    reset, message = settings_command._settings_reset_row(row, updated)

    assert reset == before
    assert message == f"reset {field.label}"


def test_master_toggle_preserves_custom_field_selection(monkeypatch):
    config = {**build_default_config(), "turn_summary_speed": False, "turn_summary_cost": True}
    before = dict(config)
    row = next(row for row in settings_command._settings_rows(config) if row["key"] == "turn_summary")
    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)

    enabled, _ = settings_command._settings_apply_quick(row, config)
    assert enabled_turn_summary_fields(enabled) == (CORE_FIELDS - {"speed"}) | {"cost"}
    disabled, _ = settings_command._settings_apply_quick(row, enabled)

    assert disabled == before
    assert enabled_turn_summary_fields(disabled) == frozenset()


@pytest.mark.parametrize("key", ["turn_summary", *(field.key for field in TURN_SUMMARY_FIELDS)])
@pytest.mark.parametrize("value", ["false", 1, None])
def test_turn_summary_flags_require_booleans(key, value):
    config = {**build_default_config(), key: value}
    errors = []

    assert not validate_config_fields(config, report=errors.append)
    assert any(key in error and "boolean" in error for error in errors)


@pytest.mark.parametrize("model_stats,agent_usage,expected", [
    (False, False, frozenset()),
    (True, False, CORE_FIELDS),
    (False, True, USAGE_FIELDS),
    (True, True, CORE_FIELDS | USAGE_FIELDS),
])
def test_legacy_switches_migrate_to_one_summary(model_stats, agent_usage, expected):
    config = {"print_usage_after_model": model_stats, "print_usage_after_agent": agent_usage}
    original = copy.deepcopy(config)

    assert enabled_turn_summary_fields(config) == expected
    assert config == original
    assert migrate_turn_summary_settings(config) is True
    assert enabled_turn_summary_fields(config) == expected
    assert "print_usage_after_model" not in config
    assert "print_usage_after_agent" not in config
    assert migrate_turn_summary_settings(config) is False
    if not model_stats and not agent_usage:
        assert config == {}
        assert enabled_turn_summary_fields({**config, "turn_summary": True}) == CORE_FIELDS


def test_migration_preserves_explicit_canonical_fields():
    config = {
        "print_usage_after_model": True,
        "print_usage_after_agent": True,
        "turn_summary": False,
        "turn_summary_cache": False,
        "turn_summary_session": False,
    }

    migrate_turn_summary_settings(config)

    assert enabled_turn_summary_fields(config) == frozenset()
    config["turn_summary"] = True
    assert enabled_turn_summary_fields(config) == (CORE_FIELDS | USAGE_FIELDS) - {"cache", "session"}


@pytest.mark.parametrize("key", ["print_usage_after_model", "print_usage_after_agent"])
@pytest.mark.parametrize("value", ["false", 1, None])
def test_invalid_legacy_flags_rejected_without_partial_mutation(key, value):
    config = {"print_usage_after_model": True, "print_usage_after_agent": True, key: value}
    before = dict(config)

    with pytest.raises(ValueError, match=key):
        migrate_turn_summary_settings(config)
    assert config == before


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", path)
    monkeypatch.setattr("jarv.history.migrate_flat_session_files", lambda: None)
    return path


def test_load_migrates_before_defaults_and_persists_once(config_path, monkeypatch):
    config_path.write_text(json.dumps({"print_usage_after_agent": True}), encoding="utf-8")

    loaded = config_module.load_config()

    assert enabled_turn_summary_fields(loaded) == USAGE_FIELDS
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted == loaded
    assert "print_usage_after_agent" not in persisted
    monkeypatch.setattr(config_module, "save_config", lambda config: pytest.fail("already migrated"))
    assert config_module.load_config() == loaded


def test_loaded_migration_keeps_snapshot_for_concurrent_settings(config_path):
    config_path.write_text(json.dumps({"print_usage_after_model": True}), encoding="utf-8")
    first = config_module.load_config()
    second = config_module.load_config()
    first["turn_summary_speed"] = False
    second["turn_summary_cost"] = True

    config_module.save_config(first)
    config_module.save_config(second)

    assert enabled_turn_summary_fields(config_module.load_config()) == (CORE_FIELDS - {"speed"}) | {"cost"}


def test_invalid_legacy_config_is_reported_and_preserved(config_path):
    original = '{"print_usage_after_agent": "false"}'
    config_path.write_text(original, encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        config_module.load_config()

    assert error.value.code == 1
    assert config_path.read_text(encoding="utf-8") == original
