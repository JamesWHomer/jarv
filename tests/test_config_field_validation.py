"""Reject malformed config fields before runtime consumers use them."""

import json

import pytest

from jarv import config as config_module
from jarv.config_schema import build_default_config, validate_config_fields


@pytest.mark.parametrize("key", [
    "system_prompt", "api_key", "base_url", "model", "auditor_model", "provider",
    "reasoning_effort",
])
@pytest.mark.parametrize("value", [123, 1.5, True, [], {}])
def test_config_rejects_non_string_text_settings(key, value):
    config = build_default_config()
    config[key] = value
    errors = []

    assert not validate_config_fields(config, report=errors.append)
    assert any(key in error and "string" in error for error in errors)
    assert not config_module.validate_config(config)


@pytest.mark.parametrize("key", [
    "system_prompt", "api_key", "base_url", "model", "auditor_model", "provider",
])
def test_config_rejects_null_text_settings(key):
    config = build_default_config()
    config[key] = None

    assert not config_module.validate_config(config)


@pytest.mark.parametrize("value", [3.5, float("inf"), float("nan"), True])
def test_config_rejects_non_integer_display_lines(value):
    config = build_default_config()
    config["tool_output_display_lines"] = value

    assert not config_module.validate_config(config)


def test_config_preserves_null_reasoning_effort_as_default():
    config = build_default_config()
    config["reasoning_effort"] = None

    assert config_module.validate_config(config)
    assert config["reasoning_effort"] == ""


def test_load_rejects_non_string_system_prompt(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", path)
    monkeypatch.setattr("jarv.history.migrate_flat_session_files", lambda: None)
    original = json.dumps({"system_prompt": 123})
    path.write_text(original, encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        config_module.load_config()

    assert error.value.code == 1
    assert path.read_text(encoding="utf-8") == original
@pytest.mark.parametrize("content", [
    b"null", b"[]", b"1", b'"text"', b"\xff", b'{"provider": []}',
    b'{"api_keys": null}', b'{"api_keys": {"openai": 123}}',
])
def test_setup_probe_handles_malformed_config(tmp_path, monkeypatch, content):
    from jarv import config

    path = tmp_path / "config.json"
    path.write_bytes(content)
    monkeypatch.setattr(config, "CONFIG_FILE", path)
    assert config.is_setup_complete() is False
    assert path.read_bytes() == content
