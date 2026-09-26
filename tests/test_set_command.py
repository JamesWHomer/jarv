"""Config values keep their declared types through /set and later turns."""

import copy

import pytest

from jarv import agent, commands, config as config_module
from jarv.config_schema import build_default_config


@pytest.fixture
def set_config(monkeypatch):
    config = build_default_config()
    saved = []
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(config_module, "save_config", saved.append)
    return config, saved


@pytest.mark.parametrize("key", [
    "system_prompt", "api_key", "base_url", "model", "auditor_model", "provider",
])
@pytest.mark.parametrize("raw", ["123", "00123", "1.5", "1e3", "true", "false", "nan", "inf"])
def test_set_preserves_text_settings(set_config, key, raw):
    config, saved = set_config
    before = copy.deepcopy(config)

    assert commands.cmd_set([key, raw]) == 0

    assert saved[0][key] == raw
    assert isinstance(saved[0][key], str)
    assert config == before


@pytest.mark.parametrize("key,raw,expected", [
    ("audit", "true", True),
    ("audit", "FALSE", False),
    ("command_timeout", "42", 42),
    ("command_timeout", "0042", 42),
    ("max_subagent_depth", "0", 0),
    ("context_budget_ratio", "0.5", 0.5),
    ("context_budget_ratio", "5e-1", 0.5),
    ("tool_output_display_lines", "3", 3),
    ("tool_output_display_lines", "auto", "auto"),
    ("tool_output_display_lines", "", "auto"),
    ("command_safety", "none", "none"),
    ("reasoning_effort", "default", ""),
    ("system_prompt", "", ""),
])
def test_set_parses_values_for_their_field(set_config, key, raw, expected):
    _, saved = set_config

    assert commands.cmd_set([key, raw]) == 0

    assert saved[0][key] == expected
    assert type(saved[0][key]) is type(expected)


@pytest.mark.parametrize("key,raw", [
    ("audit", "1"),
    ("audit", "yes"),
    ("command_timeout", "true"),
    ("command_timeout", "1.5"),
    ("command_timeout", "0"),
    ("command_timeout", "inf"),
    ("max_subagent_depth", "-1"),
    ("context_budget_ratio", "nan"),
    ("context_budget_ratio", "inf"),
    ("context_budget_ratio", "1"),
    ("tool_output_display_lines", "2"),
    ("tool_output_display_lines", "3.5"),
    ("tool_output_display_lines", "inf"),
    ("tool_output_display_lines", "true"),
    ("command_safety", "123"),
    ("reasoning_effort", "123"),
    ("model", ""),
    ("api_keys", "123"),
    ("service_tiers", "123"),
    ("disabled_tools", "123"),
])
def test_set_rejects_invalid_values_without_saving(set_config, key, raw):
    config, saved = set_config
    before = copy.deepcopy(config)

    assert commands.cmd_set([key, raw]) == 2

    assert not saved
    assert config == before


def test_set_system_prompt_can_be_loaded_and_used_on_next_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr("jarv.history.migrate_flat_session_files", lambda: None)
    monkeypatch.setattr(agent, "get_system_info", lambda **_kwargs: "test system")
    config = build_default_config()
    config["project_context"] = False
    config_module.save_config(config)

    assert commands.cmd_set(["system_prompt", "123"]) == 0

    loaded = config_module.load_config()
    assert agent.build_instructions(loaded, cwd=str(tmp_path)) == (
        "123\n\nSystem info:\ntest system"
    )


def test_set_preserves_whitespace_and_joins_prompt_arguments(set_config):
    _, saved = set_config

    assert commands.cmd_set(["system_prompt", "  first", "line\nsecond  "]) == 0

    assert saved[0]["system_prompt"] == "  first line\nsecond  "
