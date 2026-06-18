import io

from rich.console import Console

from jarv import config as config_module
from jarv import settings_command
from jarv.agent import build_agent_tools
from jarv.config import DEFAULT_CONFIG, TOOL_NAMES, validate_config
from jarv.model_catalog import ModelImageCapability
from jarv.orchestrator import build_subagent_tools


def _tool_names(tools):
    return [tool["name"] for tool in tools]


def _tool_by_name(tools, name):
    return next(tool for tool in tools if tool["name"] == name)


def test_settings_exposes_each_tool_as_a_toggle(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    rows = [
        row
        for row in settings_command._settings_rows(config)
        if row["section"] == "tools"
    ]

    assert [row["tool_name"] for row in rows] == list(TOOL_NAMES)
    assert all(
        settings_command._settings_value_text(row, config).plain == "on"
        for row in rows
    )

    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    updated, message = settings_command._settings_apply_quick(rows[0], config)

    assert updated["disabled_tools"] == ["run_command"]
    assert settings_command._settings_value_text(rows[0], updated).plain == "off"
    assert message == "saved Run commands: off"


def test_tool_toggle_reset_enables_tool(monkeypatch):
    config = {**DEFAULT_CONFIG, "disabled_tools": ["web_search"]}
    row = next(
        row
        for row in settings_command._settings_rows(config)
        if row.get("tool_name") == "web_search"
    )
    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)

    updated, message = settings_command._settings_reset_row(row, config)

    assert updated["disabled_tools"] == []
    assert message == "reset Web search"


def test_tool_reset_preview_shows_enabled_default():
    config = {**DEFAULT_CONFIG, "disabled_tools": ["web_search"]}
    row = next(
        row
        for row in settings_command._settings_rows(config)
        if row.get("tool_name") == "web_search"
    )

    preview = settings_command._settings_reset_action_bar(row, config, 80)

    assert preview.plain.startswith("Reset Web search?   off \u2192 on")


def test_disabled_tools_are_filtered_for_root_and_subagents():
    config = {
        **DEFAULT_CONFIG,
        "disabled_tools": ["run_command", "web_search", "spawn"],
    }

    assert _tool_names(build_agent_tools(config)) == ["read", "ask_user"]
    assert _tool_names(build_subagent_tools(False, config)) == ["read", "finish"]


def test_tool_builders_include_read_image_description_for_capable_model(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    monkeypatch.setattr(
        "jarv.read_tool.get_image_output_capability",
        lambda _config: ModelImageCapability(True, "responses"),
    )

    root_read = _tool_by_name(build_agent_tools(config), "read")
    child_read = _tool_by_name(build_subagent_tools(False, config), "read")

    assert "image-capable models" in root_read["description"]
    assert "image reads" in root_read["description"]
    assert "image-capable models" in child_read["description"]
    assert "image reads" in child_read["description"]


def test_tool_builders_omit_read_image_description_for_text_only_model(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    monkeypatch.setattr(
        "jarv.read_tool.get_image_output_capability",
        lambda _config: ModelImageCapability(False, reason="text-only model"),
    )

    root_read = _tool_by_name(build_agent_tools(config), "read")
    child_read = _tool_by_name(build_subagent_tools(False, config), "read")

    assert "PDFs with embedded text" in root_read["description"]
    assert "image-capable models" not in root_read["description"]
    assert "image reads" not in root_read["description"]
    assert "image-capable models" not in child_read["description"]
    assert "image reads" not in child_read["description"]


def test_subagent_finish_tool_cannot_be_disabled():
    config = {**DEFAULT_CONFIG, "disabled_tools": list(TOOL_NAMES)}

    assert _tool_names(build_agent_tools(config)) == []
    assert _tool_names(build_subagent_tools(False, config)) == ["finish"]


def test_validate_config_rejects_unknown_disabled_tool(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(
        config_module,
        "_console",
        lambda: Console(
            file=output,
            force_terminal=False,
            color_system=None,
        ),
    )

    assert not validate_config(
        {**DEFAULT_CONFIG, "disabled_tools": ["not_a_tool"]}
    )
    assert "unknown tools" in output.getvalue()


def test_validate_config_deduplicates_disabled_tools():
    config = {
        **DEFAULT_CONFIG,
        "disabled_tools": ["read", "read", "spawn"],
    }

    assert validate_config(config)
    assert config["disabled_tools"] == ["read", "spawn"]


def test_settings_exposes_tool_call_display(monkeypatch):
    config = dict(DEFAULT_CONFIG)
    row = next(
        row
        for row in settings_command._settings_rows(config)
        if row["key"] == "tool_call_display"
    )

    assert settings_command._settings_value_text(row, config).plain == "auto"
    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)

    updated, message = settings_command._settings_apply_quick(row, config)

    assert updated["tool_call_display"] == "fullscreen"
    assert message == "saved Tool calls: fullscreen"


def test_validate_config_rejects_unknown_tool_call_display(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(
        config_module,
        "_console",
        lambda: Console(
            file=output,
            force_terminal=False,
            color_system=None,
        ),
    )

    assert not validate_config(
        {**DEFAULT_CONFIG, "tool_call_display": "sideways"}
    )
    assert "tool_call_display" in output.getvalue()
