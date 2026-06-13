from jarv import model_catalog, settings_command
from jarv.config import DEFAULT_CONFIG, validate_config
from jarv.model_catalog import CatalogModel
from jarv.reasoning import (
    get_reasoning_capabilities,
    reasoning_effort_choices,
    reasoning_effort_description,
    reasoning_effort_error,
    reconcile_reasoning_effort,
)


def test_anthropic_native_capabilities_override_catalog_and_policy(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("openrouter", [
        CatalogModel(
            id="anthropic/claude-sonnet-4.6",
            metadata={
                "supported_parameters": ["reasoning", "include_reasoning"],
            },
        ),
    ])
    model_catalog._write_cache("anthropic", [
        CatalogModel(
            id="claude-sonnet-4-6",
            metadata={
                "max_tokens": 128000,
                "capabilities": {
                    "effort": {
                        "supported": True,
                        "low": {"supported": True},
                        "medium": {"supported": True},
                        "high": {"supported": True},
                        "xhigh": {"supported": False},
                        "max": {"supported": True},
                    },
                    "thinking": {
                        "supported": True,
                        "types": {
                            "enabled": {"supported": True},
                            "adaptive": {"supported": True},
                        },
                    },
                },
            },
        ),
    ])

    config = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
    }
    capabilities = get_reasoning_capabilities(config)

    assert capabilities.efforts == ("low", "medium", "high", "max")
    assert capabilities.modes == ("enabled", "adaptive")
    assert capabilities.native_effort is True
    assert capabilities.max_output_tokens == 128000
    assert capabilities.sources["efforts"] == "Anthropic Models API"
    assert reasoning_effort_error(config, "xhigh") == (
        "claude-sonnet-4-6 supports reasoning efforts: low, medium, high, max"
    )
    assert reasoning_effort_description(config) == "low, medium, high, max"


def test_provider_model_normalizers_preserve_reasoning_metadata():
    anthropic = model_catalog._normalize_anthropic_models({
        "data": [{
            "id": "claude-test",
            "display_name": "Claude Test",
            "max_input_tokens": 1000000,
            "max_tokens": 128000,
            "capabilities": {
                "thinking": {"supported": True},
                "effort": {"supported": True},
            },
        }],
    })[0]
    gemini = model_catalog._normalize_gemini_models({
        "models": [{
            "name": "models/gemini-test",
            "thinking": True,
            "outputTokenLimit": 65536,
            "supportedGenerationMethods": ["generateContent"],
        }],
    })[0]

    assert anthropic.metadata["capabilities"]["thinking"]["supported"] is True
    assert anthropic.metadata["max_tokens"] == 128000
    assert gemini.metadata["thinking"] is True
    assert gemini.metadata["outputTokenLimit"] == 65536


def test_anthropic_manual_thinking_exposes_budget_levels_without_native_effort(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("anthropic", [
        CatalogModel(
            id="claude-haiku-4-5",
            metadata={
                "capabilities": {
                    "effort": {"supported": False},
                    "thinking": {
                        "supported": True,
                        "types": {
                            "enabled": {"supported": True},
                            "adaptive": {"supported": False},
                        },
                    },
                },
            },
        ),
    ])

    capabilities = get_reasoning_capabilities({
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
    })

    assert capabilities.efforts == (
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    )
    assert capabilities.modes == ("enabled",)
    assert capabilities.native_effort is False


def test_gemini_models_metadata_can_disable_reasoning_controls(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("gemini", [
        CatalogModel(
            id="gemini-3.1-pro-preview",
            metadata={
                "thinking": False,
                "outputTokenLimit": 65536,
            },
        ),
    ])

    capabilities = get_reasoning_capabilities({
        "provider": "gemini",
        "model": "gemini-3.1-pro-preview",
    })

    assert capabilities.supported is False
    assert capabilities.max_output_tokens == 65536
    assert reasoning_effort_choices({
        "provider": "gemini",
        "model": "gemini-3.1-pro-preview",
    }) == (("", "default"),)


def test_openrouter_endpoint_metadata_overrides_aggregate_support(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("openrouter", [
        CatalogModel(
            id="vendor/model",
            metadata={
                "supported_parameters": ["reasoning", "include_reasoning"],
            },
        ),
    ])
    model_catalog._write_openrouter_endpoints("vendor/model", [
        {
            "provider_name": "Provider A",
            "supported_parameters": ["reasoning", "include_reasoning"],
            "max_completion_tokens": 64000,
        },
        {
            "provider_name": "Provider B",
            "supported_parameters": ["reasoning"],
            "max_completion_tokens": 32000,
        },
    ])

    capabilities = get_reasoning_capabilities({
        "provider": "openrouter",
        "model": "vendor/model",
    })

    assert capabilities.supported is True
    assert capabilities.returns_reasoning is True
    assert capabilities.max_output_tokens == 32000
    assert capabilities.sources["supported"] == "OpenRouter endpoint catalog"
    assert reasoning_effort_choices({
        "provider": "openrouter",
        "model": "vendor/model",
    }) == (
        ("", "default"),
        ("none", "none"),
        ("minimal", "minimal"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
    )


def test_openrouter_endpoint_refresh_falls_back_to_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    endpoints = [{
        "provider_name": "Provider A",
        "supported_parameters": ["reasoning"],
    }]
    monkeypatch.setattr(
        model_catalog,
        "discover_openrouter_endpoints",
        lambda _config, _model: endpoints,
    )

    assert model_catalog.refresh_openrouter_endpoints({}, "vendor/model") == endpoints
    monkeypatch.setattr(
        model_catalog,
        "discover_openrouter_endpoints",
        lambda _config, _model: (_ for _ in ()).throw(OSError("offline")),
    )
    assert model_catalog.refresh_openrouter_endpoints({}, "vendor/model") == endpoints


def test_unknown_model_only_offers_default(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)

    assert reasoning_effort_choices({
        "provider": "vllm",
        "model": "custom-model",
    }) == (("", "default"),)


def test_config_accepts_default_alias_and_normalizes_it():
    config = dict(DEFAULT_CONFIG)
    config["reasoning_effort"] = "default"

    assert validate_config(config) is True
    assert config["reasoning_effort"] == ""


def test_reconcile_resets_effort_for_gpt_4o():
    config = {
        "provider": "openai",
        "model": "gpt-4o",
        "reasoning_effort": "medium",
    }

    assert reconcile_reasoning_effort(config) == "medium"
    assert config["reasoning_effort"] == ""


def test_reconcile_preserves_effort_for_unknown_custom_model(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    config = {
        "provider": "vllm",
        "model": "custom-model",
        "reasoning_effort": "medium",
    }

    assert reconcile_reasoning_effort(config) is None
    assert config["reasoning_effort"] == "medium"


def test_model_change_resets_incompatible_effort(monkeypatch):
    saved = []
    monkeypatch.setattr(
        settings_command,
        "save_config",
        lambda config: saved.append(dict(config)),
    )
    config = {
        "provider": "openai",
        "model": "gpt-5.4-mini",
        "reasoning_effort": "medium",
    }
    edit = {
        "row": {"key": "model", "kind": "text", "label": "Model"},
        "model_choices": [("gpt-4o", "")],
        "selected_model_index": 0,
        "model_input_active": False,
        "buffer": "",
    }

    updated, message, style, done = settings_command._settings_commit_edit(
        edit,
        config,
    )

    assert done is True
    assert style == "green"
    assert updated["model"] == "gpt-4o"
    assert updated["reasoning_effort"] == ""
    assert message == (
        "saved Model: gpt-4o (reasoning effort reset to default)"
    )
    assert saved[-1]["reasoning_effort"] == ""
    reasoning_row = next(
        row
        for row in settings_command._settings_rows(updated)
        if row["key"] == "reasoning_effort"
    )
    assert settings_command._settings_value_text(
        reasoning_row,
        updated,
    ).plain == "default"


def test_quick_choice_handles_stale_value_explicitly(monkeypatch):
    saved = []
    monkeypatch.setattr(
        settings_command,
        "save_config",
        lambda config: saved.append(dict(config)),
    )
    config = {"reasoning_effort": "medium"}
    row = {
        "key": "reasoning_effort",
        "kind": "choice",
        "label": "Reasoning effort",
        "choices": (("", "default"),),
    }

    updated, message = settings_command._settings_apply_quick(row, config)

    assert updated["reasoning_effort"] == ""
    assert message == "saved Reasoning effort: default"
    assert saved[-1]["reasoning_effort"] == ""


def test_settings_startup_persists_reconciled_effort(monkeypatch):
    config = {
        "provider": "openai",
        "model": "gpt-4o",
        "reasoning_effort": "medium",
    }
    saved = []
    rendered = []

    class NonInteractiveInput:
        @staticmethod
        def isatty():
            return False

    monkeypatch.setattr(settings_command, "load_config", lambda: config)
    monkeypatch.setattr(
        settings_command,
        "save_config",
        lambda value: saved.append(dict(value)),
    )
    monkeypatch.setattr(
        settings_command,
        "_settings_plain",
        lambda value: rendered.append(dict(value)),
    )
    monkeypatch.setattr(settings_command.sys, "stdin", NonInteractiveInput())

    settings_command.cmd_settings()

    assert config["reasoning_effort"] == ""
    assert saved[-1]["reasoning_effort"] == ""
    assert rendered[-1]["reasoning_effort"] == ""
