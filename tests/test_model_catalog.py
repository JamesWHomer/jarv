import io
import json
import threading
import time

import httpx
import pytest
from rich.console import Console

from jarv import model_catalog, settings_command, setup
from jarv.anthropic_http import list_models as list_anthropic_models
from jarv.config import DEFAULT_CONFIG
from jarv.gemini_http import list_models as list_gemini_models
from jarv.model_catalog import (
    CatalogModel,
    get_default_model,
    get_image_output_capability,
    recommend_models,
)


def _models(*ids):
    return [CatalogModel(id=model_id, created=index) for index, model_id in enumerate(ids, 1)]


def test_openai_recommendations_update_each_tier_independently():
    choices = recommend_models("openai", _models(
        "gpt-5.5",
        "gpt-5.6",
        "gpt-5.4-mini",
        "gpt-5.6-mini-2026-07-01",
        "gpt-5.4-nano",
        "gpt-image-2",
        "gpt-5.6-pro",
    ))

    assert [model for model, _description in choices] == [
        "gpt-5.6",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ]


def test_default_model_uses_the_providers_preferred_live_tier():
    assert get_default_model(
        {"provider": "openai"},
        choices=[
            ("gpt-5.6", "Flagship - latest GPT"),
            ("gpt-5.6-mini", "Balanced - latest GPT mini"),
        ],
    ) == "gpt-5.6-mini"
    assert get_default_model(
        {"provider": "anthropic"},
        choices=[
            ("claude-fable-5", "Premium - latest Claude Fable"),
            ("claude-sonnet-4-7", "Balanced - latest Claude Sonnet"),
        ],
    ) == "claude-sonnet-4-7"


def test_anthropic_recommendations_include_fable_and_snapshot_fallback():
    choices = recommend_models("anthropic", _models(
        "claude-fable-5",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "claude-haiku-4-4-20250801",
    ))

    assert [model for model, _description in choices] == [
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ]


def test_gemini_recommendations_choose_latest_generation_per_tier():
    choices = recommend_models("gemini", _models(
        "gemini-2.5-pro",
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-embedding-001",
    ))

    assert [model for model, _description in choices] == [
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
    ]


def test_deepseek_recommendations_ignore_unrelated_models():
    choices = recommend_models("deepseek", _models(
        "deepseek-chat",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-embedding",
    ))

    assert [model for model, _description in choices] == [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
    ]


def test_openrouter_recommendations_are_diverse_and_exclude_variants():
    choices = recommend_models("openrouter", _models(
        "openrouter/auto",
        "openrouter/free",
        "openrouter/owl-alpha",
        "tencent/hy3-preview",
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-opus-4.8-fast",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.5",
        "google/gemini-3.1-pro-preview",
        "deepseek/deepseek-v4-pro",
        "moonshotai/kimi-k2.6",
        "minimax/minimax-m2.7",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    ))

    assert [model for model, _description in choices] == [
        "openrouter/auto",
        "openrouter/free",
        "openrouter/owl-alpha",
        "tencent/hy3-preview",
        "anthropic/claude-fable-5",
        "openai/gpt-5.5",
        "anthropic/claude-sonnet-4.6",
        "google/gemini-3.1-pro-preview",
        "deepseek/deepseek-v4-pro",
        "moonshotai/kimi-k2.6",
        "minimax/minimax-m2.7",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    ]
    assert "anthropic/claude-opus-4.8-fast" not in {
        model for model, _description in choices
    }


def test_openrouter_router_and_free_pricing_labels():
    assert model_catalog.model_pricing_values(
        "openrouter",
        "openrouter/auto",
    ) == ("varies", "varies", "varies")
    assert model_catalog.model_pricing_values(
        "openrouter",
        "openrouter/free",
    ) == ("$0", "$0", "$0")
    assert model_catalog.model_pricing_values(
        "openrouter",
        "google/gemma-4-31b-it:free",
    ) == ("$0", "$0", "$0")


def test_marketplace_provider_policies_select_live_families():
    together = recommend_models("together", _models(
        "deepseek-ai/DeepSeek-V4-Pro",
        "meta-llama/Llama-4-Maverick-17B",
        "Qwen/Qwen3.5-9B",
    ))
    fireworks = recommend_models("fireworks", _models(
        "accounts/fireworks/models/kimi-k2p6",
        "accounts/fireworks/models/minimax-m2p7",
        "accounts/fireworks/models/qwen3-8b",
    ))

    assert len(together) == 3
    assert len(fireworks) == 3


def test_local_provider_lists_every_installed_model():
    choices = recommend_models("ollama", _models("qwen3", "llama3.3"))

    assert choices == [
        ("llama3.3", "Installed model"),
        ("qwen3", "Installed model"),
    ]


def test_catalog_uses_disk_cache_when_refresh_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        model_catalog,
        "discover_openrouter_models",
        lambda _config: [
            CatalogModel(
                id="openai/gpt-5.6",
                metadata={
                    "pricing": {
                        "prompt": "0.000001",
                        "completion": "0.000005",
                    },
                },
            ),
        ],
    )
    monkeypatch.setattr(
        model_catalog,
        "discover_models",
        lambda _config: _models("gpt-5.6", "gpt-5.5-mini", "gpt-5.4-nano"),
    )
    model_catalog.clear_memory_cache()

    live = model_catalog.get_model_choices({"provider": "openai"}, refresh=True)

    def fail(_config):
        raise OSError("offline")

    monkeypatch.setattr(model_catalog, "discover_models", fail)
    model_catalog.clear_memory_cache()
    cached = model_catalog.get_model_choices({"provider": "openai"}, refresh=True)

    assert cached == live
    assert json.loads((tmp_path / "openai.json").read_text())["provider"] == "openai"
    assert json.loads((tmp_path / "openrouter.json").read_text())["provider"] == "openrouter"


def test_openrouter_pricing_resolves_models_for_all_providers(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("openrouter", [
        CatalogModel(id="openai/gpt-5.5"),
        CatalogModel(id="anthropic/claude-sonnet-4.6"),
        CatalogModel(id="google/gemini-3-flash-preview"),
        CatalogModel(id="deepseek/deepseek-v4-flash"),
        CatalogModel(id="openai/gpt-oss-120b"),
        CatalogModel(id="meta-llama/llama-3.3-70b-instruct"),
        CatalogModel(id="meta-llama/llama-3.1-8b-instruct"),
        CatalogModel(id="meta-llama/llama-4-maverick"),
        CatalogModel(id="moonshotai/kimi-k2.6"),
        CatalogModel(id="minimax/minimax-m2.7"),
        CatalogModel(id="qwen/qwen3-8b"),
    ])

    cases = [
        ("openai", "gpt-5.5", "openai/gpt-5.5"),
        ("anthropic", "claude-sonnet-4-6", "anthropic/claude-sonnet-4.6"),
        ("gemini", "gemini-3-flash-preview", "google/gemini-3-flash-preview"),
        ("deepseek", "deepseek-v4-flash", "deepseek/deepseek-v4-flash"),
        ("groq", "openai/gpt-oss-120b", "openai/gpt-oss-120b"),
        ("groq", "llama-3.3-70b-versatile", "meta-llama/llama-3.3-70b-instruct"),
        ("groq", "llama-3.1-8b-instant", "meta-llama/llama-3.1-8b-instruct"),
        (
            "together",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "meta-llama/llama-4-maverick",
        ),
        ("fireworks", "accounts/fireworks/models/kimi-k2p6", "moonshotai/kimi-k2.6"),
        ("fireworks", "accounts/fireworks/models/minimax-m2p7", "minimax/minimax-m2.7"),
        ("fireworks", "accounts/fireworks/models/qwen3-8b", "qwen/qwen3-8b"),
        ("openrouter", "openai/gpt-5.5", "openai/gpt-5.5"),
    ]

    for provider, selected_model, expected in cases:
        resolved = model_catalog.resolve_openrouter_model(provider, selected_model)
        assert resolved is not None
        assert resolved.id == expected


def test_image_capability_uses_openrouter_input_modalities(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("openrouter", [
        CatalogModel(
            id="openai/gpt-5.5",
            metadata={"architecture": {"input_modalities": ["text", "image"]}},
        ),
        CatalogModel(
            id="openai/gpt-5.4-mini",
            metadata={"architecture": {"input_modalities": ["text"]}},
        ),
    ])

    supported = get_image_output_capability({
        "provider": "openai",
        "model": "gpt-5.5",
    })
    unsupported = get_image_output_capability({
        "provider": "openai",
        "model": "gpt-5.4-mini",
    })

    assert supported.supported is True
    assert supported.output_format == "responses"
    assert unsupported.supported is False


def test_image_capability_rejects_ambiguous_openrouter_routes(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("openrouter", [
        CatalogModel(
            id="openrouter/auto",
            metadata={"architecture": {"input_modalities": ["text", "image"]}},
        ),
    ])

    capability = get_image_output_capability({
        "provider": "openrouter",
        "model": "openrouter/auto",
    })

    assert capability.supported is False
    assert "ambiguous OpenRouter route" in capability.reason


def test_image_capability_uses_anthropic_image_input_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("anthropic", [
        CatalogModel(
            id="claude-sonnet-4-6",
            metadata={"capabilities": {"image_input": {"supported": True}}},
        ),
    ])

    capability = get_image_output_capability({
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
    })

    assert capability.supported is True
    assert capability.output_format == "anthropic"


def test_image_capability_allows_only_gemini_3_function_responses(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    supported = get_image_output_capability({
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
    })
    unsupported = get_image_output_capability({
        "provider": "gemini",
        "model": "gemini-2.5-flash",
    })

    assert supported.supported is True
    assert supported.output_format == "gemini"
    assert unsupported.supported is False
    assert "multimodal function responses" in unsupported.reason


def test_image_capability_unknown_models_default_unsupported(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)

    capability = get_image_output_capability({
        "provider": "openai",
        "model": "unknown-model",
    })

    assert capability.supported is False
    assert "does not advertise image input capability" in capability.reason


def test_model_picker_pricing_formats_rates(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("openrouter", [
        CatalogModel(
            id="openai/gpt-5.5",
            metadata={
                "pricing": {
                    "prompt": "0.000005",
                    "input_cache_read": "0.0000005",
                    "completion": "0.00003",
                },
            },
        ),
    ])
    assert model_catalog.model_choice_description(
        "openai",
        "gpt-5.5",
        "Flagship - latest GPT",
    ) == (
        "$5.00 / $0.50 / $30.00 | Flagship - latest GPT"
    )
    assert model_catalog.model_pricing_values("openai", "gpt-5.5") == (
        "$5.00",
        "$0.50",
        "$30.00",
    )
    assert model_catalog.model_pricing_values("openai", "missing") == (
        "n/a",
        "n/a",
        "n/a",
    )


def test_settings_model_picker_shows_openrouter_pricing(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("openrouter", [
        CatalogModel(
            id="openai/gpt-5.5",
            metadata={
                "pricing": {
                    "prompt": "0.000005",
                    "completion": "0.00003",
                },
            },
        ),
    ])
    edit = {
        "row": {"key": "model", "kind": "text", "label": "Model"},
        "buffer": "gpt-5.5",
        "model_choices": [("gpt-5.5", "Flagship - latest GPT")],
    }
    settings_command.initialize_text_editor(edit, "gpt-5.5")

    rendered = "\n".join(
        line.plain
        for line in settings_command._settings_editor_lines(
            edit,
            {"provider": "openai", "model": "gpt-5.5"},
            180,
        )
    )

    assert "OpenRouter pricing per 1M" not in rendered
    assert "Model list is current" not in rendered
    assert "Model" in rendered
    assert "Input" in rendered
    assert "Cached" in rendered
    assert "Output" in rendered
    assert "Tier" in rendered
    assert "$5.00" in rendered
    assert "n/a" in rendered
    assert "$30.00" in rendered
    lines = rendered.splitlines()
    header = next(line for line in lines if "Input" in line)
    model = next(line for line in lines if "gpt-5.5" in line)
    assert header.index("Model") == model.index("gpt-5.5")
    assert header.index("Input") + len("Input") == model.index("$5.00") + len("$5.00")
    assert header.index("Cached") + len("Cached") == model.index("n/a") + len("n/a")
    assert header.index("Output") + len("Output") == model.index("$30.00") + len("$30.00")
    assert header.index("Tier") == model.index("Flagship")
    assert "\u203a 1. gpt-5.5" in rendered
    assert "\n\n  Model number, name, or custom model:" in rendered


def test_model_picker_columns_align_with_long_prices(monkeypatch):
    prices = {
        "deepseek-v4-pro": ("$0.43", "$0.003625", "$0.87"),
        "deepseek-v4-flash": ("$0.098", "$0.020", "$0.20"),
    }
    monkeypatch.setattr(
        model_catalog,
        "model_pricing_values",
        lambda _provider, model: prices[model],
    )

    lines = settings_command._settings_model_choice_lines(
        [
            ("deepseek-v4-pro", "Flagship"),
            ("deepseek-v4-flash", "Budget"),
        ],
        "deepseek",
        1,
        False,
        78,
    )
    header, pro, flash = [line.plain for line in lines]

    for heading, pro_value, flash_value in (
        ("Input", "$0.43", "$0.098"),
        ("Cached", "$0.003625", "$0.020"),
        ("Output", "$0.87", "$0.20"),
    ):
        expected_end = header.index(heading) + len(heading)
        assert pro.index(pro_value) + len(pro_value) == expected_end
        assert flash.index(flash_value) + len(flash_value) == expected_end
    assert header.index("Tier") == pro.index("Flagship")
    assert header.index("Tier") == flash.index("Budget")
    assert len(header) == len(pro) == len(flash) == 78

    narrow = settings_command._settings_model_choice_lines(
        [
            ("deepseek-v4-pro", "Flagship"),
            ("deepseek-v4-flash", "Budget"),
        ],
        "deepseek",
        1,
        False,
        60,
    )
    assert len({len(line.plain) for line in narrow}) == 1
    assert all(len(line.plain) == 60 for line in narrow)


def test_settings_model_picker_appends_and_selects_current_provider_model(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("openai", [
        CatalogModel(id="gpt-5.5"),
        CatalogModel(id="gpt-5.4-mini"),
        CatalogModel(id="gpt-5.4-nano"),
        CatalogModel(id="gpt-4o"),
    ])
    monkeypatch.setattr(
        model_catalog,
        "get_cached_model_choices",
        lambda _config: [
            ("gpt-5.5", "Flagship"),
            ("gpt-5.4-mini", "Balanced"),
            ("gpt-5.4-nano", "Budget"),
        ],
    )
    config = {"provider": "openai", "model": "gpt-4o"}

    edit = settings_command._settings_begin_edit(
        {"key": "model", "kind": "text", "label": "Model"},
        config,
    )
    rendered = "\n".join(
        line.plain
        for line in settings_command._settings_editor_lines(edit, config, 120)
    )

    assert edit["model_choices"][-1] == ("gpt-4o", "")
    assert "\u203a 4. gpt-4o" in rendered
    assert settings_command._settings_resolve_model(
        config,
        "4",
        models=edit["model_choices"],
    ) == "gpt-4o"
    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    edit["buffer"] = "4"
    updated, _message, _style, done = settings_command._settings_commit_edit(
        edit,
        config,
    )
    assert done is True
    assert updated["model"] == "gpt-4o"


def test_settings_model_picker_does_not_append_model_from_another_provider(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("openai", [CatalogModel(id="gpt-5.5")])
    monkeypatch.setattr(
        model_catalog,
        "get_cached_model_choices",
        lambda _config: [("gpt-5.5", "Flagship")],
    )

    edit = settings_command._settings_begin_edit(
        {"key": "model", "kind": "text", "label": "Model"},
        {"provider": "openai", "model": "claude-sonnet-4-6"},
    )

    assert edit["model_choices"] == [("gpt-5.5", "Flagship")]


def test_model_picker_arrows_select_rows_and_typing_activates_input(
    monkeypatch,
):
    monkeypatch.setattr(
        model_catalog,
        "get_cached_model_choices",
        lambda _config: [
            ("gpt-5.5", "Flagship"),
            ("gpt-5.4-mini", "Balanced"),
            ("gpt-5.4-nano", "Budget"),
        ],
    )
    monkeypatch.setattr(
        model_catalog,
        "cached_provider_has_model",
        lambda _config, _model: False,
    )
    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    config = {"provider": "openai", "model": "gpt-5.5"}
    edit = settings_command._settings_begin_edit(
        {"key": "model", "kind": "text", "label": "Model"},
        config,
    )

    assert edit["selected_model_index"] == 0
    assert edit["model_input_active"] is False
    assert edit["buffer"] == ""

    assert settings_command._settings_model_apply_key(edit, "DOWN") is True
    assert edit["selected_model_index"] == 1
    assert edit["model_input_active"] is False

    updated, _message, _style, done = settings_command._settings_commit_edit(
        edit,
        config,
    )
    assert done is True
    assert updated["model"] == "gpt-5.4-mini"

    assert settings_command._settings_model_apply_key(edit, "c") is True
    assert edit["model_input_active"] is True
    assert edit["buffer"] == "c"
    assert settings_command._settings_model_apply_key(edit, "u") is True
    assert edit["buffer"] == "cu"
    lines = settings_command._settings_editor_lines(edit, updated, 120)
    selected_line = next(line for line in lines if "gpt-5.4-mini" in line.plain)
    input_line = next(
        line for line in lines
        if "Model number, name, or custom model:" in line.plain
    )
    assert all(span.style != "bold bright_white" for span in selected_line.spans)
    assert any(span.style == "bold bright_white" for span in input_line.spans)

    assert settings_command._settings_model_apply_key(edit, "UP") is True
    assert edit["selected_model_index"] == 2
    assert edit["model_input_active"] is False
    assert edit["buffer"] == "cu"

    assert settings_command._settings_model_apply_key(edit, "DOWN") is True
    assert edit["selected_model_index"] == 2
    assert edit["model_input_active"] is True
    assert edit["buffer"] == "cu"


def test_model_picker_inserts_batched_text():
    edit = {
        "model_choices": [("gpt-5.5", "Flagship")],
        "selected_model_index": 0,
        "model_input_active": False,
        "buffer": "",
        "cursor": 0,
    }

    assert settings_command._settings_model_apply_key(
        edit,
        settings_command.TextInput("custom/model"),
    )
    assert edit["model_input_active"] is True
    assert edit["buffer"] == "custom/model"
    assert edit["cursor"] == len("custom/model")


def test_model_picker_down_past_last_row_focuses_preserved_input():
    edit = {
        "model_choices": [
            ("gpt-5.5", "Flagship"),
            ("gpt-5.4-mini", "Balanced"),
            ("gpt-5.4-nano", "Budget"),
        ],
        "selected_model_index": 2,
        "model_input_active": False,
        "buffer": "custom-model",
        "cursor": 6,
    }

    assert settings_command._settings_model_apply_key(edit, "DOWN") is True
    assert edit["selected_model_index"] == 2
    assert edit["model_input_active"] is True
    assert edit["buffer"] == "custom-model"
    assert edit["cursor"] == 6

    assert settings_command._settings_model_apply_key(edit, "UP") is True
    assert edit["selected_model_index"] == 2
    assert edit["model_input_active"] is False
    assert edit["buffer"] == "custom-model"
    assert edit["cursor"] == 6

    assert settings_command._settings_model_apply_key(edit, "DOWN") is True
    assert edit["model_input_active"] is True
    assert edit["buffer"] == "custom-model"
    assert edit["cursor"] == 6


def test_model_picker_typed_number_uses_number_resolution(monkeypatch):
    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    config = {"provider": "openai", "model": "gpt-5.5"}
    edit = {
        "row": {"key": "model", "kind": "text", "label": "Model"},
        "model_choices": [
            ("gpt-5.5", "Flagship"),
            ("gpt-5.4-mini", "Balanced"),
            ("gpt-5.4-nano", "Budget"),
            ("gpt-4o", ""),
        ],
        "selected_model_index": 0,
        "model_input_active": False,
        "buffer": "",
        "cursor": 0,
    }

    settings_command._settings_model_apply_key(edit, "4")
    updated, _message, _style, done = settings_command._settings_commit_edit(
        edit,
        config,
    )

    assert done is True
    assert updated["model"] == "gpt-4o"


def test_custom_model_validates_against_cached_provider_catalog(monkeypatch):
    saved = []
    monkeypatch.setattr(
        model_catalog,
        "cached_provider_has_model",
        lambda _config, model: model == "gpt-valid-custom",
    )
    monkeypatch.setattr(
        settings_command,
        "save_config",
        lambda config: saved.append(dict(config)),
    )
    config = {"provider": "openai", "model": "gpt-5.5"}
    edit = {
        "row": {"key": "model", "kind": "text", "label": "Model"},
        "model_choices": [("gpt-5.5", "Flagship")],
        "selected_model_index": 0,
        "model_input_active": True,
        "buffer": "gpt-valid-custom",
        "cursor": len("gpt-valid-custom"),
    }

    updated, _message, _style, done = settings_command._settings_commit_edit(
        edit,
        config,
    )

    assert done is True
    assert updated["model"] == "gpt-valid-custom"
    assert saved[-1]["model"] == "gpt-valid-custom"


def test_unknown_custom_model_warns_then_can_continue(monkeypatch):
    saved = []
    monkeypatch.setattr(
        model_catalog,
        "cached_provider_has_model",
        lambda _config, _model: False,
    )
    monkeypatch.setattr(
        settings_command,
        "save_config",
        lambda config: saved.append(dict(config)),
    )
    config = {"provider": "openai", "model": "gpt-5.5"}
    edit = {
        "row": {"key": "model", "kind": "text", "label": "Model"},
        "model_choices": [("gpt-5.5", "Flagship")],
        "selected_model_index": 0,
        "model_input_active": True,
        "buffer": "gpt-unknown",
        "cursor": len("gpt-unknown"),
    }

    updated, message, style, done = settings_command._settings_commit_edit(
        edit,
        config,
    )

    assert done is False
    assert style == "yellow"
    assert "not found" in message
    assert updated["model"] == "gpt-5.5"
    assert saved == []
    assert edit["model_validation_warning"] == "gpt-unknown"
    assert edit["model_warning_selection"] == 0

    rendered = "\n".join(
        line.plain
        for line in settings_command._settings_editor_lines(edit, config, 120)
    )
    assert "Not found in OpenAI's cached models." in rendered
    assert "\u203a Keep editing" in rendered
    assert "Use anyway" in rendered
    assert rendered.splitlines()[-2:] == [
        "  Not found in OpenAI's cached models.",
        "  \u203a Keep editing     Use anyway",
    ]

    assert settings_command._settings_model_apply_key(edit, "RIGHT") is True
    assert edit["model_warning_selection"] == 1
    updated, message, style, done = settings_command._settings_commit_edit(
        edit,
        config,
    )

    assert done is True
    assert style == "yellow"
    assert message == "saved Model: gpt-unknown"
    assert updated["model"] == "gpt-unknown"
    assert saved[-1]["model"] == "gpt-unknown"


def test_unknown_custom_model_suggests_clear_cached_match(monkeypatch):
    saved = []
    monkeypatch.setattr(
        model_catalog,
        "cached_provider_has_model",
        lambda _config, _model: False,
    )
    monkeypatch.setattr(
        model_catalog,
        "cached_provider_model_ids",
        lambda _config: ["gpt-5.5", "gpt-5.4-mini", "gpt-5.4-nano"],
    )
    monkeypatch.setattr(
        settings_command,
        "save_config",
        lambda config: saved.append(dict(config)),
    )
    config = {"provider": "openai", "model": "gpt-5.4-mini"}
    edit = {
        "row": {"key": "model", "kind": "text", "label": "Model"},
        "model_choices": [("gpt-5.5", "Flagship")],
        "selected_model_index": 0,
        "model_input_active": True,
        "buffer": "gpt-5.5x",
        "cursor": len("gpt-5.5x"),
    }

    _updated, _message, _style, done = settings_command._settings_commit_edit(
        edit,
        config,
    )

    assert done is False
    assert edit["model_validation_suggestion"] == "gpt-5.5"
    assert edit["model_warning_actions"] == [
        {"label": "Use gpt-5.5", "value": "gpt-5.5"},
        {"label": "Keep editing", "value": "edit"},
        {"label": "Use gpt-5.5x anyway", "value": "continue"},
    ]
    lines = settings_command._settings_editor_lines(edit, config, 120)
    rendered = "\n".join(line.plain for line in lines)
    input_line = next(
        line for line in lines
        if "Model number, name, or custom model:" in line.plain
    )
    assert "Not found. Did you mean gpt-5.5?" in rendered
    assert "\u203a Use gpt-5.5" in rendered
    assert all(str(span.style) != "reverse" for span in input_line.spans)
    assert any(str(span.style) == "dim" for span in input_line.spans)

    updated, message, style, done = settings_command._settings_commit_edit(
        edit,
        config,
    )

    assert done is True
    assert style == "yellow"
    assert message == "saved Model: gpt-5.5"
    assert updated["model"] == "gpt-5.5"
    assert saved[-1]["model"] == "gpt-5.5"


def test_model_suggestion_rejects_ambiguous_match(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "cached_provider_model_ids",
        lambda _config: ["gpt-5.4-mini", "gpt-5.4-nano"],
    )

    assert settings_command._settings_model_suggestion(
        {"provider": "openai"},
        "gpt-5.4",
    ) == ""


def test_model_suggestion_uses_canonical_prefix_family(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "cached_provider_model_ids",
        lambda _config: [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-instruct",
            "gpt-3.5-turbo-0125",
        ],
    )

    assert settings_command._settings_model_suggestion(
        {"provider": "openai"},
        "gpt-3.5",
    ) == "gpt-3.5-turbo"


def test_model_suggestion_rejects_ambiguous_prefix_families(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "cached_provider_model_ids",
        lambda _config: ["gpt-5.5", "gpt-5.4-mini", "gpt-5.4-nano"],
    )

    assert settings_command._settings_model_suggestion(
        {"provider": "openai"},
        "gpt-5",
    ) == ""


def test_unknown_custom_model_can_return_to_editing(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "cached_provider_has_model",
        lambda _config, _model: False,
    )
    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    config = {"provider": "openai", "model": "gpt-5.5"}
    edit = {
        "row": {"key": "model", "kind": "text", "label": "Model"},
        "model_choices": [("gpt-5.5", "Flagship")],
        "selected_model_index": 0,
        "model_input_active": True,
        "buffer": "gpt-unknown",
        "cursor": 4,
    }

    _updated, _message, _style, done = settings_command._settings_commit_edit(
        edit,
        config,
    )
    assert done is False

    _updated, message, style, done = settings_command._settings_commit_edit(
        edit,
        config,
    )

    assert done is False
    assert style == "dim"
    assert message == "Continue editing the model name."
    assert edit["buffer"] == "gpt-unknown"
    assert edit["cursor"] == 4
    assert "model_validation_warning" not in edit


def test_setup_model_shows_openrouter_pricing(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("openrouter", [
        CatalogModel(
            id="openai/gpt-5.5",
            metadata={
                "pricing": {
                    "prompt": "0.000005",
                    "completion": "0.00003",
                },
            },
        ),
    ])
    monkeypatch.setattr(
        model_catalog,
        "get_model_choices",
        lambda _config, refresh=False: [("gpt-5.5", "Flagship - latest GPT")],
    )
    monkeypatch.setattr(
        model_catalog,
        "get_default_model",
        lambda _config, choices=None: "gpt-5.5",
    )
    monkeypatch.setattr(setup.Prompt, "ask", lambda *args, **kwargs: "1")
    output = io.StringIO()
    monkeypatch.setattr(
        setup,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=180),
    )

    config = setup.setup_model({"provider": "openai"})

    rendered = output.getvalue()
    assert config["model"] == "gpt-5.5"
    assert "$5.00 / n/a / $30.00" in rendered
    assert "OpenRouter pricing per 1M" not in rendered


def test_settings_opens_model_editor_from_cache_without_refresh(monkeypatch):
    calls = []

    def choices(_config):
        calls.append("cached")
        return [("gpt-5.6", "Flagship")]

    monkeypatch.setattr(model_catalog, "get_cached_model_choices", choices)
    monkeypatch.setattr(
        model_catalog,
        "cached_provider_has_model",
        lambda _config, _model: False,
    )
    monkeypatch.setattr(
        model_catalog,
        "refresh_model_choices",
        lambda _config: pytest.fail("model editor performed a blocking refresh"),
    )
    row = {"key": "model", "kind": "text", "label": "Model"}

    edit = settings_command._settings_begin_edit(
        row,
        {"provider": "openai", "model": "gpt-5.5"},
    )

    assert calls == ["cached"]
    assert edit["model_choices"] == [("gpt-5.6", "Flagship")]


def test_model_update_notice_names_replacements_additions_and_removals():
    notice = settings_command._settings_model_update_notice(
        [
            ("gpt-5.5", "Flagship"),
            ("gpt-5.4-mini", "Balanced"),
            ("gpt-5.4-nano-preview", "Budget preview"),
        ],
        [
            ("gpt-5.6", "Flagship"),
            ("gpt-5.4-mini", "Balanced"),
            ("gpt-5.4-nano", "Budget"),
        ],
    )

    assert notice == (
        "Model list updated: gpt-5.5 \u2192 gpt-5.6; "
        "added gpt-5.4-nano; removed gpt-5.4-nano-preview"
    )
    assert settings_command._settings_model_update_notice(
        [("gpt-5.6", "Flagship")],
        [("gpt-5.6", "Flagship")],
    ) == ""
    assert settings_command._settings_model_update_notice(
        [("gpt-5.6", "Flagship"), ("gpt-5.4-mini", "Balanced")],
        [("gpt-5.4-mini", "Balanced"), ("gpt-5.6", "Flagship")],
    ) == ""


def test_cached_model_choices_never_attempt_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        model_catalog,
        "discover_models",
        lambda _config: pytest.fail("cached lookup attempted network discovery"),
    )
    model_catalog.clear_memory_cache()

    choices = model_catalog.get_cached_model_choices({"provider": "openai"})

    assert choices


def test_catalog_refresher_is_non_blocking_and_deduplicates(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    calls = []
    callbacks = []

    def refresh(_config):
        calls.append("refresh")
        started.set()
        assert release.wait(1)
        return [("gpt-5.6-mini", "Balanced")]

    monkeypatch.setattr(model_catalog, "refresh_model_choices", refresh)
    refresher = settings_command._ModelCatalogRefresher()

    before = time.monotonic()
    first = refresher.request(
        {"provider": "openai"},
        lambda provider, choices, generation: callbacks.append(
            (provider, choices, generation)
        ),
    )
    assert time.monotonic() - before < 0.1
    assert started.wait(1)

    second = refresher.request(
        {"provider": "openai"},
        lambda provider, choices, generation: (
            callbacks.append((provider, choices, generation)),
            completed.set(),
        ),
    )
    release.set()

    assert completed.wait(1)
    refresher.close()
    assert calls == ["refresh"]
    assert first < second
    assert callbacks == [
        ("openai", [("gpt-5.6-mini", "Balanced")], second),
    ]


def test_provider_commit_uses_cached_models_without_refresh(monkeypatch):
    provider_choices = {
        "openai": [
            ("gpt-5.6", "Flagship"),
            ("gpt-5.6-mini", "Balanced"),
        ],
        "anthropic": [
            ("claude-opus-4-8", "Flagship"),
            ("claude-sonnet-4-6", "Balanced"),
        ],
    }

    monkeypatch.setattr(
        model_catalog,
        "get_cached_model_choices",
        lambda config: provider_choices[config["provider"]],
    )
    monkeypatch.setattr(
        model_catalog,
        "refresh_model_choices",
        lambda _config: pytest.fail("provider commit performed a live refresh"),
    )
    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    edit = {
        "row": {"key": "provider", "kind": "setup", "label": "Provider"},
        "buffer": "",
        "selected_provider": "anthropic",
    }

    config, _message, _style, done = settings_command._settings_commit_edit(
        edit,
        {"provider": "openai", "model": "gpt-5.6-mini"},
    )

    assert done is True
    assert config["provider"] == "anthropic"
    assert config["model"] == "claude-sonnet-4-6"


def test_model_reset_uses_active_provider_default(monkeypatch):
    provider_choices = {
        "openai": [("gpt-5.4-mini", "Balanced")],
        "anthropic": [("claude-sonnet-4-6", "Balanced")],
    }
    saved = []

    monkeypatch.setattr(
        model_catalog,
        "get_cached_model_choices",
        lambda config: provider_choices[config["provider"]],
    )
    monkeypatch.setattr(
        settings_command,
        "save_config",
        lambda config: saved.append(dict(config)),
    )

    config = {
        **DEFAULT_CONFIG,
        "provider": "anthropic",
        "model": "custom-claude-model",
    }
    row = next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == "model"
    )

    action_bar = settings_command._settings_reset_action_bar(row, config, 100)
    assert "custom-claude-model \u2192 claude-sonnet-4-6" in action_bar.plain

    updated, message, style = settings_command._settings_finish_reset(row, config, "y")

    assert updated["model"] == "claude-sonnet-4-6"
    assert message == "reset Model"
    assert style == "cyan"
    assert saved[-1]["model"] == "claude-sonnet-4-6"


def test_single_column_model_descriptions_align_with_settings_description_column():
    inner_width = 140
    lines = settings_command._settings_choice_grid_lines(
        [
            (1, "gpt-5.5", "Flagship - latest GPT"),
            (2, "gpt-5.4-mini", "Balanced - latest GPT mini"),
            (3, "gpt-5.4-nano", "Budget - latest GPT nano"),
        ],
        inner_width,
        max_columns=1,
        align_descriptions=True,
    )

    rendered = [line.plain for line in lines]
    description_start = settings_command._settings_column_layout(inner_width)[3]
    assert rendered[0].index("Flagship") == description_start
    assert rendered[1].index("Balanced") == description_start
    assert rendered[2].index("Budget") == description_start


def test_provider_editor_aligns_keys_and_notes_with_settings_columns():
    inner_width = 140
    lines = settings_command._settings_provider_choice_lines(
        {"provider": "openai"},
        inner_width,
        selected_provider="openai",
    )
    openai = next(line.plain for line in lines if "OpenAI-hosted" in line.plain)
    _label_width, _value_width, value_start, description_start = (
        settings_command._settings_column_layout(inner_width)
    )

    assert openai.index("openai") == value_start
    assert openai.index("OpenAI-hosted") == description_start


def test_anthropic_model_listing_follows_pagination():
    requests = []

    def handler(request):
        requests.append(request)
        after_id = request.url.params.get("after_id")
        if after_id:
            return httpx.Response(200, json={
                "data": [{"id": "claude-sonnet-4-6"}],
                "has_more": False,
                "last_id": "claude-sonnet-4-6",
            })
        return httpx.Response(200, json={
            "data": [{"id": "claude-fable-5"}],
            "has_more": True,
            "last_id": "claude-fable-5",
        })

    client = httpx.Client(
        base_url="https://api.anthropic.test",
        transport=httpx.MockTransport(handler),
    )
    result = list_anthropic_models(client)

    assert [item["id"] for item in result["data"]] == [
        "claude-fable-5",
        "claude-sonnet-4-6",
    ]
    assert requests[1].url.params["after_id"] == "claude-fable-5"


def test_gemini_model_listing_follows_page_tokens():
    def handler(request):
        token = request.url.params.get("pageToken")
        if token:
            return httpx.Response(200, json={
                "models": [{"name": "models/gemini-3-flash-preview"}],
            })
        return httpx.Response(200, json={
            "models": [{"name": "models/gemini-3.1-pro-preview"}],
            "nextPageToken": "next",
        })

    client = httpx.Client(
        base_url="https://generativelanguage.test/v1beta",
        transport=httpx.MockTransport(handler),
    )
    result = list_gemini_models(client)

    assert [item["name"] for item in result["models"]] == [
        "models/gemini-3.1-pro-preview",
        "models/gemini-3-flash-preview",
    ]
