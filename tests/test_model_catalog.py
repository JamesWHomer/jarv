import json
import threading
import time

import httpx
import pytest

from jarv import model_catalog, settings_command
from jarv.anthropic_http import list_models as list_anthropic_models
from jarv.config import DEFAULT_CONFIG
from jarv.gemini_http import list_models as list_gemini_models
from jarv.model_catalog import CatalogModel, get_default_model, recommend_models


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


def test_openrouter_uses_stable_family_aliases_not_variants():
    choices = recommend_models("openrouter", _models(
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-opus-4.8-fast",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-haiku-4.5",
    ))

    assert [model for model, _description in choices] == [
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-4.6",
        "anthropic/claude-haiku-4.5",
    ]


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


def test_settings_opens_model_editor_from_cache_without_refresh(monkeypatch):
    calls = []

    def choices(_config):
        calls.append("cached")
        return [("gpt-5.6", "Flagship")]

    monkeypatch.setattr(model_catalog, "get_cached_model_choices", choices)
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
    assert edit["catalog_status"] == "loading"


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
