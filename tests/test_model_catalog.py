import json

import httpx

from jarv import model_catalog, settings_command
from jarv.anthropic_http import list_models as list_anthropic_models
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


def test_settings_refreshes_catalog_when_model_editor_opens(monkeypatch):
    calls = []

    def choices(_config, *, refresh=False):
        calls.append(refresh)
        return [("gpt-5.6", "Flagship")]

    monkeypatch.setattr(model_catalog, "get_model_choices", choices)
    row = {"key": "model", "kind": "text", "label": "Model"}

    edit = settings_command._settings_begin_edit(
        row,
        {"provider": "openai", "model": "gpt-5.5"},
    )

    assert calls == [True]
    assert edit["model_choices"] == [("gpt-5.6", "Flagship")]


def test_single_column_model_descriptions_follow_model_names():
    lines = settings_command._settings_choice_grid_lines(
        [
            (1, "gpt-5.5", "Flagship - latest GPT"),
            (2, "gpt-5.4-mini", "Balanced - latest GPT mini"),
            (3, "gpt-5.4-nano", "Budget - latest GPT nano"),
        ],
        140,
        max_columns=1,
    )

    rendered = [line.plain for line in lines]
    assert rendered[0].index("Flagship") < 25
    assert rendered[1].index("Balanced") == rendered[0].index("Flagship")
    assert rendered[2].index("Budget") == rendered[0].index("Flagship")


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
