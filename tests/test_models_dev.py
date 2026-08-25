import json

import httpx
import pytest

from conftest import model_facts
from jarv import models_dev
from jarv.provider_catalog import MODELS_DEV_PROVIDERS


def _mock_httpx(monkeypatch, handler):
    """Serve refresh() from a handler instead of the network."""
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )


def test_prune_keeps_wanted_providers_and_known_fields():
    payload = {
        "openai": {
            "id": "openai",
            "name": "OpenAI",
            "env": ["OPENAI_API_KEY"],
            "npm": "@ai-sdk/openai",
            "models": {
                "gpt-5.5": {
                    "id": "gpt-5.5",
                    "cost": {"input": 5, "output": 30},
                    "limit": {"context": 1_050_000, "output": 128_000},
                    "benchmarks": [{"name": "SWE-Bench", "score": 80}],
                    "description": "a long paragraph nobody reads",
                },
            },
        },
        "some-reseller": {"id": "some-reseller", "models": {"x": {"id": "x"}}},
    }

    pruned = models_dev.prune_catalog(payload, {"openai"})

    assert set(pruned) == {"openai"}
    assert "npm" not in pruned["openai"]
    assert pruned["openai"]["env"] == ["OPENAI_API_KEY"]
    model = pruned["openai"]["models"]["gpt-5.5"]
    assert set(model) == {"id", "cost", "limit"}


def test_prune_survives_malformed_entries():
    payload = {
        "openai": {"models": "not a dict"},
        "groq": {"models": {"good": {"id": "good"}, "bad": ["not", "a", "dict"]}},
        "anthropic": "not a dict at all",
    }

    pruned = models_dev.prune_catalog(payload, {"openai", "groq", "anthropic"})

    assert set(pruned) == {"groq"}
    assert set(pruned["groq"]["models"]) == {"good"}


def test_lookup_matches_exact_case_and_snapshot_variants(models_dev_catalog):
    models_dev_catalog({
        "anthropic": {"claude-opus-4-8": model_facts()},
        "togetherai": {"deepseek-ai/DeepSeek-V4-Pro": model_facts()},
    })

    assert models_dev.lookup("anthropic", "claude-opus-4-8").id == "claude-opus-4-8"
    assert models_dev.lookup("anthropic", "claude-opus-4-8-20260528").id == "claude-opus-4-8"
    assert models_dev.lookup(
        "together",
        "deepseek-ai/deepseek-v4-pro",
    ).id == "deepseek-ai/DeepSeek-V4-Pro"


def test_lookup_puts_the_openrouter_namespace_back_on(models_dev_catalog):
    models_dev_catalog({
        "openrouter": {"openai/gpt-5.5": model_facts(cost={"input": 5, "output": 30})},
    })

    facts = models_dev.lookup("openrouter", "gpt-5.5")

    assert facts is not None
    assert facts.id == "openai/gpt-5.5"
    assert facts.native is True


def test_ambiguous_matches_resolve_to_the_paid_entry(models_dev_catalog):
    models_dev_catalog({
        "openrouter": {
            "google/gemma-4-31b-it": model_facts(cost={"input": 0.1, "output": 0.2}),
            "google/gemma-4-31b-it:free": model_facts(cost={"input": 0, "output": 0}),
        },
    })

    facts = models_dev.lookup("openrouter", "gemma-4-31b-it")

    assert facts is not None
    assert facts.id == "google/gemma-4-31b-it"


def test_another_providers_entry_is_not_native_and_has_no_rates(models_dev_catalog):
    models_dev_catalog({
        "deepseek": {
            "deepseek-v4-pro": model_facts(
                cost={"input": 0.435, "output": 0.87},
                modalities={"input": ["text"], "output": ["text"]},
                limit={"context": 1_000_000, "output": 384_000},
            ),
        },
    })

    facts = models_dev.lookup("groq", "deepseek-v4-pro")

    assert facts is not None
    assert facts.native is False
    assert facts.context_limit == 1_000_000
    assert facts.rates() is None
    assert models_dev.prices("groq", "deepseek-v4-pro") is None


def test_local_providers_have_no_catalog_entry(models_dev_catalog):
    models_dev_catalog({"openai": {"gpt-5.5": model_facts()}})

    assert models_dev.lookup("ollama", "gpt-5.5") is not None  # capabilities only
    assert models_dev.prices("ollama", "gpt-5.5") is None
    assert models_dev.models_for("ollama") == []
    assert models_dev.nearest_family("ollama", "gpt-5.5") is None


def test_context_tiers_switch_prices_at_the_threshold(models_dev_catalog):
    models_dev_catalog({
        "openai": {
            "gpt-5.5": model_facts(
                cost={
                    "input": 5,
                    "output": 30,
                    "cache_read": 0.5,
                    "tiers": [
                        {
                            "input": 10,
                            "output": 45,
                            "cache_read": 1,
                            "tier": {"type": "context", "size": 272_000},
                        },
                    ],
                },
            ),
        },
    })

    assert models_dev.prices("openai", "gpt-5.5", input_tokens=272_000)["input"] == 5
    assert models_dev.prices("openai", "gpt-5.5", input_tokens=272_001)["input"] == 10
    assert models_dev.prices("openai", "gpt-5.5", input_tokens=272_001)["cached_input"] == 1
    assert models_dev.prices("openai", "gpt-5.5")["input"] == 5


def test_legacy_context_surcharge_is_honoured(models_dev_catalog):
    models_dev_catalog({
        "google": {
            "gemini-3.1-pro-preview": model_facts(
                cost={
                    "input": 2,
                    "output": 12,
                    "context_over_200k": {"input": 4, "output": 18},
                },
            ),
        },
    })

    assert models_dev.prices("gemini", "gemini-3.1-pro-preview", input_tokens=1_000)["input"] == 2
    assert models_dev.prices("gemini", "gemini-3.1-pro-preview", input_tokens=300_000)["input"] == 4


def test_models_without_a_price_yield_no_rates(models_dev_catalog):
    models_dev_catalog({"openrouter": {"openrouter/auto": model_facts(cost={})}})

    assert models_dev.lookup("openrouter", "openrouter/auto") is not None
    assert models_dev.prices("openrouter", "openrouter/auto") is None


def test_nearest_family_answers_for_models_newer_than_the_catalog(models_dev_catalog):
    models_dev_catalog({
        "openai": {
            "gpt-5.4-mini": model_facts(
                family="gpt-mini",
                release_date="2026-03-05",
                reasoning=True,
            ),
            "gpt-5.2-mini": model_facts(
                family="gpt-mini",
                release_date="2025-12-11",
                reasoning=True,
            ),
            "gpt-5.5": model_facts(family="gpt", release_date="2026-04-23"),
        },
    })

    facts = models_dev.nearest_family("openai", "gpt-5.9-mini")

    assert facts is not None
    assert facts.id == "gpt-5.4-mini"
    # A family match describes behaviour, never price.
    assert facts.native is False
    assert facts.rates() is None
    assert models_dev.nearest_family("openai", "something-unrelated") is None


def test_refresh_writes_a_pruned_cache_that_overlays_the_snapshot(
    tmp_path,
    monkeypatch,
    models_dev_catalog,
):
    models_dev_catalog({"openai": {"gpt-5.5": model_facts(cost={"input": 5, "output": 30})}})

    def handler(request):
        assert "If-None-Match" not in request.headers
        return httpx.Response(
            200,
            json={
                "openai": {
                    "id": "openai",
                    "models": {
                        "gpt-5.5": {"id": "gpt-5.5", "cost": {"input": 4, "output": 20}},
                    },
                },
                "not-a-jarv-provider": {"id": "x", "models": {"y": {"id": "y"}}},
            },
            headers={"ETag": '"abc123"'},
        )

    _mock_httpx(monkeypatch, handler)

    assert models_dev.refresh({}) is True
    assert models_dev.prices("openai", "gpt-5.5")["input"] == 4

    written = json.loads(models_dev.CACHE_PATH.read_text(encoding="utf-8"))
    assert written["etag"] == '"abc123"'
    assert set(written["providers"]) == {"openai"}


def test_refresh_revalidates_with_the_stored_etag(tmp_path, monkeypatch, models_dev_catalog):
    models_dev_catalog({"openai": {"gpt-5.5": model_facts(cost={"input": 5, "output": 30})}})
    models_dev.CACHE_PATH.write_text(
        json.dumps({
            "etag": '"cached"',
            "providers": {
                "openai": {
                    "id": "openai",
                    "models": {"gpt-5.5": {"id": "gpt-5.5", "cost": {"input": 9, "output": 9}}},
                },
            },
        }),
        encoding="utf-8",
    )
    models_dev.clear_cache()
    seen = []

    def handler(request):
        seen.append(request.headers.get("If-None-Match"))
        return httpx.Response(304)

    _mock_httpx(monkeypatch, handler)

    assert models_dev.refresh({}) is True
    assert seen == ['"cached"']
    assert models_dev.prices("openai", "gpt-5.5")["input"] == 9


@pytest.mark.parametrize("failure", [httpx.ConnectError("offline"), httpx.Response(500)])
def test_refresh_failures_leave_the_catalog_alone(monkeypatch, models_dev_catalog, failure):
    models_dev_catalog({"openai": {"gpt-5.5": model_facts(cost={"input": 5, "output": 30})}})

    def handler(_request):
        if isinstance(failure, Exception):
            raise failure
        return failure

    _mock_httpx(monkeypatch, handler)

    assert models_dev.refresh({}) is False
    assert models_dev.prices("openai", "gpt-5.5")["input"] == 5


def test_refresh_rejects_a_payload_with_no_usable_providers(monkeypatch, models_dev_catalog):
    models_dev_catalog({"openai": {"gpt-5.5": model_facts(cost={"input": 5, "output": 30})}})

    _mock_httpx(monkeypatch, lambda _request: httpx.Response(200, json={"junk": {}}))

    assert models_dev.refresh({}) is False
    assert models_dev.prices("openai", "gpt-5.5")["input"] == 5


def test_bundled_snapshot_covers_every_cloud_provider(bundled_catalog_only):
    catalog = models_dev.catalog()

    assert set(MODELS_DEV_PROVIDERS.values()) <= set(catalog)
    for provider_id, entry in catalog.items():
        assert entry["models"], provider_id
        for model_id, model in entry["models"].items():
            limit = model.get("limit") or {}
            assert isinstance(limit.get("context"), int), f"{provider_id}/{model_id}"


def test_bundled_snapshot_prices_the_default_model_of_each_cloud_provider(
    bundled_catalog_only,
):
    from jarv.provider_catalog import LOCAL_PROVIDERS, PROVIDER_CHOICES

    unpriced = [
        f"{provider}/{default_model}"
        for provider, _label, default_model in PROVIDER_CHOICES
        if provider not in LOCAL_PROVIDERS
        and models_dev.prices(provider, default_model) is None
    ]

    assert unpriced == []
