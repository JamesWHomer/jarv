"""Malformed provider numbers must not erase usage or break model selection."""

import math

import pytest

from conftest import model_facts
from jarv import model_catalog, models_dev, reasoning, usage
from jarv.model_catalog import CatalogModel


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), True, -3])
def test_invalid_token_count_uses_valid_provider_alias(value):
    normalized = usage.usage_from_response({"usage": {
        "input_tokens": value, "prompt_tokens": 20,
        "output_tokens": 3,
    }})
    assert normalized["input_tokens"] == 20
    assert normalized["total_tokens"] == 23


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), 10 ** 400])
def test_invalid_provider_cost_falls_back_to_estimate(tmp_path, monkeypatch, value):
    monkeypatch.setattr(usage, "estimate_token_cost_usd", lambda *args: 0.25)
    path = tmp_path / "usage.json"
    usage.record_response_usage(
        path, "test-session", "model",
        {"usage": {"input_tokens": 20, "output_tokens": 3, "cost": value}},
        "root", provider="test", record_global=False,
    )
    record = usage.load_usage(path)["last_request"]
    assert record is not None
    assert record["cost_status"] == "estimated"
    assert record["estimated_cost_usd"] == 0.25
    assert "provider_cost_usd" not in record


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), True, 0.5])
def test_invalid_catalog_limits_remain_unknown(models_dev_catalog, value):
    models_dev_catalog({"openai": {"test-model": model_facts(
        limit={"context": value, "output": value},
    )}})
    facts = models_dev.lookup("openai", "test-model")
    assert facts.context_limit is None
    assert facts.output_limit is None


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), True])
def test_invalid_native_context_and_output_limits_do_not_override_catalog(
    tmp_path, monkeypatch, models_dev_catalog, value,
):
    models_dev_catalog({"anthropic": {"claude-test": model_facts()}})
    monkeypatch.setattr(model_catalog, "CACHE_DIR", tmp_path)
    model_catalog._write_cache("anthropic", [CatalogModel(
        id="claude-test", metadata={"max_input_tokens": value, "max_tokens": value},
    )])
    config = {"provider": "anthropic", "model": "claude-test"}
    assert usage.known_context_window("claude-test", config) == 128_000
    assert reasoning.get_reasoning_capabilities(config).max_output_tokens == 16_000


@pytest.mark.parametrize("field", ["input", "output", "cache_read", "cache_write"])
def test_overflowed_price_does_not_break_catalog_lookup(field):
    rates = models_dev._rates({"input": 1, "output": 2, field: 10 ** 400})
    assert rates is None or all(math.isfinite(rate) for rate in rates.values())


@pytest.mark.parametrize("threshold", [float("nan"), float("-inf"), True, -1])
def test_invalid_price_tier_threshold_does_not_apply_surcharge(threshold):
    rates = models_dev._rates({"input": 1, "output": 2, "tiers": [{
        "tier": {"type": "context", "size": threshold}, "input": 9,
    }]}, input_tokens=100)
    assert rates["input"] == 1


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
@pytest.mark.parametrize("keep_input", [False, True])
@pytest.mark.parametrize("invalid", [float("inf"), float("nan"), "invalid", True, -2])
def test_provider_normalization_preserves_reply_despite_invalid_usage(provider, keep_input, invalid):
    input_tokens, output_tokens = (20, invalid) if keep_input else (invalid, 3)
    if provider == "anthropic":
        from jarv.anthropic_http import normalize_response

        normalized = normalize_response({
            "content": [{"type": "text", "text": "Reply"}],
            "usage": {
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "cache_creation_input_tokens": invalid, "cache_read_input_tokens": invalid,
            },
        })
    else:
        from jarv.gemini_http import normalize_response

        normalized = normalize_response({
            "candidates": [{"content": {"parts": [{"text": "Reply"}]}}],
            "usageMetadata": {
                "promptTokenCount": input_tokens, "candidatesTokenCount": output_tokens,
                "cachedContentTokenCount": invalid, "thoughtsTokenCount": invalid,
                "totalTokenCount": invalid,
            },
        })

    assert normalized["output_text"] == "Reply"
    assert normalized["usage"]["input_tokens"] == (20 if keep_input else 0)
    assert normalized["usage"]["output_tokens"] == (0 if keep_input else 3)
    assert normalized["usage"]["cached_input_tokens"] == 0
    assert normalized["usage"]["total_tokens"] == (20 if keep_input else 3)
