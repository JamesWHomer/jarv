"""Ultrafast selection, transport, auxiliary calls, and billing contracts."""

import copy
import json

import httpx
import pytest

from jarv import auditor, cli, commands, model_catalog, provider, settings_command
from jarv.config import DEFAULT_CONFIG, validate_config
from jarv.headsup import _model_status
from jarv.provider_catalog import configured_service_tier
from jarv.settings_schema import settings_service_tier_choices
from jarv.usage import load_usage, record_response_usage


@pytest.fixture
def astra():
    return {
        **copy.deepcopy(DEFAULT_CONFIG),
        "provider": "openai", "model": "gpt-6-astra",
        "service_tiers": {"openai": "ultrafast"},
    }


def test_cli_selection_reaches_responses_and_preserves_reasoning(astra):
    original = {**astra, "service_tiers": {}}
    args = cli._build_parser().parse_args(["--service-tier", "ultrafast"])
    config = cli._apply_cli_overrides(original, args)
    assert validate_config(config)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        event = {"type": "response.completed", "response": {
            "id": "resp_test", "status": "completed", "service_tier": "ultrafast",
            "output": [], "usage": {"input_tokens": 10, "output_tokens": 2},
        }}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content="data: " + json.dumps(event) + "\n\n")

    with httpx.Client(base_url="https://api.openai.com/v1", transport=httpx.MockTransport(handler)) as client:
        events = list(provider.stream_response(
            client, config, config["model"], "system", [],
            [{"role": "user", "content": "hi"}], reasoning={"effort": "high"},
        ))
    assert requests[0]["model"] == "gpt-6-astra"
    assert requests[0]["service_tier"] == "ultrafast"
    assert requests[0]["reasoning"] == {"effort": "high"}
    assert events[-1].response["service_tier"] == "ultrafast"
    assert original["service_tiers"] == {}
    assert _model_status(config).endswith(" / ultrafast")


@pytest.mark.parametrize("changes", [
    {"model": "gpt-6-sol"},
    {"model": "gpt-6-astra-fake"},
    {"base_url": "https://gateway.example/v1"},
    {"base_url": "https://eu.api.openai.com/v1"},
    {"provider": "openrouter", "service_tiers": {"openrouter": "ultrafast"}},
    {"provider": "gemini", "service_tiers": {"gemini": "ultrafast"}},
])
def test_unsupported_choices_rejected_in_cli_and_config(astra, changes):
    config = {**astra, **changes}
    assert not validate_config(config)
    assert "ultrafast" not in dict(settings_service_tier_choices(config))
    args = cli._build_parser().parse_args(["--service-tier", "ultrafast"])
    with pytest.raises(ValueError, match="not supported"):
        cli._apply_cli_overrides(config, args)


def test_inactive_openai_preference_is_preserved(astra):
    config = {**astra, "provider": "anthropic", "model": "claude-sonnet-5"}
    assert validate_config(config)
    assert configured_service_tier(config) == "standard"
    assert config["service_tiers"] == {"openai": "ultrafast"}


def test_actual_request_model_cannot_inherit_incompatible_tier(astra):
    with pytest.raises(provider.ProviderError, match="gpt-6-sol"):
        list(provider.stream_response(object(), astra, "gpt-6-sol", "", [], []))


@pytest.mark.parametrize("arguments", [
    ["--model", "gpt-6-sol"],
    ["--base-url", "http://localhost:8000/v1"],
    ["-c", "model=gpt-6-sol"],
])
def test_cli_switch_resets_saved_tier_without_changing_persistent_config(astra, arguments, capsys):
    config = cli._apply_cli_overrides(astra, cli._build_parser().parse_args(arguments))
    assert config["service_tiers"]["openai"] == "standard"
    assert astra["service_tiers"]["openai"] == "ultrafast"
    assert "reset to standard" in capsys.readouterr().err


def test_explicit_config_override_is_not_silently_reconciled(astra):
    args = cli._build_parser().parse_args([
        "--model", "gpt-6-sol", "-c", 'service_tiers={"openai":"ultrafast"}',
    ])
    config = cli._apply_cli_overrides(astra, args)
    assert not validate_config(config)
    assert config["service_tiers"]["openai"] == "ultrafast"


@pytest.mark.parametrize("warning", [False, True])
def test_settings_model_switch_visibly_resets_ultrafast(astra, monkeypatch, warning):
    saved = []
    monkeypatch.setattr(settings_command, "save_config", lambda config: saved.append(copy.deepcopy(config)))
    edit = {
        "row": {"key": "model", "kind": "text", "label": "Model"},
        "model_choices": [("gpt-6-sol", "")], "selected_model_index": 0,
        "model_input_active": False, "buffer": "",
    }
    if warning:
        edit.update(model_validation_warning="gpt-6-sol", model_warning_actions=[{"value": "continue"}])
    updated, message, _style, done = settings_command._settings_commit_edit(edit, astra)
    assert done
    assert updated["service_tiers"]["openai"] == "standard"
    assert saved[-1]["service_tiers"]["openai"] == "standard"
    assert "processing tier reset to standard" in message


def test_settings_offer_and_persist_ultrafast_for_astra(astra, monkeypatch):
    monkeypatch.setattr(settings_command, "save_config", lambda config: None)
    astra["service_tiers"]["openai"] = "priority"
    row = next(row for row in settings_command._settings_rows(astra) if row["key"] == "service_tier")
    assert "6x" in row["desc"]
    updated, message = settings_command._settings_apply_quick(row, astra)
    assert updated["service_tiers"]["openai"] == "ultrafast"
    assert "ultrafast" in message


def test_set_model_reconciles_saved_tier(astra, monkeypatch):
    saved = []
    monkeypatch.setattr("jarv.config.load_config", lambda: astra)
    monkeypatch.setattr("jarv.config.save_config", lambda config: saved.append(config))
    assert commands.cmd_set(["model", "gpt-6-sol"]) == 0
    assert saved[-1]["service_tiers"]["openai"] == "standard"
    assert astra["service_tiers"]["openai"] == "ultrafast"


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.4-mini"])
def test_auditor_uses_and_records_standard_for_initial_and_retry_calls(astra, monkeypatch, tmp_path, model):
    calls = []
    def create_chat(_client, payload, **kwargs):
        calls.append(payload)
        content = "unparseable" if len(calls) == 1 else '{"allow": true, "reason": "safe"}'
        return {"choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    monkeypatch.setattr(auditor, "_get_auditor_client", lambda *args: object())
    monkeypatch.setattr("jarv.openai_http.create_chat", create_chat)
    path = tmp_path / "usage.json"
    result = auditor._call_openai_compat(
        astra, model, "git status", {"base_url": None}, usage_path=path,
        session_id="session", global_usage_path=tmp_path / "global.jsonl",
    )
    assert result == (True, "safe")
    assert len(calls) == 2
    assert all(call["service_tier"] == "default" for call in calls)
    assert all(call["model"] == model and "max_completion_tokens" in call for call in calls)
    assert load_usage(path, "session")["last_request"]["requested_service_tier"] == "standard"
    assert astra["service_tiers"]["openai"] == "ultrafast"


@pytest.mark.parametrize("input_tokens, rates", [
    (272_000, (60, 6, 75, 300)),
    (272_001, (120, 12, 150, 450)),
])
@pytest.mark.parametrize("source", ["root", "subagent"])
def test_ultrafast_usage_prices_cached_write_and_output_tokens(tmp_path, monkeypatch, input_tokens, rates, source):
    # Exercise the documented fallback independently of this machine's catalog.
    monkeypatch.setattr(model_catalog.models_dev, "prices", lambda *args, **kwargs: None)
    path = tmp_path / "usage.json"
    response = {"service_tier": "ultrafast", "usage": {
        "input_tokens": input_tokens, "output_tokens": 1_000,
        "input_tokens_details": {"cached_tokens": 100_000, "cache_write_tokens": 50_000},
    }}
    record_response_usage(path, "session", "gpt-6-astra", response, source,
                          provider="openai", requested_service_tier="ultrafast", record_global=False)
    usage = load_usage(path, "session")
    record = usage["last_request"]
    inp, cached, write, out = rates
    expected = ((input_tokens - 150_000) * inp + 100_000 * cached + 50_000 * write + 1_000 * out) / 1_000_000
    assert record["estimated_cost_usd"] == pytest.approx(expected)
    assert record["requested_service_tier"] == record["served_service_tier"] == "ultrafast"
    assert usage["totals"]["estimated_cost_usd"] == pytest.approx(expected)


@pytest.mark.parametrize("served, cost", [(None, None), ("default", 0.2), ("ultrafast", 1.2)])
def test_billing_uses_served_tier_or_stays_unknown(tmp_path, monkeypatch, served, cost):
    monkeypatch.setattr(model_catalog.models_dev, "prices", lambda *args, **kwargs: None)
    path = tmp_path / "usage.json"
    response = {"usage": {"input_tokens": 10_000, "output_tokens": 2_000}}
    if served:
        response["service_tier"] = served
    record_response_usage(path, "session", "gpt-6-astra", response, "root",
                          provider="openai", requested_service_tier="ultrafast", record_global=False)
    record = load_usage(path, "session")["last_request"]
    if cost is None:
        assert record["cost_status"] == "unknown"
        assert "estimated_cost_usd" not in record
    else:
        assert record["estimated_cost_usd"] == pytest.approx(cost)


def test_catalog_astra_prices_take_precedence(monkeypatch):
    prices = {"input": 9.0, "output": 45.0}
    monkeypatch.setattr(model_catalog.models_dev, "prices", lambda *args, **kwargs: prices)
    assert model_catalog.model_prices("openai", "gpt-6-astra") == prices
