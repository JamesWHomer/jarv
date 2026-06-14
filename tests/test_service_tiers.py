from jarv import settings_command
from jarv.anthropic_http import build_payload as build_anthropic_payload
from jarv.config import DEFAULT_CONFIG, validate_config
from jarv.gemini_http import build_payload as build_gemini_payload
from jarv.openai_http import build_chat_payload, build_responses_payload
from jarv.provider_catalog import (
    configured_service_tier,
    provider_service_tier,
    service_tier_choices,
)


def test_service_tier_defaults_to_standard_and_translates_per_provider():
    assert configured_service_tier({"provider": "openai"}) == "standard"
    assert provider_service_tier({"provider": "openai"}) == "default"
    assert provider_service_tier({"provider": "openrouter"}) is None
    assert provider_service_tier({"provider": "gemini"}) is None
    assert provider_service_tier({"provider": "anthropic"}) == "standard_only"
    assert provider_service_tier({"provider": "groq"}) is None


def test_provider_service_tier_choices_hide_unsupported_modes():
    assert service_tier_choices("openai") == ("standard", "flex", "priority")
    assert service_tier_choices("openrouter") == ("standard", "flex", "priority")
    assert service_tier_choices("gemini") == ("standard", "flex", "priority")
    assert service_tier_choices("anthropic") == ("standard", "priority")
    assert service_tier_choices("groq") == ("standard",)


def test_openai_payloads_include_native_service_tier():
    responses = build_responses_payload(
        "model",
        "system",
        [],
        [],
        service_tier="priority",
    )
    chat = build_chat_payload(
        "model",
        [],
        service_tier="flex",
    )

    assert responses["service_tier"] == "priority"
    assert chat["service_tier"] == "flex"


def test_native_provider_payloads_translate_service_tier():
    anthropic = build_anthropic_payload(
        {"provider": "anthropic", "service_tiers": {"anthropic": "priority"}},
        "claude",
        "",
        [],
        [{"role": "user", "content": "hi"}],
    )
    gemini = build_gemini_payload(
        {"provider": "gemini", "service_tiers": {"gemini": "flex"}},
        "gemini",
        "",
        [],
        [{"role": "user", "content": "hi"}],
    )

    assert anthropic["service_tier"] == "auto"
    assert gemini["service_tier"] == "flex"


def test_settings_store_tiers_per_provider_without_mutating_defaults(monkeypatch):
    monkeypatch.setattr(settings_command, "save_config", lambda _config: None)
    config = dict(DEFAULT_CONFIG)
    row = next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == "service_tier"
    )

    config, _message = settings_command._settings_apply_quick(row, config)
    assert config["service_tiers"] == {"openai": "flex"}
    assert DEFAULT_CONFIG["service_tiers"] == {}

    config["provider"] = "anthropic"
    anthropic_row = next(
        row for row in settings_command._settings_rows(config)
        if row["key"] == "service_tier"
    )
    assert [value for value, _label in anthropic_row["choices"]] == [
        "standard",
        "priority",
    ]
    config, _message = settings_command._settings_apply_quick(anthropic_row, config)
    assert config["service_tiers"] == {
        "openai": "flex",
        "anthropic": "priority",
    }


def test_settings_hide_processing_tier_for_unsupported_provider():
    config = {
        **DEFAULT_CONFIG,
        "provider": "groq",
        "service_tiers": {"openai": "priority"},
    }

    assert all(
        row["key"] != "service_tier"
        for row in settings_command._settings_rows(config)
    )
    assert config["service_tiers"] == {"openai": "priority"}


def test_validate_config_rejects_invalid_or_unsupported_service_tiers():
    invalid = dict(DEFAULT_CONFIG)
    invalid["service_tiers"] = {"openai": "auto"}
    assert not validate_config(invalid)

    unsupported = dict(DEFAULT_CONFIG)
    unsupported["service_tiers"] = {"anthropic": "flex"}
    assert not validate_config(unsupported)
