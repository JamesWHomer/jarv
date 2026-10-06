"""Settings actions preserve the same validation and discard semantics as edits."""

from types import SimpleNamespace

import pytest

from jarv import settings_command, settings_editor
from jarv.config_schema import build_default_config


@pytest.mark.parametrize("key", [
    "context_budget_ratio", "context_compaction_threshold", "context_output_reserve_ratio",
])
def test_oversized_ratio_is_a_validation_error(key):
    from jarv.config_schema import validate_config_fields

    config = build_default_config()
    config[key] = 10 ** 400
    errors = []

    assert not validate_config_fields(config, report=errors.append)
    assert len(errors) == 1 and key in errors[0]


def test_clearing_stored_key_requires_discard_confirmation():
    config = build_default_config()
    config["api_keys"] = {"openai": "stored-secret"}
    row = next(row for row in settings_command._settings_rows(config) if row["key"] == "api_key")
    edit = settings_command._settings_begin_edit(row, config)
    catalog = SimpleNamespace(cancel_pending=lambda: None, request=lambda *a, **k: None)

    _, outcome = settings_editor.apply_editor_key(
        edit, config, "BACKSPACE", 1, catalog=catalog, inner_width=80,
    )
    assert outcome.kind == "continue"
    _, outcome = settings_editor.apply_editor_key(
        edit, config, "ESC", 1, catalog=catalog, inner_width=80,
    )
    assert outcome.kind == "continue"
    assert edit["discard_armed"] is True
    _, outcome = settings_editor.apply_editor_key(
        edit, config, "ESC", 1, catalog=catalog, inner_width=80,
    )
    assert outcome.kind == "cancelled"
    assert config["api_keys"] == {"openai": "stored-secret"}


def test_reset_model_reconciles_incompatible_reasoning(monkeypatch, models_dev_catalog):
    from conftest import model_facts

    models_dev_catalog({"openai": {
        "reasoning-model": model_facts(reasoning=True, reasoning_options=[
            {"type": "effort", "values": ["high"]},
        ]),
        "plain-model": model_facts(reasoning=False),
    }})
    config = build_default_config()
    config.update(model="reasoning-model", reasoning_effort="high")
    row = next(row for row in settings_command._settings_rows(config) if row["key"] == "model")
    monkeypatch.setattr(settings_command, "_settings_default_model", lambda config: "plain-model")
    saved = []
    monkeypatch.setattr(settings_command, "save_config", lambda config: saved.append(dict(config)))

    result, message = settings_command._settings_reset_row(row, config)

    assert saved and result["model"] == "plain-model"
    assert result["reasoning_effort"] == ""
    assert "reasoning effort reset" in message
