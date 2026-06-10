from jarv import commands
from jarv.provider import KEY_PATTERNS, LOCAL_PROVIDERS, PROVIDERS
from jarv.provider_catalog import FALLBACK_PROVIDER_MODELS, PROVIDER_CHOICES


def test_command_facade_exports_moved_handlers():
    assert commands.cmd_sessions.__module__ == "jarv.session_commands"
    assert commands.cmd_history.__module__ == "jarv.session_commands"
    assert commands.cmd_usage.__module__ == "jarv.usage_command"
    assert commands.cmd_settings.__module__ == "jarv.settings_command"
    assert commands.cmd_undo.__module__ == "jarv.undo_commands"
    assert commands.cmd_redo.__module__ == "jarv.undo_commands"


def test_provider_catalog_covers_setup_choices():
    provider_keys = set(PROVIDERS)
    choice_keys = {key for key, _label, _default_model in PROVIDER_CHOICES}

    assert choice_keys <= provider_keys
    assert LOCAL_PROVIDERS <= provider_keys
    assert set(KEY_PATTERNS) <= provider_keys

    for provider, _label, default_model in PROVIDER_CHOICES:
        preset_models = {
            model
            for model, _description in FALLBACK_PROVIDER_MODELS.get(provider, [])
        }
        assert provider in LOCAL_PROVIDERS or default_model in preset_models
