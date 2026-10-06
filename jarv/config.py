import json
import sys

from .config_schema import (
    COMMAND_SAFETY_CHOICES,
    DEFAULT_SYSTEM_PROMPT,
    LEGACY_READ_ONLY_COMMAND_DISPLAY_CHOICES,
    READ_ONLY_COMMAND_DISPLAY_CHOICES,
    TOOL_CALL_DISPLAY_CHOICES,
    TOOL_NAMES,
    build_default_config,
    get_setting,
    setting_default,
    validate_config_fields,
)
from .paths import CONFIG_DIR, CONFIG_FILE
from .storage import StorageError, read_json, transaction, write_json

DEFAULT_CONFIG = build_default_config()

__all__ = [
    "COMMAND_SAFETY_CHOICES",
    "CONFIG_DIR",
    "CONFIG_FILE",
    "DEFAULT_CONFIG",
    "DEFAULT_SYSTEM_PROMPT",
    "LEGACY_READ_ONLY_COMMAND_DISPLAY_CHOICES",
    "READ_ONLY_COMMAND_DISPLAY_CHOICES",
    "TOOL_CALL_DISPLAY_CHOICES",
    "TOOL_NAMES",
    "build_default_config",
    "get_setting",
    "is_setup_complete",
    "load_config",
    "save_config",
    "setting_default",
    "validate_config",
]


def _console():
    from .display import console

    return console

def load_config() -> dict:
    """Load a settings snapshot whose baseline survives other reads and saves."""
    from .history import migrate_flat_session_files

    try:
        migrate_flat_session_files()
        # Recover interrupted commits before reading; keep first-run creation
        # and migrations under the same lock as ordinary settings saves.
        with transaction(CONFIG_FILE):
            config = _load_config()
    except StorageError as e:
        _console().print(f"[red]Could not load config:[/red] {e}")
        sys.exit(1)

    # Every entry point funnels through here -- including the slash commands
    # dispatched before cli.main() loads the run config -- so this is the one
    # place that reaches /help, /settings, and the heads-up TUI alike.
    from .display import configure_monochrome
    from .tui_panel import configure_menu_border

    configure_monochrome(not get_setting(config, "colour"))
    configure_menu_border(get_setting(config, "headsup_border"))

    return config


def _load_config() -> dict:
    """Initialize or migrate settings while the caller holds the storage lock."""
    config = read_json(CONFIG_FILE, {}, dict)
    changed = config.baseline is None
    if "monochrome" in config:
        config.setdefault("colour", not config.pop("monochrome"))
        changed = True

    for k, v in build_default_config().items():
        if k not in config:
            config[k] = v
            changed = True

    if config.get("read_only_command_display") in LEGACY_READ_ONLY_COMMAND_DISPLAY_CHOICES:
        config["read_only_command_display"] = "fullscreen"
        changed = True

    # Reject malformed containers and flags before migrations or consumers use
    # them (for example, a string "false" is truthy in Python).
    if not validate_config_fields(config, report=_console().print):
        sys.exit(1)

    # Migrate legacy flat api_key → per-provider api_keys
    if config.get("api_key") and not config.get("api_keys"):
        provider = config.get("provider", "openai")
        config.setdefault("api_keys", {})[provider] = config["api_key"]
        config["api_key"] = ""
        changed = True

    if changed:
        save_config(config)
    return config


def is_setup_complete(config: dict | None = None) -> bool:
    from .provider_auth import resolve_api_key
    from .provider_catalog import LOCAL_PROVIDERS

    if config is None:
        if CONFIG_FILE.exists():
            try:
                config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                config = {}
        else:
            config = {}

    if not isinstance(config, dict):
        return False
    provider = config.get("provider", "openai")
    if not isinstance(provider, str):
        return False
    if provider in LOCAL_PROVIDERS:
        return True
    keys = config.get("api_keys", {})
    if not isinstance(keys, dict):
        return False
    key = resolve_api_key(config)
    return isinstance(key, str) and bool(key.strip())


def save_config(config: dict) -> None:
    """Atomically merge snapshot edits, rejecting changes to the same setting."""
    try:
        write_json(CONFIG_FILE, dict(config), merge=True,
                   snapshot=config if hasattr(config, "baseline") else None)
    except StorageError as e:
        _console().print(f"[red]Could not save config:[/red] {e}")
        sys.exit(1)


def validate_config(config: dict) -> bool:
    from .provider_catalog import SERVICE_TIERS, PROVIDER_SERVICE_TIERS, service_tier_error

    console = _console()
    ok = validate_config_fields(config, report=console.print)
    if not ok:
        return False

    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        console.print("[red]Config 'model' must be a non-empty string.[/red]")
        ok = False

    effort = config.get("reasoning_effort", "")
    if effort is None:
        config["reasoning_effort"] = ""
    elif isinstance(effort, str):
        normalized_effort = effort.strip().lower()
        config["reasoning_effort"] = (
            "" if normalized_effort == "default" else normalized_effort
        )
    from .reasoning import reasoning_effort_error

    effort_error = reasoning_effort_error(config)
    if effort_error:
        console.print(f"[red]Invalid reasoning effort:[/red] {effort_error}.")
        ok = False

    service_tiers = config.get("service_tiers", {})
    if isinstance(service_tiers, dict):
        for provider, tier in service_tiers.items():
            if tier not in SERVICE_TIERS:
                choices = ", ".join(SERVICE_TIERS)
                console.print(
                    f"[red]Config service tier for '{provider}' must be one of: {choices}.[/red]"
                )
                ok = False
            elif tier not in PROVIDER_SERVICE_TIERS.get(str(provider), ("standard",)):
                choices = ", ".join(PROVIDER_SERVICE_TIERS.get(str(provider), ("standard",)))
                console.print(
                    f"[red]Provider '{provider}' supports service tiers: {choices}.[/red]"
                )
                ok = False

    tier_error = service_tier_error(config)
    if tier_error:
        console.print(f"[red]{tier_error}[/red]")
        ok = False
    return ok
