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
    CONFIG_DIR.mkdir(exist_ok=True)
    from .history import migrate_flat_session_files
    migrate_flat_session_files()
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        return dict(DEFAULT_CONFIG)
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        backup = CONFIG_FILE.with_suffix(".json.bak")
        CONFIG_FILE.replace(backup)
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        console = _console()
        console.print(f"[red]Config file was invalid JSON:[/red] {e}")
        console.print(f"[yellow]Backed it up to[/yellow] {backup}")
        console.print(f"[green]Created a fresh config at[/green] {CONFIG_FILE}")
        sys.exit(1)
    except (OSError, UnicodeDecodeError) as e:
        _console().print(f"[red]Could not read config:[/red] {e}")
        sys.exit(1)
    if not isinstance(config, dict):
        _console().print(f"[red]Config must be a JSON object:[/red] {CONFIG_FILE}")
        sys.exit(1)
    changed = False
    for k, v in DEFAULT_CONFIG.items():
        if k not in config:
            config[k] = v
            changed = True

    if config.get("read_only_command_display") in LEGACY_READ_ONLY_COMMAND_DISPLAY_CHOICES:
        config["read_only_command_display"] = "fullscreen"
        changed = True

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
            except (json.JSONDecodeError, OSError):
                config = {}
        else:
            config = {}

    provider = config.get("provider", "openai")
    if provider in LOCAL_PROVIDERS:
        return True
    if resolve_api_key(config):
        return True
    return False


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    try:
        CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except OSError as e:
        _console().print(f"[red]Could not save config:[/red] {e}")
        sys.exit(1)


def validate_config(config: dict) -> bool:
    from .provider_catalog import SERVICE_TIERS, service_tier_choices

    console = _console()
    ok = validate_config_fields(config, report=console.print)

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
            elif tier not in service_tier_choices(str(provider)):
                choices = ", ".join(service_tier_choices(str(provider)))
                console.print(
                    f"[red]Provider '{provider}' supports service tiers: {choices}.[/red]"
                )
                ok = False

    return ok
