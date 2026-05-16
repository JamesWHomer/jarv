import json
import sys
from pathlib import Path

from .display import console

CONFIG_DIR = Path.home() / ".jarv"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarv, a helpful CLI assistant. "
    "You can run shell commands when needed to answer questions or complete tasks. "
    "Be concise and direct. "
    "When the user asks about jarv commands, behavior, config, updating, or usage, "
    "run `jarv help` before answering. Do not invent unsupported commands."
)

DEFAULT_CONFIG = {
    "provider": "openai",
    "api_key": "",
    "api_keys": {},
    "base_url": "",
    "model": "gpt-5.4-mini",
    "reasoning_effort": "",
    "max_history": 40,
    "command_timeout": 60,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "max_subagent_depth": 4,
    "subagent_thread_pool_max_workers": 8,
    "check_updates": True,
}

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
        console.print(f"[red]Config file was invalid JSON:[/red] {e}")
        console.print(f"[yellow]Backed it up to[/yellow] {backup}")
        console.print(f"[green]Created a fresh config at[/green] {CONFIG_FILE}")
        sys.exit(1)
    except (OSError, UnicodeDecodeError) as e:
        console.print(f"[red]Could not read config:[/red] {e}")
        sys.exit(1)
    if not isinstance(config, dict):
        console.print(f"[red]Config must be a JSON object:[/red] {CONFIG_FILE}")
        sys.exit(1)
    changed = False
    for k, v in DEFAULT_CONFIG.items():
        if k not in config:
            config[k] = v
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
    from .provider import LOCAL_PROVIDERS, resolve_api_key

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
        console.print(f"[red]Could not save config:[/red] {e}")
        sys.exit(1)


def validate_config(config: dict) -> bool:
    ok = True
    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        console.print("[red]Config 'model' must be a non-empty string.[/red]")
        ok = False

    effort = config.get("reasoning_effort", "")
    if effort is None:
        config["reasoning_effort"] = ""

    for key in ("max_history", "command_timeout"):
        try:
            value = int(config.get(key, DEFAULT_CONFIG[key]))
            if value <= 0:
                raise ValueError
            config[key] = value
        except (TypeError, ValueError):
            console.print(f"[red]Config '{key}' must be a positive integer.[/red]")
            ok = False

    return ok
