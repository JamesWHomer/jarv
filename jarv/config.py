import json
import sys
from pathlib import Path

CONFIG_DIR = Path.home() / ".jarv"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarv, a helpful CLI assistant. "
    "You can run shell commands when needed to answer questions or complete tasks. "
    "Be concise and direct. "
    "When the user asks about jarv commands, behavior, config, updating, or usage, "
    "run `jarv /help` before answering. Do not invent unsupported commands."
)

READ_ONLY_COMMAND_DISPLAY_CHOICES = ("fullscreen", "print")
LEGACY_READ_ONLY_COMMAND_DISPLAY_CHOICES = ("auto", "inline")

DEFAULT_CONFIG = {
    "provider": "openai",
    "api_key": "",
    "api_keys": {},
    "base_url": "",
    "model": "gpt-5.4-mini",
    "reasoning_effort": "",
    "max_history": 40,
    "context_budget_ratio": 0.75,
    "context_compaction_threshold": 0.85,
    "context_output_reserve_ratio": 0.15,
    "context_window_fallback": 128_000,
    "max_stdin_chars": 200000,
    "max_tool_output_chars": 20000,
    "command_timeout": 60,
    "command_safety": "risky",
    "audit": True,
    "auditor_auto_approve": True,
    "auditor_model": "",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "max_subagent_depth": 4,
    "subagent_thread_pool_max_workers": 8,
    "check_updates": True,
    "read_only_command_display": "fullscreen",
    "print_usage_after_agent": False,
}


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
        _console().print(f"[red]Could not save config:[/red] {e}")
        sys.exit(1)


def validate_config(config: dict) -> bool:
    ok = True
    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        _console().print("[red]Config 'model' must be a non-empty string.[/red]")
        ok = False

    effort = config.get("reasoning_effort", "")
    if effort is None:
        config["reasoning_effort"] = ""

    for key in ("max_history", "max_stdin_chars", "max_tool_output_chars", "command_timeout"):
        try:
            value = int(config.get(key, DEFAULT_CONFIG[key]))
            if value <= 0:
                raise ValueError
            config[key] = value
        except (TypeError, ValueError):
            _console().print(f"[red]Config '{key}' must be a positive integer.[/red]")
            ok = False

    safety = config.get("command_safety", "risky")
    if safety not in ("all", "risky", "none"):
        _console().print(f"[red]Config 'command_safety' must be one of: all, risky, none.[/red]")
        ok = False

    display_mode = config.get("read_only_command_display", DEFAULT_CONFIG["read_only_command_display"])
    if display_mode not in READ_ONLY_COMMAND_DISPLAY_CHOICES:
        choices = ", ".join(READ_ONLY_COMMAND_DISPLAY_CHOICES)
        _console().print(f"[red]Config 'read_only_command_display' must be one of: {choices}.[/red]")
        ok = False

    try:
        depth = int(config.get("max_subagent_depth", DEFAULT_CONFIG["max_subagent_depth"]))
        if depth < 0:
            raise ValueError
        config["max_subagent_depth"] = depth
    except (TypeError, ValueError):
        _console().print("[red]Config 'max_subagent_depth' must be a non-negative integer.[/red]")
        ok = False

    try:
        workers = int(config.get(
            "subagent_thread_pool_max_workers",
            DEFAULT_CONFIG["subagent_thread_pool_max_workers"],
        ))
        if workers <= 0:
            raise ValueError
        config["subagent_thread_pool_max_workers"] = workers
    except (TypeError, ValueError):
        _console().print("[red]Config 'subagent_thread_pool_max_workers' must be a positive integer.[/red]")
        ok = False

    for key in (
        "context_budget_ratio",
        "context_compaction_threshold",
        "context_output_reserve_ratio",
    ):
        try:
            value = float(config.get(key, DEFAULT_CONFIG[key]))
            if not (0.0 < value < 1.0):
                raise ValueError
            config[key] = value
        except (TypeError, ValueError):
            _console().print(f"[red]Config '{key}' must be a number between 0 and 1.[/red]")
            ok = False

    try:
        fallback = int(config.get(
            "context_window_fallback",
            DEFAULT_CONFIG["context_window_fallback"],
        ))
        if fallback <= 0:
            raise ValueError
        config["context_window_fallback"] = fallback
    except (TypeError, ValueError):
        _console().print("[red]Config 'context_window_fallback' must be a positive integer.[/red]")
        ok = False

    return ok
