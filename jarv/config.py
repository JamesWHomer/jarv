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
    "api_key": "",
    "model": "gpt-5.4-mini",
    "reasoning_effort": "",
    "max_history": 40,
    "command_timeout": 60,
    "history_scope": "global",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "max_subagent_depth": 4,
    "subagent_thread_pool_max_workers": 8,
    "check_updates": True,
}


def load_config() -> dict:
    CONFIG_DIR.mkdir(exist_ok=True)
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        console.print(f"[green]Config created at[/green] {CONFIG_FILE}")
        console.print("[dim]Set your OpenAI API key there or via the OPENAI_API_KEY env var.[/dim]")
        sys.exit(0)
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

    if changed:
        save_config(config)

    return config


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

    history_scope = config.get("history_scope", DEFAULT_CONFIG["history_scope"])
    if history_scope not in {"global", "terminal"}:
        console.print("[red]Config 'history_scope' must be 'global' or 'terminal'.[/red]")
        ok = False

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
