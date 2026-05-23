import json
import os
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

from rich.console import Group
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import CONFIG_DIR, CONFIG_FILE, DEFAULT_CONFIG, load_config, save_config
from .display import console, flatten_headings, jarv_panel, section_rule, status_line
from .history import (
    SESSIONS_DIR,
    SESSIONS_FILE,
    forget_current_session,
    load_history,
    prepare_session_context,
)
from .command_input import _read_key
from .session_commands import (
    archive_session_files,
    cmd_archive,
    cmd_history,
    cmd_sessions,
    delete_session_files,
    unarchive_session_files,
)
from .settings_command import cmd_settings
from .undo_commands import cmd_redo, cmd_undo
from .usage_command import cmd_usage


GITHUB_REPO = "JamesWHomer/jarv"
PYPI_VERSION_URL = "https://pypi.org/pypi/jarv/json"
UPDATE_FLAG_FILE = CONFIG_DIR / "update_available.txt"
LAST_CHECK_FILE = CONFIG_DIR / "last_update_check.txt"
UPDATE_CHECK_INTERVAL_HOURS = 24



def coerce_value(value: str):
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def cmd_set(args: list) -> None:
    if len(args) < 2:
        console.print(status_line("✗", "jarv /set <key> <value>", prefix_style="bold red", message_style="dim"))
        console.print(f"  [dim]Keys: {', '.join(DEFAULT_CONFIG.keys())}[/dim]")
        return
    key, raw = args[0], " ".join(args[1:])
    config = load_config()
    value = coerce_value(raw)
    config[key] = value
    save_config(config)
    display = "[dim]***[/dim]" if key == "api_key" else f"[green]{repr(value)}[/green]"
    console.print(f"[bold cyan]✓[/bold cyan] [bold cyan]{key}[/bold cyan] [dim]=[/dim] {display}")


def cmd_unset(args: list) -> None:
    if not args:
        console.print(status_line("✗", "jarv /unset <key>", prefix_style="bold red", message_style="dim"))
        return
    key = args[0]
    config = load_config()
    if key not in config:
        console.print(f"[yellow]○[/yellow] [bold]{key}[/bold] [dim]is not set.[/dim]")
        return
    if key in DEFAULT_CONFIG:
        config[key] = DEFAULT_CONFIG[key]
        save_config(config)
        console.print(f"[bold cyan]↺[/bold cyan] [bold cyan]{key}[/bold cyan] [dim]reset to default →[/dim] [green]{repr(DEFAULT_CONFIG[key])}[/green]")
    else:
        del config[key]
        save_config(config)
        console.print(f"[bold cyan]✓[/bold cyan] [bold cyan]{key}[/bold cyan] [dim]removed.[/dim]")


def print_help() -> None:
    cmd_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    cmd_table.add_column(style="bold cyan", no_wrap=True)
    cmd_table.add_column(style="white")
    cmd_table.add_row("jarv", "Start heads-up mode for repeated prompts")
    cmd_table.add_row("jarv <question>", "Ask jarv anything")
    cmd_table.add_row("jarv /set <key> <value>", "Set a config value")
    cmd_table.add_row("jarv /unset <key>", "Reset a config key to its default")
    cmd_table.add_row("jarv /new", "Start a fresh session on the next message")
    cmd_table.add_row("jarv /archive", "Archive this terminal's session and start a fresh one")
    cmd_table.add_row("jarv /sessions, /session", "List sessions (all in a TTY; 5 most recent when piped/non-TTY)")
    cmd_table.add_row("jarv /sessions <id>", "Load a specific session into this terminal by id prefix")
    cmd_table.add_row("jarv /history", "Show recent conversation history")
    cmd_table.add_row("jarv /usage", "Show token usage for this session")
    cmd_table.add_row("jarv /undo [n]", "Unsend the last n exchanges (default 1)")
    cmd_table.add_row("jarv /redo [n]", "Restore the last n undone exchanges (default 1)")
    cmd_table.add_row("jarv /settings", "Open the interactive settings menu")
    cmd_table.add_row("jarv /config", "Show raw config values")
    cmd_table.add_row("jarv /update", "Update jarv to the latest version")
    cmd_table.add_row("jarv /about", "Show detailed information about jarv")
    cmd_table.add_row("jarv /setup", "Run the setup wizard")
    cmd_table.add_row("jarv /help", "Show this help")

    key_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    key_table.add_column(style="bold yellow", no_wrap=True)
    key_table.add_column(style="white")
    key_table.add_row("provider", "API provider (openai, openrouter, anthropic, gemini, groq, deepseek, ...)")
    key_table.add_row("api_key", "API key (or use provider-specific env var)")
    key_table.add_row("base_url", "Custom API base URL (overrides provider default)")
    key_table.add_row("model", "Model name (default: gpt-5.4-mini)")
    key_table.add_row("reasoning_effort", "Reasoning effort value (empty to disable)")
    key_table.add_row("max_history", "Recent history items included as model context")
    key_table.add_row("max_stdin_chars", "Maximum piped stdin characters attached to one-shot prompts")
    key_table.add_row("max_tool_output_chars", "Maximum tool output characters returned to the model")
    key_table.add_row("command_timeout", "Seconds before a shell command is killed")
    key_table.add_row("command_safety", "Command confirmation level (all, risky, none)")
    key_table.add_row("audit", "LLM auditor for flagged commands (true/false)")
    key_table.add_row("auditor_auto_approve", "Let auditor auto-approve safe commands (true/false)")
    key_table.add_row("auditor_model", "Model for auditor (empty = active model)")
    key_table.add_row("system_prompt", "System prompt sent to the model")
    key_table.add_row("max_subagent_depth", "Max spawn depth for nested subagents")
    key_table.add_row("subagent_thread_pool_max_workers", "Parallel subagents per spawn call")
    key_table.add_row("check_updates", "Non-blocking background update check on one-shot runs (true/false)")

    paths_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    paths_table.add_column(style="dim", no_wrap=True)
    paths_table.add_column(style="dim")
    paths_table.add_row("Config", str(CONFIG_FILE))
    paths_table.add_row("Sessions index", str(SESSIONS_FILE))
    paths_table.add_row("Session data", str(SESSIONS_DIR))

    body = Group(
        section_rule("commands"),
        Text(""),
        cmd_table,
        Text(""),
        section_rule("config keys"),
        Text(""),
        key_table,
        Text(""),
        section_rule("paths"),
        Text(""),
        paths_table,
    )
    console.print(jarv_panel(body, title="help"))


def print_about() -> None:
    about = f"""# jarv

jarv is a command-line AI assistant that supports multiple AI providers including OpenAI, Anthropic, Google Gemini, OpenRouter, Groq, DeepSeek, and more.

## Basic usage

- `jarv` - Start heads-up mode so you can keep sending prompts without rerunning the command.
- `jarv <question>` - Ask jarv anything. Your words after `jarv` are sent as the user message.
- `command | jarv <instruction>` - Attach piped stdin as input for a one-shot prompt.
- `jarv /help` - Show the short command overview. (`jarv help` also works as a permanent alias.)
- `jarv /about` - Show this detailed overview.
- `jarv /config` - Show raw config values. The API key is masked.
- `jarv /set <key> <value>` - Set a config value. Values like `true`, `false`, integers, and floats are coerced.
- `jarv /unset <key>` - Reset a default config key, or remove a custom key.
- `jarv /history` - Show recent user and assistant messages.
- `jarv /usage` - Show token usage for the current session.
- `jarv /undo [n]` - Unsend the last n exchanges (default 1). The removed exchange is pushed onto a redo stack.
- `jarv /redo [n]` - Restore the last n undone exchanges (default 1). Sending a new message clears the redo stack.
- `jarv /settings` - Open an interactive settings menu for provider/model, command review, audit, runtime, and updates.
- `jarv /new` - Start a fresh session on the next message.
- `jarv /archive` - Archive this terminal's session history and start a fresh one on the next message.
- `jarv /sessions` / `jarv /session` - List sessions by recency. In an interactive terminal you can scroll through all of them; when stdout is not a TTY (e.g. piped), only the 5 most recent are listed.
- `jarv /sessions <id>` - Bind this terminal to a specific session id (prefix match).
- `jarv /update` - Check PyPI for the latest version and install it with pip.

## Heads-up mode

Run `jarv` with no prompt to start an interactive session. Type a prompt and press Enter to send it. Commands start with `/` (e.g. `/new`, `/history`). Type `exit`, `quit`, or `/exit`, or press Ctrl+C, to leave.

## How jarv works

1. Loads config from `{CONFIG_FILE}`.
2. Detects the current terminal and resolves its active session (default: one session per terminal).
3. Loads recent conversation history from that session's history file.
4. Sends your query, recent history, the configured system prompt, and system info to the configured provider's API.
5. Streams the assistant response in the terminal.
6. When the model issues tool calls, jarv runs the matching handler and feeds results back into the model (for `run_command`, that means showing the command, running it, printing stdout/stderr/exit status, and returning output up to `max_tool_output_chars`).
7. Saves the full session history. On future prompts, `max_history` limits only the recent history items sent back as model context.

## Tools and shell commands

- The root model sees three tools: `run_command`, `spawn`, and `read_artifact`.
- Spawned subagents also get a mandatory `finish` tool (to return output) and may get `spawn` when the parent sets `sterile: false`.
- Subagent internal transcripts are discarded. Root history stores the parent `spawn`/`read_artifact` tool calls and their returned outputs.
- Shell commands run only when the model calls `run_command`.
- On Windows, `run_command` uses PowerShell.
- On other platforms, `run_command` uses the system shell.
- Command output shown in the terminal is shortened after 30 lines, and tool output returned to the model is capped by `max_tool_output_chars`.
- Commands are killed after `command_timeout` seconds.
- Interrupted commands/process trees are terminated when possible.

## Config

Config file: `{CONFIG_FILE}`

Keys:

- `provider` - API provider. Options: openai, openrouter, anthropic, gemini, groq, deepseek, together, fireworks, ollama, lm_studio, vllm. Default: `openai`.
- `api_key` - API key. Can also be provided via provider-specific env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
- `base_url` - Custom API base URL. Overrides the provider's default endpoint.
- `model` - Model name. Default: `{DEFAULT_CONFIG['model']}`.
- `reasoning_effort` - Optional reasoning effort value. Empty disables this setting.
- `max_history` - Number of recent stored history items included as model context. It does not delete saved history. Stored items include user messages, assistant messages, reasoning items, function calls, and function call outputs. Default: `{DEFAULT_CONFIG['max_history']}`.
- `max_stdin_chars` - Maximum piped stdin characters attached to a one-shot prompt. Default: `{DEFAULT_CONFIG['max_stdin_chars']}`.
- `max_tool_output_chars` - Maximum tool output characters returned to the model. Default: `{DEFAULT_CONFIG['max_tool_output_chars']}`.
- `command_timeout` - Seconds before a shell command is killed. Default: `{DEFAULT_CONFIG['command_timeout']}`.
- `command_safety` - Command confirmation level. `all` = confirm every command, `risky` = confirm only dangerous commands (destructive ops, privilege escalation, network exfil, etc.), `none` = no confirmation. Default: `risky`.
- `audit` - When `true`, flagged commands are sent to a fast LLM auditor (uses extra tokens). The auditor's verdict appears inside the safety panel. Works with both `risky` and `all` safety levels. Default: `true`.
- `auditor_auto_approve` - When `true`, the auditor auto-approves commands it deems safe. When `false`, the auditor only shows a recommendation and the user always decides. Default: `true`.
- `auditor_model` - Model used for the auditor. Empty = use the active model. Default: empty.
- `system_prompt` - Instructions sent to the model before each request.
- `max_subagent_depth` - Maximum recursion depth for `spawn` (root is 0). Default: `{DEFAULT_CONFIG['max_subagent_depth']}`.
- `subagent_thread_pool_max_workers` - Max parallel children in one `spawn` batch. Default: `{DEFAULT_CONFIG['subagent_thread_pool_max_workers']}`.
- `check_updates` - When `true`, a one-shot `jarv <question>` run fires a non-blocking background check against GitHub. If a new version is found it is flagged locally and shown at the start of the next run. Default: `true`. Set to `false` to disable entirely. Heads-up mode (`jarv` with no args) and slash commands do not run this check.
- `/usage` model metadata comes from LiteLLM.

If the config file does not exist, jarv creates it and exits so you can add an API key.
If the config file is invalid JSON, jarv backs it up and creates a fresh default config.

## History and sessions

Session metadata file: `{SESSIONS_FILE}`

Each terminal is bound to exactly one session at a time. By default a fresh terminal gets its own session (id derived from terminal fingerprint). Per-session history and artifact sidecars live in `{SESSIONS_DIR}` as `history-<hash>.json` and `artifacts-<hash>.json`.

- `jarv /new` starts a fresh session by unmapping the current terminal. The next prompt creates a new session.
- `jarv /archive` archives the current session's history+artifacts and removes the terminal's mapping. The next prompt starts a fresh session.
- `jarv /sessions` / `jarv /session` lists sessions by recency (all in a TTY; 5 most recent when stdout is not a TTY).
- `jarv /sessions <id>` binds a specific session id (prefix match) to this terminal.

## Updates

- `jarv /update` checks PyPI for the latest version and installs it with pip.
- A one-shot `jarv <question>` (arguments on the command line, not heads-up mode) fires a fully non-blocking background check when `check_updates` is true. If an update is found it is saved locally; the next invocation shows the notification instantly with no network wait.
- The background check is throttled to at most once every {UPDATE_CHECK_INTERVAL_HOURS} hours.
- Set `check_updates` to `false` (`jarv /set check_updates false`) to disable the background check entirely.
- After updating, run `jarv` again to use the new version.

## Files

- Config directory: `{CONFIG_DIR}`
- Config file: `{CONFIG_FILE}`
- Session metadata file: `{SESSIONS_FILE}`
- Session history and artifacts: `{SESSIONS_DIR}`

## Version

jarv {__version__}
"""
    console.print(jarv_panel(Markdown(flatten_headings(about)), title="about", subtitle=f"v{__version__}"))


def _fetch_latest_pypi_version() -> str | None:
    try:
        req = urllib.request.Request(PYPI_VERSION_URL, headers={"User-Agent": "jarv-updater"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:
        return None


def _should_check_now() -> bool:
    """Return True if enough time has passed since the last update check."""
    import time
    if not LAST_CHECK_FILE.exists():
        return True
    try:
        last = float(LAST_CHECK_FILE.read_text().strip())
        return (time.time() - last) >= UPDATE_CHECK_INTERVAL_HOURS * 3600
    except Exception:
        return True


def _record_check_time() -> None:
    import time
    CONFIG_DIR.mkdir(exist_ok=True)
    LAST_CHECK_FILE.write_text(str(time.time()))


def _check_update_background() -> None:
    """Check PyPI for a newer version and write a flag file if one is available.

    Runs in a daemon thread — never blocks the main process. The flag is read
    (and cleared) on the *next* jarv invocation so there is zero network wait
    on the current run.
    """
    if not _should_check_now():
        return
    _record_check_time()
    latest = _fetch_latest_pypi_version()
    if not latest:
        return
    if latest != __version__:
        CONFIG_DIR.mkdir(exist_ok=True)
        UPDATE_FLAG_FILE.write_text(latest)


def maybe_print_update_available() -> None:
    """Show a pending update notification written by a previous run's background check."""
    if not UPDATE_FLAG_FILE.exists():
        return
    try:
        latest = UPDATE_FLAG_FILE.read_text().strip()
        UPDATE_FLAG_FILE.unlink(missing_ok=True)
        if latest and latest != __version__:
            console.print(f"[yellow]Update available![/yellow] [dim]v{__version__} → v{latest}[/dim]  Run [bold]jarv /update[/bold] to install.")
    except Exception:
        pass


def _is_pipx_env() -> bool:
    """Detect if jarv is running inside a pipx-managed virtualenv."""
    exe = Path(sys.executable).resolve()
    return "pipx" in exe.parts

def _run_pip_upgrade() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "jarv"],
        capture_output=True, text=True,
    )

def _is_externally_managed_error(result: subprocess.CompletedProcess) -> bool:
    return result.returncode != 0 and "externally-managed-environment" in (result.stderr or "")

def cmd_update() -> None:
    console.print("[dim]⟳ Checking for updates…[/dim]")
    latest = _fetch_latest_pypi_version()
    if latest is None:
        console.print("[bold red]✗[/bold red] [red]Could not reach PyPI.[/red]")
        return
    if latest == __version__:
        console.print(f"[bold green]✓[/bold green] [green]Already up to date.[/green] [dim](v{__version__})[/dim]")
        return
    console.print(f"[bold cyan]↓[/bold cyan] Update found [dim](v{__version__} → v{latest})[/dim]. Installing…")

    if _is_pipx_env():
        with console.status("[dim]Running pipx upgrade…[/dim]", spinner="dots"):
            result = subprocess.run(
                ["pipx", "upgrade", "jarv"],
                capture_output=True, text=True,
            )
    else:
        with console.status("[dim]Running pip install…[/dim]", spinner="dots"):
            result = _run_pip_upgrade()
        if _is_externally_managed_error(result):
            console.print("[dim]System Python detected — retrying with pipx…[/dim]")
            pipx_available = subprocess.run(
                ["pipx", "--version"], capture_output=True, text=True,
            ).returncode == 0
            if pipx_available:
                with console.status("[dim]Running pipx upgrade…[/dim]", spinner="dots"):
                    result = subprocess.run(
                        ["pipx", "upgrade", "jarv"],
                        capture_output=True, text=True,
                    )
            else:
                console.print("[bold red]✗[/bold red] [red]Update failed:[/red] pip is blocked by your system Python (PEP 668).")
                console.print("[dim]Install pipx and reinstall jarv with it:[/dim]")
                console.print("  [bold]brew install pipx && pipx install jarv[/bold]")
                return

    if result.returncode == 0:
        UPDATE_FLAG_FILE.unlink(missing_ok=True)
        console.print("[bold green]✓[/bold green] [green]Updated successfully.[/green] [dim]Run jarv again to use the new version.[/dim]")
    else:
        console.print("[bold red]✗[/bold red] [red]Update failed:[/red]")
        output = "\n".join(filter(None, [result.stdout.strip(), result.stderr.strip()]))
        if output:
            console.print(output, style="dim")


def cmd_new() -> None:
    session_context = prepare_session_context()
    history = load_history(session_context.history_file)
    if not history:
        console.print("[dim]○ Already on a new session.[/dim]")
        return
    forget_current_session()
    console.print("[bold green]✓[/bold green] [green]New session starts on your next message.[/green]")


def cmd_config() -> None:
    config = load_config()
    table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for k, v in config.items():
        if k == "api_key" and v:
            val = Text("***", style="dim")
        elif isinstance(v, bool):
            val = Text(repr(v), style="bold magenta")
        elif isinstance(v, (int, float)):
            val = Text(repr(v), style="bold yellow")
        elif isinstance(v, str):
            val = Text(repr(v), style="green")
        else:
            val = Text(repr(v))
        table.add_row(k, val)

    body = Group(
        section_rule("settings"),
        Text(""),
        table,
    )
    console.print(jarv_panel(body, title="config", subtitle=str(CONFIG_FILE)))


