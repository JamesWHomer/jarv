import os
import subprocess
import sys
from pathlib import Path

from rich.console import Group
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import CONFIG_DIR, CONFIG_FILE, DEFAULT_CONFIG, load_config, save_config
from .display import console, flatten_headings, status_line
from .history import (
    SESSIONS_DIR,
    SESSIONS_FILE,
    forget_current_session,
    load_history,
    prepare_session_context,
)
from .read_only_display import show_read_only_command
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
from .update_check import (
    UPDATE_CHECK_INTERVAL_HOURS,
    UPDATE_FLAG_FILE,
    _fetch_latest_pypi_release,
)
from .usage_command import cmd_usage


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


def _help_body() -> Group:
    def literal_command(value: str) -> Text:
        return Text(value, style="bold cyan", no_wrap=True)

    usage_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    usage_table.add_column(style="bold cyan", no_wrap=True)
    usage_table.add_column(style="white")
    usage_table.add_row("jarv", "Start heads-up mode for repeated prompts")
    usage_table.add_row("jarv <prompt>", "Ask once, then exit")
    usage_table.add_row("command | jarv <instruction>", "Attach piped stdin to a one-shot prompt")
    usage_table.add_row("git diff | jarv review this", "Review a patch from stdin")
    usage_table.add_row("jarv help", "Show this help")

    flag_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    flag_table.add_column(style="bold yellow", no_wrap=True)
    flag_table.add_column(style="white")
    flag_table.add_row("-m, --model MODEL", "Override the model for this run")
    flag_table.add_row("-e, --effort EFFORT", "Override reasoning effort")
    flag_table.add_row("--timeout SECONDS", "Override shell command timeout")
    flag_table.add_row("-s, --system PROMPT", "Override the system prompt")
    flag_table.add_row("--new", "Start this run with a fresh session")
    flag_table.add_row("--incognito", "Do not load or save session history")
    flag_table.add_row("--version", "Print the version and exit")

    cmd_table = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
    cmd_table.add_column(style="dim", no_wrap=True)
    cmd_table.add_column(style="bold cyan", no_wrap=True)
    cmd_table.add_column(style="white")
    cmd_table.add_row("chat", literal_command("/new"), "Start a fresh session on the next message")
    cmd_table.add_row("chat", literal_command("/history"), "Show recent conversation history")
    cmd_table.add_row("chat", literal_command("/undo [n]"), "Unsend the last n exchanges")
    cmd_table.add_row("chat", literal_command("/redo [n]"), "Restore undone exchanges")
    cmd_table.add_row("sessions", literal_command("/sessions, /session"), "List sessions")
    cmd_table.add_row("sessions", literal_command("/sessions <id>"), "Load a session by id prefix")
    cmd_table.add_row("sessions", literal_command("/archive"), "Archive this session and start fresh")
    cmd_table.add_row("settings", literal_command("/settings"), "Open common controls")
    cmd_table.add_row("settings", literal_command("/config"), "Show raw config values")
    cmd_table.add_row("settings", literal_command("/set <key> <value>"), "Set a raw config value")
    cmd_table.add_row("settings", literal_command("/unset <key>"), "Reset or remove a raw config value")
    cmd_table.add_row("settings", literal_command("/setup [provider|key|model|safety|base_url]"), "Run setup or jump to a step")
    cmd_table.add_row("usage", literal_command("/usage [day|week|month|--all [--since 24h]]"), "Show token usage")
    cmd_table.add_row("updates", literal_command("/update"), "Update jarv")
    cmd_table.add_row("info", literal_command("/help"), "Show this help")
    cmd_table.add_row("info", literal_command("/about"), "Show detailed reference info")
    cmd_table.add_row("heads-up", literal_command("exit, quit, /exit, /quit"), "Leave heads-up mode")

    more = Text.assemble(
        ("Use ", "white"),
        ("/settings", "bold cyan"),
        (" for common controls and ", "white"),
        ("/config", "bold cyan"),
        (" for raw values. Use ", "white"),
        ("/about", "bold cyan"),
        (" for detailed reference info, internals, and longer examples. README/GitHub: ", "white"),
        ("https://github.com/JamesWHomer/jarv", "bold cyan"),
    )

    return Group(
        usage_table,
        Text(""),
        flag_table,
        Text(""),
        cmd_table,
        Text(""),
        more,
    )


def print_help(*, mode: str | None = None, include_setup_nudge: bool = True) -> None:
    show_read_only_command(
        _help_body(),
        title="help",
        mode=mode,
        include_setup_nudge=include_setup_nudge,
    )


def _about_body() -> Markdown:
    about = f"""jarv is a command-line AI assistant that supports multiple AI providers including OpenAI, Anthropic, Google Gemini, OpenRouter, Groq, DeepSeek, and more.

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
- `jarv /usage day|week|month` - Show system-wide usage for the last 24h, 7d, or 30d.
- `jarv /usage --all [--since 24h]` - Show system-wide usage across Jarv sessions.
- `jarv /undo [n]` - Unsend the last n exchanges (default 1). The removed exchange is pushed onto a redo stack.
- `jarv /redo [n]` - Restore the last n undone exchanges (default 1). Sending a new message clears the redo stack.
- `jarv /settings` - Open an interactive settings menu for provider/model, command review, audit, runtime, and updates.
- `jarv /settings` also controls how read-only commands display: `auto`, `print`, `inline`, or `fullscreen`.
- `jarv /new` - Start a fresh session on the next message.
- `jarv /archive` - Archive this terminal's session history and start a fresh one on the next message.
- `jarv /sessions` / `jarv /session` - List sessions by recency. In an interactive terminal you can scroll through all of them; when stdout is not a TTY (e.g. piped), only the 5 most recent are listed.
- `jarv /sessions <id>` - Bind this terminal to a specific session id (prefix match).
- `jarv /update` - Check PyPI for the latest version and install it with pip.

## Heads-up mode

Run `jarv` with no prompt to start an interactive session. Type a prompt and press Enter to send it. Commands start with `/` (e.g. `/new`, `/history`). During a response, Ctrl+C stops further work, checkpoints the turn in history/context, and restores its prompt. Use `/undo` to remove that turn. At the prompt, Ctrl+C clears text and exits when the prompt is already empty. Type `exit`, `quit`, or `/exit` to leave directly.

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
- `read_only_command_display` - How `/help`, `/about`, `/usage`, and `/config` are displayed in an interactive terminal. `auto` chooses inline for short output and fullscreen for longer output. `print` preserves permanent terminal output. `inline` and `fullscreen` force those temporary views. Default: `auto`.
- `print_usage_after_agent` - When `true`, print a compact token usage line after each completed agent run. Default: `false`.
- `/usage` model metadata and cost estimates come from LiteLLM. System-wide views read future usage from `{CONFIG_DIR / "usage.json"}`.

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
    return Markdown(flatten_headings(about))


def print_about(*, mode: str | None = None, include_setup_nudge: bool = True) -> None:
    show_read_only_command(
        _about_body(),
        title="about",
        subtitle=f"v{__version__}",
        mode=mode,
        include_setup_nudge=include_setup_nudge,
    )



def _is_pipx_env() -> bool:
    """Detect if jarv is running inside a pipx-managed virtualenv."""
    return any(part.lower() == "pipx" for part in Path(sys.executable).parts)

def _run_pip_upgrade(package_spec: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", package_spec],
        capture_output=True, text=True,
    )

def _is_externally_managed_error(result: subprocess.CompletedProcess) -> bool:
    return result.returncode != 0 and "externally-managed-environment" in (result.stderr or "")

def _installed_version() -> str | None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.metadata as m; print(m.version('jarv'))",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None

def _pipx_installed_version() -> str | None:
    result = subprocess.run(
        ["pipx", "runpip", "jarv", "show", "jarv"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            return line.partition(":")[2].strip() or None
    return None

def cmd_update() -> None:
    console.print("[dim]⟳ Checking for updates…[/dim]")
    release = _fetch_latest_pypi_release()
    if release is None:
        console.print("[bold red]✗[/bold red] [red]Could not reach PyPI.[/red]")
        return
    latest, artifact_url = release
    if latest == __version__:
        console.print(f"[bold green]✓[/bold green] [green]Already up to date.[/green] [dim](v{__version__})[/dim]")
        return
    console.print(f"[bold cyan]↓[/bold cyan] Update found [dim](v{__version__} → v{latest})[/dim]. Installing…")

    install_target = "pipx environment" if _is_pipx_env() else "active environment"
    with console.status(f"[dim]Installing release into {install_target}…[/dim]", spinner="dots"):
        result = _run_pip_upgrade(artifact_url)
    installed_version = _installed_version
    if _is_externally_managed_error(result):
        console.print("[dim]System Python detected — retrying with pipx…[/dim]")
        pipx_available = subprocess.run(
            ["pipx", "--version"], capture_output=True, text=True,
        ).returncode == 0
        if pipx_available:
            with console.status("[dim]Installing release with pipx…[/dim]", spinner="dots"):
                result = subprocess.run(
                    ["pipx", "install", "--force", artifact_url],
                    capture_output=True, text=True,
                )
            installed_version = _pipx_installed_version
        else:
            console.print("[bold red]✗[/bold red] [red]Update failed:[/red] pip is blocked by your system Python (PEP 668).")
            console.print("[dim]Install pipx and reinstall jarv with it:[/dim]")
            console.print("  [bold]brew install pipx && pipx install jarv[/bold]")
            return

    installed = installed_version() if result.returncode == 0 else None
    if result.returncode == 0:
        if installed == latest:
            UPDATE_FLAG_FILE.unlink(missing_ok=True)
            console.print("[bold green]✓[/bold green] [green]Updated successfully.[/green] [dim]Run jarv again to use the new version.[/dim]")
            return

    console.print("[bold red]✗[/bold red] [red]Update failed:[/red]")
    output = "\n".join(filter(None, [result.stdout.strip(), result.stderr.strip()]))
    if result.returncode == 0:
        output = f"Expected jarv {latest}, but the active installation is {installed or 'unknown'}."
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

    body = Group(table)
    show_read_only_command(body, title="config", subtitle=str(CONFIG_FILE), config=config)


