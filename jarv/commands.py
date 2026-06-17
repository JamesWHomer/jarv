import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Group
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from . import __version__
from .config import CONFIG_DIR, CONFIG_FILE, DEFAULT_CONFIG, load_config, save_config, validate_config
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
    _is_newer_version,
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


def _mask_config_value(key: str, value) -> str:
    if key == "api_key" and value:
        return "[dim]***[/dim]"
    if key == "api_keys" and isinstance(value, dict) and value:
        masked = {k: ("***" if v else v) for k, v in value.items()}
        return f"[green]{repr(masked)}[/green]"
    return f"[green]{repr(value)}[/green]"


def cmd_set(args: list) -> None:
    if len(args) < 2:
        console.print(status_line("✗", "jarv /set <key> <value>", prefix_style="bold red", message_style="dim"))
        console.print(f"  [dim]Keys: {', '.join(DEFAULT_CONFIG.keys())}[/dim]")
        return
    key, raw = args[0], " ".join(args[1:])
    if key not in DEFAULT_CONFIG:
        console.print(
            f"[yellow]⚠[/yellow] [yellow]Unknown config key[/yellow] [bold]{key}[/bold] "
            f"[dim](known: {', '.join(DEFAULT_CONFIG.keys())})[/dim]"
        )
    config = load_config()
    value = coerce_value(raw)
    trial = dict(config)
    trial[key] = value
    if key in ("model", "provider"):
        from .reasoning import reconcile_reasoning_effort

        reconcile_reasoning_effort(trial)
    if not validate_config(trial):
        return
    save_config(trial)
    display = _mask_config_value(key, value)
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
        trial = dict(config)
        trial[key] = DEFAULT_CONFIG[key]
        if key in ("model", "provider"):
            from .reasoning import reconcile_reasoning_effort

            reconcile_reasoning_effort(trial)
        if not validate_config(trial):
            return
        save_config(trial)
        console.print(f"[bold cyan]↺[/bold cyan] [bold cyan]{key}[/bold cyan] [dim]reset to default →[/dim] [green]{repr(DEFAULT_CONFIG[key])}[/green]")
    else:
        trial = dict(config)
        del trial[key]
        if not validate_config(trial):
            return
        save_config(trial)
        console.print(f"[bold cyan]✓[/bold cyan] [bold cyan]{key}[/bold cyan] [dim]removed.[/dim]")


def _help_body() -> Group:
    help_table = Table(box=None, show_header=True, padding=(0, 2), pad_edge=False, width=91)
    help_table.add_column("COMMAND / FLAG", header_style="bold", no_wrap=True, width=37)
    help_table.add_column("DESCRIPTION", header_style="bold", style="white", width=52)

    divider = Text("\u2500" * 36, style="dim")
    description_divider = Text("\u2500" * 50, style="dim")
    help_table.add_row(divider, description_divider)

    groups = [
        [
            ("jarv", "Start heads-up mode", "bold cyan"),
            ("jarv <prompt>", "Ask once, then exit", "bold cyan"),
            ("command | jarv <instruction>", "Attach piped input to a one-shot prompt", "bold cyan"),
            ("git diff | jarv review this", "Review a patch from stdin", "bold cyan"),
        ],
        [
            ("--provider <provider>", "Override the provider", "bold yellow"),
            ("-m, --model <model>", "Override the model", "bold yellow"),
            ("-e, --effort <effort>", "Override reasoning effort", "bold yellow"),
            ("--timeout <seconds>", "Override shell command timeout", "bold yellow"),
            ("-s, --system <prompt>", "Override the system prompt", "bold yellow"),
            ("--new", "Start with a fresh session", "bold yellow"),
            ("--incognito", "Do not load or save session history", "bold yellow"),
            ("--version", "Print the version and exit", "bold yellow"),
        ],
        [
            ("/new", "Start a fresh session", "bold cyan"),
            ("/history", "Show recent conversation history", "bold cyan"),
            ("/undo [n]", "Unsend the last n exchanges", "bold cyan"),
            ("/redo [n]", "Restore undone exchanges", "bold cyan"),
            ("/sessions", "List sessions", "bold cyan"),
            ("/sessions <id>", "Load a session by ID prefix", "bold cyan"),
            ("/archive", "Archive this session and start fresh", "bold cyan"),
        ],
        [
            ("/settings", "Open common controls", "bold cyan"),
            ("/config", "Show raw configuration values", "bold cyan"),
            ("/set <key> <value>", "Set a configuration value", "bold cyan"),
            ("/unset <key>", "Reset or remove a configuration value", "bold cyan"),
            ("/setup [step]", "Run setup or jump to a step", "bold cyan"),
        ],
        [
            ("/usage [period]", "Show token usage", "bold cyan"),
            ("/update", "Update jarv", "bold cyan"),
            ("/help", "Show this help", "bold cyan"),
            ("/about", "Show detailed reference information", "bold cyan"),
            ("exit, quit, /exit, /quit", "Leave heads-up mode", "bold cyan"),
        ],
    ]
    for group_index, rows in enumerate(groups):
        if group_index:
            help_table.add_row("", "")
        for command, description, style in rows:
            help_table.add_row(Text(command, style=style, no_wrap=True), description)

    footer = Text.assemble(
        ("Common controls: ", "white"),
        ("/settings", "bold cyan"),
        ("    Raw configuration: ", "white"),
        ("/config", "bold cyan"),
        ("    Full reference: ", "white"),
        ("/about", "bold cyan"),
    )

    return Group(help_table, Text(""), footer)


def print_help(*, mode: str | None = None, include_setup_nudge: bool = True) -> None:
    show_read_only_command(
        _help_body(),
        title="help",
        mode=mode,
        include_setup_nudge=include_setup_nudge,
        max_width=95,
        close_hint="q / Esc / Enter  Close",
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
- `jarv /settings` also controls how read-only commands display: `fullscreen` or `print`.
- `jarv /new` - Start a fresh session on the next message.
- `jarv /archive` - Archive this terminal's session history and start a fresh one on the next message.
- `jarv /sessions` / `jarv /session` - List sessions by recency. In an interactive terminal you can scroll through all of them; when stdout is not a TTY (e.g. piped), only the 5 most recent are listed.
- `jarv /sessions <id>` - Bind this terminal to a specific session id (prefix match).
- `jarv /update` - Check PyPI for the latest version and update the active pip, pipx, or uv installation.

## Heads-up mode

Run `jarv` with no prompt to start an interactive session. Type a prompt and press Enter to send it. Commands start with `/` (e.g. `/new`, `/history`). During a response, Ctrl+C stops further work, checkpoints the turn in history/context, and restores its prompt. Use `/undo` to remove that turn. At the prompt, Ctrl+C clears text and exits when the prompt is already empty. Type `exit`, `quit`, or `/exit` to leave directly.

## How jarv works

1. Loads config from `{CONFIG_FILE}`.
2. Detects the current terminal and resolves its active session (default: one session per terminal).
3. Loads recent conversation history from that session's history file.
4. Sends your query, recent history, the configured system prompt, and system info to the configured provider backend (OpenAI Responses, Anthropic Messages, Gemini, or an OpenAI-compatible API).
5. Streams the assistant response in the terminal.
6. When the model issues tool calls, jarv runs the matching handler and feeds results back into the model (for `run_command`, that means showing the command, running it, printing stdout/stderr/exit status, and returning the requested output head and tail).
7. Saves the full session history. On future prompts, `max_history` limits only the recent history items sent back as model context.

## Tools and shell commands

- The root model can see five tools: `run_command`, `web_search`, `read`, `spawn`, and `ask_user`. The enabled subset is controlled from `/settings`.
- `read(input, offset, size)` pages through retained command output, visible artifacts, HTTP(S) URLs, local text files, and embedded-text PDFs using Unicode character offsets. Consecutive reads run concurrently.
- Direct local and HTTP(S) image reads (`png`, `jpeg`, `webp`, plus provider-supported `gif`) are returned as native image input when the active model advertises image capability in Jarv's cached provider/OpenRouter catalog. Image reads ignore `offset` and `size`, are capped at 10 MiB, and fall back to a text "no image capability" result when the selected model route is text-only or unknown.
- `web_search` supports any positive result count and a non-negative result offset. URL reads preserve HTTP(S) links as absolute URLs.
- Spawned subagents also get a mandatory `finish` tool (to return output) and may get `spawn` when the parent sets `sterile: false`.
- Subagent internal transcripts are discarded. Root history stores the parent `spawn`/`read` tool calls and their returned outputs. Artifact longform content persists per session in `artifacts-<hash>.json`.
- Shell commands run only when the model calls `run_command`.
- On Windows, `run_command` uses PowerShell.
- On other platforms, `run_command` uses the system shell.
- Command output shown in the terminal uses at most one-third of the screen height, biased roughly 2:1 toward the first lines, with the omitted middle count displayed. The UI also shows the resolved `head_chars` and `tail_chars` returned to the model. Truncated model output is retained under a session-scoped ID for later `read` calls.
- Commands are killed after `command_timeout` seconds.
- Web requests are killed after `web_timeout` seconds. Text and PDF responses are limited to 2 MiB.
- Interrupted commands/process trees are terminated when possible.

## Config

Config file: `{CONFIG_FILE}`

Keys:

- `provider` - API provider. Options: openai, openrouter, anthropic, gemini, groq, deepseek, together, fireworks, ollama, lm_studio, vllm. Default: `openai`.
- `api_key` - API key. Can also be provided via provider-specific env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
- `base_url` - Custom API base URL. Overrides the provider's default endpoint.
- `model` - Model name. Default: `{DEFAULT_CONFIG['model']}`.
- `service_tiers` - Per-provider processing tier. Values are `standard`, `flex`, or `priority`; missing providers use `standard`.
- `reasoning_effort` - Model-supported reasoning effort. Empty uses the provider/model default; `none` explicitly disables reasoning only where supported.
- `max_history` - Maximum stored history items included as model context (item cap before token trimming). It does not delete saved history. Stored items include user messages, assistant messages, reasoning items, function calls, and function call outputs. Default: `{DEFAULT_CONFIG['max_history']}`.
- `context_budget_ratio` - Share of the context window used for input. Default: `{DEFAULT_CONFIG['context_budget_ratio']}`.
- `context_compaction_threshold` - Fill ratio that triggers history compaction. Default: `{DEFAULT_CONFIG['context_compaction_threshold']}`.
- `context_output_reserve_ratio` - Context window share reserved for model output. Default: `{DEFAULT_CONFIG['context_output_reserve_ratio']}`.
- `context_window_fallback` - Context window when model metadata is unknown. Default: `{DEFAULT_CONFIG['context_window_fallback']}`.
- `max_stdin_chars` - Maximum piped stdin characters attached to a one-shot prompt. Default: `{DEFAULT_CONFIG['max_stdin_chars']}`.
- `max_tool_output_chars` - Maximum generic tool output characters returned to the model and the default combined head/tail budget for `run_command`. Default: `{DEFAULT_CONFIG['max_tool_output_chars']}`.
- `disabled_tools` - Tool names omitted from root agents and subagents. Use `/settings` to toggle `run_command`, `web_search`, `read`, `spawn`, and `ask_user`. Default: `[]`.
- `command_timeout` - Seconds before a shell command is killed. Default: `{DEFAULT_CONFIG['command_timeout']}`.
- `web_timeout` - Seconds before a web search or URL read is killed. Default: `{DEFAULT_CONFIG['web_timeout']}`.
- `command_safety` - Command confirmation level. `all` = confirm every command, `risky` = confirm only dangerous commands (destructive ops, privilege escalation, network exfil, etc.), `none` = no confirmation. Default: `risky`.
- `audit` - When `true`, flagged commands are sent to a fast LLM auditor (uses extra tokens). The auditor's verdict appears inside the safety panel. Works with both `risky` and `all` safety levels. Default: `true`.
- `auditor_auto_approve` - When `true`, the auditor auto-approves commands it deems safe. When `false`, the auditor only shows a recommendation and the user always decides. Default: `true`.
- `auditor_model` - Model used for the auditor. Empty = use the active model. Default: empty.
- `system_prompt` - Instructions sent to the model before each request.
- `max_subagent_depth` - Maximum recursion depth for `spawn` (root is 0). Default: `{DEFAULT_CONFIG['max_subagent_depth']}`.
- `subagent_thread_pool_max_workers` - Max parallel children in one `spawn` batch. Default: `{DEFAULT_CONFIG['subagent_thread_pool_max_workers']}`.
- `check_updates` - When `true`, a one-shot `jarv <question>` run fires a non-blocking background check against PyPI. If a new version is found it is flagged locally and shown at the start of the next run. Default: `true`. Set to `false` to disable entirely. Heads-up mode (`jarv` with no args) and slash commands do not run this check.
- `read_only_command_display` - How `/help`, `/about`, `/usage`, and `/config` are displayed in an interactive terminal. `fullscreen` uses a temporary alternate-screen view, compact when content fits and scrollable when it does not. `print` preserves permanent terminal output. Default: `fullscreen`.
- `tool_call_display` - How agent tool calls are rendered. `auto` uses resize-safe `print` mode for one-shot runs and bordered `fullscreen` cards in heads-up mode. Explicit `print` and `fullscreen` modes override that choice. Default: `auto`.
- `print_usage_after_agent` - When `true`, print a compact token usage line after each completed agent run. Default: `false`.
- `/usage` stores request-level provider and processing-tier cost provenance. Provider-reported cost is preferred; otherwise OpenRouter catalog pricing is marked estimated, while unknown and contract-priced requests remain explicit. System-wide views read future usage from `{CONFIG_DIR / "usage.json"}`.

If the config file does not exist, jarv creates it and exits so you can add an API key.
If the config file is invalid JSON, jarv backs it up and creates a fresh default config.

## History and sessions

Session metadata file: `{SESSIONS_FILE}`

Each terminal is bound to exactly one session at a time. By default a fresh terminal gets its own session (id derived from terminal fingerprint). Per-session history, artifact, and retained-output sidecars live in `{SESSIONS_DIR}` as `history-<hash>.json`, `artifacts-<hash>.json`, and `reads-<hash>.json`.

- `jarv /new` starts a fresh session by unmapping the current terminal. The next prompt creates a new session.
- `jarv /archive` archives the current session's history and sidecars and removes the terminal's mapping. The next prompt starts a fresh session.
- `jarv /sessions` / `jarv /session` lists sessions by recency (all in a TTY; 5 most recent when stdout is not a TTY).
- `jarv /sessions <id>` binds a specific session id (prefix match) to this terminal.

## Updates

- `jarv /update` checks PyPI for the latest version and updates the active pip, pipx, or uv installation. Editable source installs are left untouched.
- A one-shot `jarv <question>` (arguments on the command line, not heads-up mode) fires a fully non-blocking background check when `check_updates` is true. If an update is found it is saved locally; the next invocation shows the notification instantly with no network wait.
- The background check is throttled to at most once every {UPDATE_CHECK_INTERVAL_HOURS} hours.
- Set `check_updates` to `false` (`jarv /set check_updates false`) to disable the background check entirely.
- After updating, run `jarv` again to use the new version.

## Files

- Config directory: `{CONFIG_DIR}`
- Config file: `{CONFIG_FILE}`
- Session metadata file: `{SESSIONS_FILE}`
- Session history, artifacts, and retained command outputs: `{SESSIONS_DIR}`

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


def _is_uv_tool_env() -> bool:
    """Detect the standard uv tool environment path."""
    parts = [part.lower() for part in Path(sys.executable).parts]
    return any(parts[index:index + 2] == ["uv", "tools"] for index in range(len(parts) - 1))


def _installation_manager() -> str:
    if _is_pipx_env():
        return "pipx"
    if _is_uv_tool_env():
        return "uv"
    return "pip"


def _is_editable_install() -> bool:
    try:
        direct_url = importlib.metadata.distribution("jarv").read_text("direct_url.json")
        if not direct_url:
            return False
        metadata = json.loads(direct_url)
        return bool(metadata.get("dir_info", {}).get("editable"))
    except Exception:
        return False


UPDATE_INSTALL_TIMEOUT_SECONDS = 180


def _run_update_command(command: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=UPDATE_INSTALL_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(command, 127, "", f"Command not found: {command[0]}")
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        message = f"Update command timed out after {UPDATE_INSTALL_TIMEOUT_SECONDS} seconds."
        return subprocess.CompletedProcess(command, 124, stdout, "\n".join(filter(None, [stderr, message])))


def _run_update_install(manager: str, package_spec: str) -> subprocess.CompletedProcess:
    if manager == "pipx":
        return _run_update_command(["pipx", "install", "--force", package_spec])
    if manager == "uv":
        return _run_update_command(["uv", "tool", "install", "--force", package_spec])
    return _run_update_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", package_spec]
    )

def _is_externally_managed_error(result: subprocess.CompletedProcess) -> bool:
    output = "\n".join(filter(None, [result.stdout or "", result.stderr or ""])).lower()
    return result.returncode != 0 and "externally-managed-environment" in output


def _installed_version() -> str | None:
    result = _run_update_command(
        [
            sys.executable,
            "-c",
            "import importlib.metadata as m; print(m.version('jarv'))",
        ]
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _tool_installed_version(manager: str) -> str | None:
    if manager == "pipx":
        result = _run_update_command(["pipx", "runpip", "jarv", "show", "jarv"])
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    return line.partition(":")[2].strip() or None
    elif manager == "uv":
        result = _run_update_command(["uv", "tool", "list"])
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) >= 2 and fields[0].lower() == "jarv":
                    return fields[1].removeprefix("v") or None
    return _installed_version()


def _fallback_tool_manager() -> str | None:
    if shutil.which("uv"):
        return "uv"
    if shutil.which("pipx"):
        return "pipx"
    return None


def cmd_update() -> int:
    console.print("[dim]⟳ Checking for updates…[/dim]")
    release = _fetch_latest_pypi_release()
    if release is None:
        console.print("[bold red]✗[/bold red] [red]Could not reach PyPI.[/red]")
        return 1
    latest, package_spec = release
    if not _is_newer_version(latest, __version__):
        detail = f"v{__version__}"
        if latest != __version__:
            detail += f", PyPI v{latest}"
        console.print(f"[bold green]✓[/bold green] [green]Already up to date.[/green] [dim]({detail})[/dim]")
        return 0
    if _is_editable_install():
        console.print("[bold yellow]⚠[/bold yellow] [yellow]Editable install detected; automatic update skipped.[/yellow]")
        console.print("[dim]Update the source checkout, then reinstall it with your development workflow.[/dim]")
        return 1
    console.print(f"[bold cyan]↓[/bold cyan] Update found [dim](v{__version__} → v{latest})[/dim]. Installing…")

    manager = _installation_manager()
    install_target = {
        "pip": "active Python environment",
        "pipx": "pipx tool environment",
        "uv": "uv tool environment",
    }[manager]
    with console.status(f"[dim]Installing release into {install_target}…[/dim]", spinner="dots"):
        result = _run_update_install(manager, package_spec)
    if manager == "pip" and _is_externally_managed_error(result):
        fallback = _fallback_tool_manager()
        if fallback:
            console.print(f"[dim]System Python detected — retrying with {fallback}…[/dim]")
            with console.status(f"[dim]Installing release with {fallback}…[/dim]", spinner="dots"):
                result = _run_update_install(fallback, package_spec)
            manager = fallback
        else:
            console.print("[bold red]✗[/bold red] [red]Update failed:[/red] pip is blocked by your system Python (PEP 668).")
            console.print("[dim]Install uv or pipx, then reinstall Jarv as an isolated tool.[/dim]")
            console.print(f"  [bold]uv tool install {package_spec}[/bold]")
            return 1

    installed = (
        _installed_version() if manager == "pip" else _tool_installed_version(manager)
    ) if result.returncode == 0 else None
    if result.returncode == 0:
        if installed == latest:
            UPDATE_FLAG_FILE.unlink(missing_ok=True)
            console.print("[bold green]✓[/bold green] [green]Updated successfully.[/green] [dim]Run jarv again to use the new version.[/dim]")
            return 0

    console.print("[bold red]✗[/bold red] [red]Update failed:[/red]")
    output = "\n".join(
        filter(None, [(result.stdout or "").strip(), (result.stderr or "").strip()])
    )
    if result.returncode == 0:
        output = (
            f"Expected jarv {latest}, but this process still sees {installed or 'an unknown version'}. "
            f"The {manager} installation may not own the jarv command on PATH."
        )
    if output:
        console.print(output, style="dim")
    return 1


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
        elif k == "api_keys" and isinstance(v, dict) and v:
            masked = {pk: ("***" if pv else pv) for pk, pv in v.items()}
            val = Text(repr(masked), style="green")
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


