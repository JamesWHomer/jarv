from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .config import CONFIG_DIR, CONFIG_FILE
from .display import console, status_line

UPDATE_CHECK_INTERVAL_HOURS = 24
UPDATE_FLAG_FILE = CONFIG_DIR / "update_available.txt"

def _fetch_latest_pypi_release() -> tuple[str, str] | None:
    from .update_check import _fetch_latest_pypi_release as fetch

    return fetch()


def _is_newer_version(candidate: str, current: str) -> bool:
    from .update_check import _is_newer_version as is_newer

    return is_newer(candidate, current)


def show_read_only_command(*args, **kwargs):
    from .read_only_display import show_read_only_command as show

    return show(*args, **kwargs)


def load_config() -> dict:
    from .config import load_config as load

    return load()


def cmd_btw(args: list | None = None) -> None:
    # /btw is an in-session affordance: it asks an aside, then folds it off the
    # main thread. Outside the interactive heads-up loop there is no live thread
    # to return to, so point the user at where it works.
    console.print(
        "[dim]/btw works inside the interactive session — run [bold]jarv[/bold], "
        "then [bold]/btw <question>[/bold].[/dim]"
    )


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
    from .config import DEFAULT_CONFIG, load_config, save_config, validate_config

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
    from .config import DEFAULT_CONFIG, load_config, save_config, validate_config

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


def _command_help_rows(names: list[str]) -> list[tuple[str, str, str]]:
    """Build (label, summary, style) help rows from the command registry."""
    from .command_registry import COMMANDS

    rows: list[tuple[str, str, str]] = []
    for name in names:
        meta = COMMANDS[name]
        label = f"/{name} {meta.arg_hint}".rstrip()
        rows.append((label, meta.summary, "bold cyan"))
    return rows


def _help_body() -> Group:
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    help_table = Table(box=None, show_header=True, padding=(0, 2), pad_edge=False, width=91)
    help_table.add_column("COMMAND / FLAG", header_style="bold", no_wrap=True, width=37)
    help_table.add_column("DESCRIPTION", header_style="bold", style="white", width=52)

    divider = Text("\u2500" * 36, style="dim")
    description_divider = Text("\u2500" * 50, style="dim")
    help_table.add_row(divider, description_divider)

    groups = [
        [
            ("jarv", "Start heads-up mode", "bold cyan"),
            ("jarv <prompt>", "Ask once; can run interactive commands", "bold cyan"),
            ("command | jarv <instruction>", "Attach piped input to a one-shot prompt", "bold cyan"),
            ("git diff | jarv review this", "Review a patch from stdin", "bold cyan"),
        ],
        [
            ("--provider <provider>", "Override the provider", "bold yellow"),
            ("-m, --model <model>", "Override the model", "bold yellow"),
            ("-e, --effort <effort>", "Override reasoning effort", "bold yellow"),
            ("--timeout <seconds>", "Override command timeout/check-in", "bold yellow"),
            ("-s, --system <prompt>", "Override the system prompt", "bold yellow"),
            ("--new", "Start with a fresh session", "bold yellow"),
            ("--incognito", "Do not load or save session history", "bold yellow"),
            ("--version", "Print the version and exit", "bold yellow"),
        ],
        _command_help_rows(["new", "history", "tree", "btw", "undo", "redo", "sessions", "archive"]),
        _command_help_rows(["settings", "config", "set", "unset", "setup"]),
        _command_help_rows(["usage", "update", "uninstall", "help", "about"])
        + [("exit, quit, /exit, /quit", "Leave heads-up mode", "bold cyan")],
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
        close_hint="q / Esc / Enter  Close",
        fill_screen=False,
    )


def _about_body() -> Markdown:
    from rich.markdown import Markdown

    from .config import DEFAULT_CONFIG
    from .config_schema import config_about_lines
    from .display import flatten_headings
    from .history import SESSIONS_DIR, SESSIONS_FILE

    about = f"""jarv is a command-line AI assistant that supports multiple AI providers including OpenAI, Anthropic, Google Gemini, OpenRouter, Groq, DeepSeek, and more.

## Basic usage

- `jarv` - Start heads-up mode so you can keep sending prompts without rerunning the command.
- `jarv <question>` - Ask jarv anything. Your words after `jarv` are sent as the user message.
- `command | jarv <instruction>` - Attach piped stdin as input for a one-shot prompt.
- `jarv /help` - Show the short command overview. (`jarv help` also works as a permanent alias.)
- `jarv /about` - Show this detailed overview.
- `jarv /setup` - Run the setup wizard to choose a provider, enter an API key, and pick a model.
- `jarv /config` - Show raw config values. The API key is masked.
- `jarv /set <key> <value>` - Set a config value. Values like `true`, `false`, integers, and floats are coerced.
- `jarv /unset <key>` - Reset a default config key, or remove a custom key.
- `jarv /history` - Show recent user and assistant messages.
- `jarv /tree` - Browse the session as a tree; fork, edit, or resume from any earlier prompt.
- `jarv /usage` - Open the interactive usage screen. `←/→` (or `1-5` / `s t w m a`) switches scope live between Session, Today, Week, Month, and All.
- `jarv /usage <session|day|week|month|all>` - Open straight to a scope. `day`/`today` is a rolling 24h window; `all` reads the full system-wide history.
- `jarv /undo [n]` - Unsend the last n exchanges (default 1). The removed exchange is pushed onto a redo stack.
- `jarv /redo [n]` - Restore the last n undone exchanges (default 1). Sending a new message clears the redo stack.
- `jarv /btw <question>` - Ask an aside without derailing the main thread.
- `jarv /settings` - Open an interactive settings menu for provider/model, command review, audit, runtime, updates, and how read-only commands display (`fullscreen` or `print`).
- `jarv /new` - Start a fresh session on the next message.
- `jarv /archive` - Archive this terminal's session history and start a fresh one on the next message.
- `jarv /sessions` / `jarv /session` - List sessions by recency. In an interactive terminal you can scroll through all of them; when stdout is not a TTY (e.g. piped), only the 5 most recent are listed.
- `jarv /sessions <id>` - Bind this terminal to a specific session id (prefix match).
- `jarv /update` - Update Jarv through the active install channel. Standalone builds update from GitHub Releases; Python installs update through pip, pipx, or uv.
- `jarv /uninstall [--purge] [--yes]` - Uninstall Jarv or show the command for its package manager. User data is kept unless `--purge` is supplied; `--yes` skips the confirmation prompt (required when stdin is not a terminal).

## Heads-up mode

Run `jarv` with no prompt to start an interactive session. Type a prompt and press Enter to send it. Commands start with `/` (e.g. `/new`, `/history`). During a response, Esc or Ctrl+C stops further work, checkpoints the turn in history/context, and restores its prompt. Use `/undo` to remove that turn. At the prompt, Esc or Ctrl+C clears text and exits when the prompt is already empty. Type `exit`, `quit`, or `/exit` to leave directly.

## How jarv works

1. Loads config from `{CONFIG_FILE}`.
2. Detects the current terminal and resolves its active session (default: one session per terminal).
3. Loads recent conversation history from that session's history file.
4. Sends your query, recent history, the configured system prompt, and system info to the configured provider backend (OpenAI Responses, Anthropic Messages, Gemini, or an OpenAI-compatible API).
5. Streams the assistant response in the terminal.
6. When the model issues tool calls, jarv runs the matching handler and feeds results back into the model. See "Tools and shell commands" below for how `run_command` output, truncation, and interactive stdin are handled.
7. Saves the full session history. On future prompts, `max_history` limits only the recent history items sent back as model context.

## Tools and shell commands

- The root model can see six tools: `run_command`, `web_search`, `read`, `edit`, `spawn`, and `ask_user`. The enabled subset is controlled from `/settings`.
- `edit(path, old_text, new_text, replace_all)` makes an exact string replacement in an existing UTF-8 text file; `old_text` must match exactly once unless `replace_all` is true. Edits are gated by `command_safety` with a diff preview.
- `read(input, offset, size)` pages through retained command output, visible artifacts, HTTP(S) URLs, local text files, and embedded-text PDFs using Unicode character offsets. Consecutive reads run concurrently.
- Direct local and HTTP(S) image reads (`png`, `jpeg`, `webp`, plus provider-supported `gif`) are returned as native image input when the active model advertises image capability in Jarv's cached provider/OpenRouter catalog. Image reads ignore `offset` and `size`, are capped at 10 MiB, and fall back to a text "no image capability" result when the selected model route is text-only or unknown.
- `web_search` supports any positive result count and a non-negative result offset. URL reads preserve HTTP(S) links as absolute URLs.
- Spawned subagents also get a mandatory `finish` tool (to return output) and may get `spawn` when the parent sets `sterile: false`.
- Subagent internal transcripts are discarded. Root history stores the parent `spawn`/`read` tool calls and their returned outputs. Artifact longform content persists per session in `artifacts-<hash>.json`.
- Shell commands run only when the model calls `run_command`.
- On Windows, `run_command` uses PowerShell.
- On other platforms, `run_command` uses the system shell.
- Shell state (current directory, environment variables, activated venv) persists across `run_command` calls. Subagents inherit a snapshot of the parent's state at spawn; their changes do not propagate back. State is kept in memory only and resets when jarv exits. If a command is killed (timeout/interrupt) or ends with `exit` on Windows, the state from before that command is kept.
- If a command is still running and appears to be waiting for input, jarv asks the model for terminal input instead of printing the assistant response as chat. Plain text is sent with Enter; the interactive prompt also exposes temporary controls such as wait, interrupt, EOF, Enter, Tab, Escape, and arrow keys only while they are relevant.
- During the interactive loop, `command_timeout` is a check-in interval. If the command keeps running past it, jarv asks the model what to do next and includes elapsed/idle time instead of killing the process.
- Interactive command output is delta-only. Each terminal input/output step is displayed separately, and jarv sends only the new stdout/stderr since the previous interaction back to the model.
- Command output shown in the terminal uses at most one-third of the screen height, biased roughly 2:1 toward the first lines, with the omitted middle count displayed. The UI also shows the resolved `head_chars` and `tail_chars` returned to the model. Truncated model output is retained under a session-scoped ID for later `read` calls.
- Non-interactive command execution is killed after `command_timeout` seconds.
- Web requests are killed after `web_timeout` seconds. Text and PDF responses are limited to 2 MiB.
- Interrupted commands/process trees are terminated when possible.

## Config

Config file: `{CONFIG_FILE}`

Keys:

{chr(10).join(config_about_lines(DEFAULT_CONFIG))}
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
- `jarv /tree` opens an interactive tree of the session's prompts; from any node you can fork, edit, or resume.

## Updates

- `jarv /update` uses GitHub Releases for standalone binaries and PyPI for Python installs. Editable source installs are left untouched.
- A one-shot `jarv <question>` (arguments on the command line, not heads-up mode) fires a fully non-blocking background check when `check_updates` is true. Standalone installs check GitHub Releases; Python installs check PyPI. If an update is found it is saved locally; the next invocation shows the notification instantly with no network wait.
- The background check is throttled to at most once every {UPDATE_CHECK_INTERVAL_HOURS} hours.
- Set `check_updates` to `false` (`jarv /set check_updates false`) to disable the background check entirely.
- After updating, run `jarv` again to use the new version.

## Files

Everything lives under `{CONFIG_DIR}`: `config.json`, `sessions.json`, and a `sessions/` directory of per-session history, artifact, and retained-output files.

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


@dataclass(frozen=True)
class UpdateOutcome:
    """Result of an update attempt, independent of how it is displayed.

    ``kind`` is one of ``updated`` / ``staged`` / ``current`` / ``editable`` /
    ``failed``; ``message`` is the one-line plain-text summary and ``detail``
    an optional multi-line elaboration. Presenters (the CLI printer below, the
    heads-up transcript) own their styling and restart hints.
    """

    kind: str
    message: str
    detail: str = ""
    latest: str | None = None

    @property
    def ok(self) -> bool:
        return self.kind in ("updated", "staged", "current")


#: Reports a long-running update stage ("Checking for updates", "Installing
#: v1.2.3 into …"); each call supersedes the previous stage.
UpdateStage = Callable[[str], None]


def perform_update(stage: UpdateStage) -> UpdateOutcome:
    """Run the update through the active install channel, reporting stages.

    UI-agnostic: all long waits (network, installer subprocess) happen between
    ``stage`` calls, so callers can run this on a worker thread and render
    progress however suits them.
    """
    from .standalone import is_standalone_install

    if is_standalone_install():
        return _perform_standalone_update(stage)
    return _perform_python_update(stage)


def _perform_standalone_update(stage: UpdateStage) -> UpdateOutcome:
    from . import standalone

    stage("Checking for updates")
    manifest = standalone.fetch_release_manifest()
    if manifest is None:
        return UpdateOutcome("failed", "Could not reach GitHub Releases.")
    latest = str(manifest["version"])
    if not _is_newer_version(latest, __version__):
        return UpdateOutcome("current", "Already up to date.", detail=f"v{__version__}", latest=latest)
    asset = standalone.select_release_asset(manifest)
    if asset is None:
        return UpdateOutcome(
            "failed",
            "No standalone release asset matches this system.",
            detail=(
                f"Needed platform={standalone.normalize_platform()} "
                f"architecture={standalone.normalize_architecture()}."
            ),
            latest=latest,
        )

    stage(f"Downloading and verifying v{latest}")
    try:
        result = standalone.install_standalone_asset(asset)
    except Exception as exc:
        return UpdateOutcome("failed", "Update failed.", detail=str(exc), latest=latest)

    UPDATE_FLAG_FILE.unlink(missing_ok=True)
    if result == "staged":
        return UpdateOutcome("staged", "Update staged.", latest=latest)
    return UpdateOutcome("updated", "Updated successfully.", latest=latest)


def _perform_python_update(stage: UpdateStage) -> UpdateOutcome:
    stage("Checking for updates")
    release = _fetch_latest_pypi_release()
    if release is None:
        return UpdateOutcome("failed", "Could not reach PyPI.")
    latest, package_spec = release
    if not _is_newer_version(latest, __version__):
        detail = f"v{__version__}"
        if latest != __version__:
            detail += f", PyPI v{latest}"
        return UpdateOutcome("current", "Already up to date.", detail=detail, latest=latest)
    if _is_editable_install():
        return UpdateOutcome(
            "editable",
            "Editable install detected; automatic update skipped.",
            detail="Update the source checkout, then reinstall it with your development workflow.",
            latest=latest,
        )

    manager = _installation_manager()
    install_target = {
        "pip": "active Python environment",
        "pipx": "pipx tool environment",
        "uv": "uv tool environment",
    }[manager]
    stage(f"Installing v{latest} into {install_target}")
    result = _run_update_install(manager, package_spec)
    if manager == "pip" and _is_externally_managed_error(result):
        fallback = _fallback_tool_manager()
        if fallback is None:
            return UpdateOutcome(
                "failed",
                "Update failed: pip is blocked by your system Python (PEP 668).",
                detail=(
                    "Install uv or pipx, then reinstall Jarv as an isolated tool:\n"
                    f"  uv tool install {package_spec}"
                ),
                latest=latest,
            )
        stage(f"System Python detected — retrying with {fallback}")
        result = _run_update_install(fallback, package_spec)
        manager = fallback

    installed = (
        _installed_version() if manager == "pip" else _tool_installed_version(manager)
    ) if result.returncode == 0 else None
    if result.returncode == 0 and installed == latest:
        UPDATE_FLAG_FILE.unlink(missing_ok=True)
        return UpdateOutcome("updated", "Updated successfully.", latest=latest)

    if result.returncode == 0:
        detail = (
            f"Expected jarv {latest}, but this process still sees {installed or 'an unknown version'}. "
            f"The {manager} installation may not own the jarv command on PATH."
        )
    else:
        detail = "\n".join(
            filter(None, [(result.stdout or "").strip(), (result.stderr or "").strip()])
        )
    return UpdateOutcome("failed", "Update failed.", detail=detail, latest=latest)


@contextmanager
def _console_update_stages():
    """Yield an UpdateStage that renders each stage as a console spinner."""
    active: list = []

    def _stop() -> None:
        if active:
            active.pop().__exit__(None, None, None)

    def stage(text: str) -> None:
        _stop()
        status = console.status(f"[dim]{text}…[/dim]", spinner="dots")
        status.__enter__()
        active.append(status)

    try:
        yield stage
    finally:
        _stop()


_UPDATE_OUTCOME_STYLES = {
    "updated": ("✓", "green"),
    "staged": ("✓", "green"),
    "current": ("✓", "green"),
    "editable": ("⚠", "yellow"),
    "failed": ("✗", "red"),
}

_UPDATE_RESTART_HINTS = {
    "updated": "Run jarv again to use the new version.",
    "staged": "Exit Jarv, then run it again to finish the update.",
}


def _print_update_outcome(outcome: UpdateOutcome) -> int:
    icon, style = _UPDATE_OUTCOME_STYLES[outcome.kind]
    line = f"[bold {style}]{icon}[/bold {style}] [{style}]{outcome.message}[/{style}]"
    hint = _UPDATE_RESTART_HINTS.get(outcome.kind)
    if hint:
        line += f" [dim]{hint}[/dim]"
    elif outcome.kind == "current" and outcome.detail:
        line += f" [dim]({outcome.detail})[/dim]"
    console.print(line)
    if outcome.kind in ("editable", "failed") and outcome.detail:
        console.print(outcome.detail, style="dim")
    return 0 if outcome.ok else 1


def _cmd_update_standalone() -> int:
    with _console_update_stages() as stage:
        outcome = _perform_standalone_update(stage)
    return _print_update_outcome(outcome)


def cmd_update() -> int:
    from .standalone import is_standalone_install

    if is_standalone_install():
        return _cmd_update_standalone()
    with _console_update_stages() as stage:
        outcome = _perform_python_update(stage)
    return _print_update_outcome(outcome)


def cmd_new() -> None:
    from .history import forget_current_session, load_history, prepare_session_context

    session_context = prepare_session_context()
    history = load_history(session_context.history_file)
    if not history:
        console.print("[dim]○ Already on a new session.[/dim]")
        return
    forget_current_session()
    console.print("[bold green]✓[/bold green] [green]New session starts on your next message.[/green]")


def cmd_config() -> None:
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

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
    show_read_only_command(
        body,
        title="config",
        subtitle=str(CONFIG_FILE),
        config=config,
        fill_screen=True,
    )


