import argparse
import importlib
import os
import sys
import threading

from . import __version__
from .config import load_config, validate_config

STDIN_LABEL = "Input from stdin"

_CONFIG_MUTATING_COMMANDS = frozenset({"/set", "/unset", "/settings", "/setup"})

_COMMAND_TAKES_REST = {
    "setup": True,
    "help": False,
    "about": False,
    "update": False,
    "new": False,
    "archive": False,
    "session": True,
    "sessions": True,
    "history": False,
    "usage": True,
    "set": True,
    "unset": True,
    "config": False,
    "settings": False,
    "undo": True,
    "redo": True,
}


def _console():
    from .display import console

    return console


def _start_agent_import() -> tuple[dict, threading.Event]:
    state: dict = {}
    ready = threading.Event()

    def load() -> None:
        try:
            state["module"] = importlib.import_module("jarv.agent")
        except BaseException as exc:
            state["error"] = exc
        finally:
            ready.set()

    threading.Thread(target=load, daemon=True, name="jarv-agent-import").start()
    return state, ready


def _setup_nudge() -> None:
    """Print a one-line nudge if the env key is missing."""
    from .config import is_setup_complete
    if not is_setup_complete():
        _console().print("[dim]Tip: run [bold cyan]jarv /setup[/bold cyan] to configure your API key and get started.[/dim]\n")


def _lazy_commands():
    from .commands import (
        cmd_archive,
        cmd_new,
        cmd_config,
        cmd_history,
        cmd_redo,
        cmd_sessions,
        cmd_settings,
        cmd_set,
        cmd_undo,
        cmd_unset,
        cmd_update,
        cmd_usage,
        print_about,
        print_help,
    )
    return {
        "/setup":    (cmd_setup, False, True),
        "/help":     (print_help, False, False),
        "/about":    (print_about, False, False),
        "/update":   (cmd_update, True, False),
        "/new":      (cmd_new, True, False),
        "/archive":  (cmd_archive, True, False),
        "/session":  (cmd_sessions, True, True),
        "/sessions": (cmd_sessions, True, True),
        "/history":  (cmd_history, True, False),
        "/usage":    (cmd_usage, False, True),
        "/set":      (cmd_set, True, True),
        "/unset":    (cmd_unset, True, True),
        "/config":   (cmd_config, False, False),
        "/settings": (cmd_settings, True, False),
        "/undo":     (cmd_undo, True, True),
        "/redo":     (cmd_redo, True, True),
    }


def _run_slash_command(command: str, rest: list[str]) -> bool:
    """Run a slash command. Returns True if handled, False if unknown."""
    dispatch = _lazy_commands()
    entry = dispatch.get(command)
    if entry is None:
        return False

    handler, needs_nudge, takes_rest = entry
    if needs_nudge:
        _setup_nudge()
    if takes_rest:
        handler(rest)
    else:
        handler()
    return True


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply one-run CLI flag overrides on top of a loaded config."""
    config = dict(config)
    if args.provider:
        config["provider"] = args.provider
    if args.model:
        config["model"] = args.model
    if args.effort:
        config["reasoning_effort"] = args.effort
    elif args.provider or args.model:
        from .reasoning import reconcile_reasoning_effort

        reconcile_reasoning_effort(config)
    if args.timeout is not None:
        config["command_timeout"] = args.timeout
    if args.system:
        config["system_prompt"] = args.system
    return config


def _client_needs_refresh(old: dict, new: dict) -> bool:
    keys = ("provider", "base_url", "api_key", "api_keys")
    return any(old.get(key) != new.get(key) for key in keys)


def _reload_heads_up_runtime(
    config: dict,
    client,
    args: argparse.Namespace,
) -> tuple[dict, object]:
    """Reload config from disk and recreate the API client when needed."""
    from .provider import create_client

    refreshed = _apply_cli_overrides(load_config(), args)
    if not validate_config(refreshed):
        return config, client
    if _client_needs_refresh(config, refreshed) or client is None:
        client = create_client(refreshed)
    return refreshed, client


def _handle_heads_up_slash_command(
    command: str,
    rest: list[str],
    *,
    config: dict,
    client,
    args: argparse.Namespace | None,
    unknown_help_hint: bool = False,
) -> tuple[dict, object]:
    """Run a slash command and reload heads-up runtime after config changes."""
    handled = _run_slash_command(command, rest)
    if not handled:
        console = _console()
        console.print(f"[red]Unknown command:[/red] {command}")
        if unknown_help_hint:
            console.print("[dim]Run [bold]/help[/bold] for a list of commands.[/dim]")
        return config, client
    if args is not None and command in _CONFIG_MUTATING_COMMANDS:
        return _reload_heads_up_runtime(config, client, args)
    return config, client


def _maybe_command(first_word: str, rest: list[str]) -> tuple[bool, str, list[str]] | None:
    """Check if the first word looks like a slash command without the slash.

    Only prompts when usage matches the command signature: commands that don't
    take arguments only match when used alone, commands that take arguments
    match regardless.

    Returns (is_command, command, rest) if the user confirms it's a command,
    None if they want to treat it as a regular message.
    """
    name = first_word.lower()
    takes_rest = _COMMAND_TAKES_REST.get(name)
    if takes_rest is None:
        return None

    if not takes_rest and rest:
        return None

    full_input = " ".join([first_word] + rest) if rest else first_word
    console = _console()
    console.print(f"\n[yellow]Did you mean the command [bold]/{name}[/bold] or a message to jarv?[/yellow]")
    console.print(f"  [bold]1.[/bold] Run command [cyan]/{name}[/cyan]")
    console.print(f"  [bold]2.[/bold] Send as message: [dim]{full_input}[/dim]")

    try:
        choice = console.input("[bold yellow]>[/bold yellow] ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if choice == "1":
        return (True, f"/{name}", rest)
    return None


def cmd_setup(rest: list[str] | None = None) -> dict | None:
    """Run the interactive setup wizard. Returns config or None."""
    from .setup import run_setup_wizard, SETUP_STEPS
    console = _console()
    step = None
    if rest:
        step = rest[0].lower().lstrip("-")
        if step not in SETUP_STEPS:
            console.print(f"[red]Unknown setup step '{step}'.[/red]")
            console.print(f"[dim]Available: {', '.join(sorted(SETUP_STEPS))}[/dim]")
            return None
    try:
        return run_setup_wizard(step=step)
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Setup cancelled.[/dim]")
        return None


def _build_parser() -> argparse.ArgumentParser:
    from .provider_catalog import PROVIDERS

    parser = argparse.ArgumentParser(
        prog="jarv",
        description="AI-powered CLI agent",
        add_help=True,
    )
    parser.add_argument("query", nargs="*", help="Prompt to run (omit for heads-up mode)")
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        type=str.lower,
        metavar="PROVIDER",
        help="Override provider for this run",
    )
    parser.add_argument("-m", "--model", metavar="MODEL", help="Override model for this run (e.g. gpt-4o)")
    parser.add_argument(
        "-e",
        "--effort",
        metavar="EFFORT",
        help="Override model-supported reasoning effort (none/minimal/low/medium/high/xhigh/max)",
    )
    parser.add_argument("--timeout", type=int, metavar="SECONDS", help="Override command timeout in seconds")
    parser.add_argument("-s", "--system", metavar="PROMPT", help="Override system prompt for this run")
    parser.add_argument("--new", action="store_true", help="Start a fresh session (ignore prior history, but still save)")
    parser.add_argument("--incognito", action="store_true", help="Don't load or save session history")
    parser.add_argument("--version", action="version", version=f"jarv {__version__}")
    return parser


def _stdin_is_piped(stdin=None) -> bool:
    stdin = stdin or sys.stdin
    isatty = getattr(stdin, "isatty", None)
    return callable(isatty) and not isatty()


def _read_piped_stdin(max_chars: int, stdin=None) -> tuple[str, bool]:
    stdin = stdin or sys.stdin
    try:
        limit = int(max_chars)
        if limit <= 0:
            limit = 200000
    except (TypeError, ValueError):
        limit = 200000

    from .unicode_safety import sanitize_text

    text = sanitize_text(stdin.read(limit + 1))
    if "\x00" in text:
        raise ValueError("stdin appears to contain binary data; pass text input instead.")
    if len(text) > limit:
        return text[:limit], True
    return text, False


def _compose_query(query_parts: list[str], stdin_text: str = "", stdin_truncated: bool = False) -> str:
    query = " ".join(query_parts).strip()
    if not stdin_text:
        return query

    stdin_body = stdin_text.rstrip("\n")
    if query:
        suffix = "\n\n[stdin truncated because it exceeded max_stdin_chars]" if stdin_truncated else ""
        return f"{query}\n\n{STDIN_LABEL}:\n```text\n{stdin_body}{suffix}\n```"

    suffix = "\n\n[stdin truncated because it exceeded max_stdin_chars]" if stdin_truncated else ""
    return f"{stdin_body}{suffix}".strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = _build_parser()
    args = parser.parse_args()
    query_parts: list[str] = args.query
    console = _console()

    # "jarv help" permanent alias (only when help is the sole argument)
    if len(query_parts) == 1 and query_parts[0].lower() == "help":
        from .commands import print_help
        print_help(mode="print", include_setup_nudge=False)
        return

    # Slash commands — flags are silently ignored for these
    if query_parts and query_parts[0].startswith("/"):
        command = query_parts[0].lower()
        if command == "/update":
            from .commands import cmd_update

            status = cmd_update()
            if status:
                raise SystemExit(status)
            return
        if not _run_slash_command(command, query_parts[1:]):
            console.print(f"[red]Unknown command:[/red] {command}")
            console.print("[dim]Run [bold]jarv /help[/bold] for a list of commands.[/dim]")
        return

    # Check if user typed a command name without the slash (e.g. "jarv set" instead of "jarv /set")
    if query_parts and not query_parts[0].startswith("/") and not _stdin_is_piped():
        result = _maybe_command(query_parts[0], query_parts[1:])
        if result is not None:
            _, command, rest = result
            if not _run_slash_command(command, rest):
                console.print(f"[red]Unknown command:[/red] {command}")
            return

    # First-run: auto-trigger setup wizard if no config exists yet
    from .config import is_setup_complete

    config = dict(load_config())
    if not args.provider and not is_setup_complete(config):
        result = cmd_setup()
        if result is None or not is_setup_complete(result):
            sys.exit(1)
        config = result

    config = _apply_cli_overrides(config, args)

    if not validate_config(config):
        sys.exit(1)

    from .provider import resolve_api_key, LOCAL_PROVIDERS

    provider_name = config.get("provider", "openai")
    api_key = resolve_api_key(config)
    if not api_key and provider_name not in LOCAL_PROVIDERS:
        console.print("[red]No API key found.[/red] Run [bold cyan]jarv /setup[/bold cyan] to get started.")
        sys.exit(1)

    stdin_text = ""
    stdin_truncated = False
    if _stdin_is_piped():
        try:
            stdin_text, stdin_truncated = _read_piped_stdin(config.get("max_stdin_chars", 200000))
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)

    query = _compose_query(query_parts, stdin_text, stdin_truncated)

    if not query:
        from .provider import create_client

        agent_loader = _start_agent_import()
        client = create_client(config)
        run_heads_up_mode(config, client, args=args, agent_loader=agent_loader)
        return

    if config.get("check_updates", True):
        from .update_check import _check_update_background, maybe_print_update_available

        maybe_print_update_available()
        threading.Thread(target=_check_update_background, daemon=True).start()

    from .agent import run_agent
    try:
        result = run_agent(query, config, client=None, new_session=args.new, incognito=args.incognito)
        if getattr(result, "cancelled", False) is True:
            console.print("\n[dim]Cancelled.[/dim]")
            sys.exit(130)
        if isinstance(getattr(result, "error", None), str):
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(130)


def run_heads_up_mode(
    config: dict,
    client,
    *,
    args: argparse.Namespace | None = None,
    agent_loader: tuple[dict, threading.Event] | None = None,
) -> None:
    console = _console()
    console.print("[bold cyan]jarv heads-up mode[/bold cyan]")
    console.print("[dim]Type a prompt and press Enter. Use /help for commands. Ctrl+C clears; press it again to leave.[/dim]")
    agent_import, agent_ready = agent_loader or _start_agent_import()
    prefill = ""
    while True:
        try:
            from .command_input import read_editable_line

            console.print()
            query = read_editable_line(
                "\x1b[1;36mjarv>\x1b[0m ",
                initial=prefill,
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            return
        prefill = ""

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            console.print("[dim]Goodbye.[/dim]")
            return

        if query.startswith("/"):
            parts = query.split()
            command = parts[0].lower()
            if command in {"/exit", "/quit"}:
                console.print("[dim]Goodbye.[/dim]")
                return
            config, client = _handle_heads_up_slash_command(
                command,
                parts[1:],
                config=config,
                client=client,
                args=args,
                unknown_help_hint=True,
            )
            continue

        # Check if user typed a command name without the slash
        parts = query.split()
        result = _maybe_command(parts[0], parts[1:])
        if result is not None:
            _, command, rest = result
            config, client = _handle_heads_up_slash_command(
                command,
                rest,
                config=config,
                client=client,
                args=args,
            )
            continue

        try:
            agent_ready.wait()
            if "error" in agent_import:
                raise agent_import["error"]
            result = agent_import["module"].run_agent(query, config, client)
            if getattr(result, "cancelled", False) is True:
                console.print("\n[dim]Cancelled.[/dim]")
                prefill = result.prompt or query
            elif isinstance(getattr(result, "error", None), str):
                continue
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/dim]")
            prefill = query


if __name__ == "__main__":
    main()
