import argparse
import os
import sys
import threading

from . import __version__
from .config import load_config, validate_config

STDIN_LABEL = "Input from stdin"

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
    parser = argparse.ArgumentParser(
        prog="jarv",
        description="AI-powered CLI agent",
        add_help=True,
    )
    parser.add_argument("query", nargs="*", help="Prompt to run (omit for heads-up mode)")
    parser.add_argument("-m", "--model", metavar="MODEL", help="Override model for this run (e.g. gpt-4o)")
    parser.add_argument("-e", "--effort", metavar="EFFORT", help="Override reasoning effort (low/medium/high)")
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

    # "jarv help" permanent alias
    if query_parts and query_parts[0].lower() == "help":
        from .commands import print_help
        print_help(mode="print", include_setup_nudge=False)
        return

    # Slash commands — flags are silently ignored for these
    if query_parts and query_parts[0].startswith("/"):
        command = query_parts[0].lower()
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

    config = load_config()
    if not is_setup_complete(config):
        result = cmd_setup()
        if result is None or not is_setup_complete(result):
            sys.exit(1)
        config = result
    if not validate_config(config):
        sys.exit(1)

    # Apply flag overrides on top of config
    if args.model:
        config["model"] = args.model
    if args.effort:
        config["reasoning_effort"] = args.effort
    if args.timeout is not None:
        config["command_timeout"] = args.timeout
    if args.system:
        config["system_prompt"] = args.system

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

        client = create_client(config)
        run_heads_up_mode(config, client)
        return

    if config.get("check_updates", True):
        from .update_check import _check_update_background, maybe_print_update_available

        maybe_print_update_available()
        threading.Thread(target=_check_update_background, daemon=True).start()

    from .agent import run_agent
    try:
        run_agent(query, config, client=None, new_session=args.new, incognito=args.incognito)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)


def run_heads_up_mode(config: dict, client) -> None:
    console = _console()
    console.print("[bold cyan]jarv heads-up mode[/bold cyan]")
    console.print("[dim]Type a prompt and press Enter. Use /help for commands. Press Ctrl+C to leave.[/dim]")
    while True:
        try:
            query = console.input("\n[bold cyan]jarv>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            return

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
            if not _run_slash_command(command, parts[1:]):
                console.print(f"[red]Unknown command:[/red] {command}")
                console.print("[dim]Run [bold]/help[/bold] for a list of commands.[/dim]")
            continue

        # Check if user typed a command name without the slash
        parts = query.split()
        result = _maybe_command(parts[0], parts[1:])
        if result is not None:
            _, command, rest = result
            if not _run_slash_command(command, rest):
                console.print(f"[red]Unknown command:[/red] {command}")
            continue

        try:
            from .agent import run_agent
            run_agent(query, config, client, propagate_keyboard_interrupt=True)
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye.[/dim]")
            return


if __name__ == "__main__":
    main()
