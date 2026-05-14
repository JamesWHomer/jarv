import argparse
import os
import sys
import threading

from . import __version__
from .config import CONFIG_FILE, load_config, validate_config
from .display import console


SLASH_COMMANDS = {"/help", "/about", "/update", "/clear", "/archive", "/session", "/sessions", "/history", "/usage", "/set", "/unset", "/config", "/undo", "/redo"}


def _run_slash_command(command: str, rest: list[str]) -> bool:
    """Run a slash command. Returns True if handled, False if unknown."""
    from .commands import (
        _check_update_background,
        cmd_archive,
        cmd_clear,
        cmd_config,
        cmd_history,
        cmd_redo,
        cmd_sessions,
        cmd_set,
        cmd_undo,
        cmd_unset,
        cmd_update,
        cmd_usage,
        print_about,
        print_help,
    )
    if command == "/help":
        print_help()
    elif command == "/about":
        print_about()
    elif command == "/update":
        cmd_update()
    elif command == "/clear":
        cmd_clear()
    elif command == "/archive":
        cmd_archive()
    elif command in {"/session", "/sessions"}:
        cmd_sessions(rest)
    elif command == "/history":
        cmd_history()
    elif command == "/usage":
        cmd_usage()
    elif command == "/set":
        cmd_set(rest)
    elif command == "/unset":
        cmd_unset(rest)
    elif command == "/config":
        cmd_config()
    elif command == "/undo":
        cmd_undo(rest)
    elif command == "/redo":
        cmd_redo(rest)
    else:
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarv",
        description="OpenAI-powered CLI agent",
        add_help=True,
    )
    parser.add_argument("query", nargs="*", help="Prompt to run (omit for heads-up mode)")
    parser.add_argument("-m", "--model", metavar="MODEL", help="Override model for this run (e.g. gpt-4o)")
    parser.add_argument("-e", "--effort", metavar="EFFORT", help="Override reasoning effort (low/medium/high)")
    parser.add_argument("--timeout", type=int, metavar="SECONDS", help="Override command timeout in seconds")
    parser.add_argument("-s", "--system", metavar="PROMPT", help="Override system prompt for this run")
    parser.add_argument("--no-history", action="store_true", help="Don't load or save session history")
    parser.add_argument("--version", action="version", version=f"jarv {__version__}")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = _build_parser()
    args = parser.parse_args()
    query_parts: list[str] = args.query

    # "jarv help" permanent alias
    if query_parts and query_parts[0].lower() == "help":
        from .commands import print_help
        print_help()
        return

    # Slash commands — flags are silently ignored for these
    if query_parts and query_parts[0].startswith("/"):
        command = query_parts[0].lower()
        if not _run_slash_command(command, query_parts[1:]):
            console.print(f"[red]Unknown command:[/red] {command}")
            console.print("[dim]Run [bold]jarv /help[/bold] for a list of commands.[/dim]")
        return

    config = load_config()
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

    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print(f"[red]No API key found.[/red] Edit {CONFIG_FILE} or set OPENAI_API_KEY.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    if not query_parts:
        run_heads_up_mode(config, client)
        return

    if config.get("check_updates", True):
        from .commands import _check_update_background, maybe_print_update_available
        maybe_print_update_available()
        threading.Thread(target=_check_update_background, daemon=True).start()

    from .agent import run_agent
    query = " ".join(query_parts)
    try:
        run_agent(query, config, client, no_history=args.no_history)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)


def run_heads_up_mode(config: dict, client) -> None:
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

        try:
            from .agent import run_agent
            run_agent(query, config, client, propagate_keyboard_interrupt=True)
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye.[/dim]")
            return


if __name__ == "__main__":
    main()
