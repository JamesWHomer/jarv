import os
import sys
import threading

from openai import OpenAI

from .agent import run_agent
from .commands import (
    _check_update_background,
    cmd_cleanup,
    cmd_clear,
    cmd_config,
    cmd_history,
    cmd_load,
    cmd_set,
    cmd_unset,
    cmd_update,
    maybe_print_update_available,
    print_about,
    print_help,
)
from .config import CONFIG_FILE, load_config, validate_config
from .display import console


SLASH_COMMANDS = {"/help", "/about", "/update", "/cleanup", "/clear", "/load", "/history", "/set", "/unset", "/config"}


def _run_slash_command(command: str, rest: list[str]) -> bool:
    """Run a slash command. Returns True if handled, False if unknown."""
    if command == "/help":
        print_help()
    elif command == "/about":
        print_about()
    elif command == "/update":
        cmd_update()
    elif command == "/cleanup":
        cmd_cleanup()
    elif command == "/clear":
        cmd_clear()
    elif command == "/load":
        cmd_load(rest)
    elif command == "/history":
        cmd_history()
    elif command == "/set":
        cmd_set(rest)
    elif command == "/unset":
        cmd_unset(rest)
    elif command == "/config":
        cmd_config()
    else:
        return False
    return True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        config = load_config()
        if not validate_config(config):
            sys.exit(1)
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            console.print(f"[red]No API key found.[/red] Edit {CONFIG_FILE} or set OPENAI_API_KEY.")
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        run_heads_up_mode(config, client)
        return

    args = sys.argv[1:]
    command = args[0].lower()

    # "jarv help" is a permanent alias regardless of slash convention
    if command == "help":
        print_help()
        return

    if command.startswith("/"):
        if not _run_slash_command(command, args[1:]):
            console.print(f"[red]Unknown command:[/red] {command}")
            console.print("[dim]Run [bold]jarv /help[/bold] for a list of commands.[/dim]")
        return

    query = " ".join(args)
    config = load_config()
    if not validate_config(config):
        sys.exit(1)

    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        console.print(f"[red]No API key found.[/red] Edit {CONFIG_FILE} or set OPENAI_API_KEY.")
        sys.exit(1)

    if config.get("check_updates", True):
        update_thread = threading.Thread(target=_check_update_background, daemon=True)
        update_thread.start()
        update_thread.join(timeout=0.2)
        maybe_print_update_available()

    client = OpenAI(api_key=api_key)
    try:
        run_agent(query, config, client)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)


def run_heads_up_mode(config: dict, client: OpenAI) -> None:
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
            run_agent(query, config, client, propagate_keyboard_interrupt=True)
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye.[/dim]")
            return


if __name__ == "__main__":
    main()
