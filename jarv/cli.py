import os
import sys
import threading

from openai import OpenAI

from .agent import run_agent
from .commands import (
    _check_update_background,
    cmd_clear,
    cmd_config,
    cmd_history,
    cmd_set,
    cmd_unset,
    cmd_update,
    maybe_print_update_available,
    print_about,
    print_help,
)
from .config import CONFIG_FILE, load_config, validate_config
from .display import console
from .history import independent_session


def main() -> None:
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

    if command == "help":
        print_help()
        return

    if command == "about":
        print_about()
        return

    if command == "session":
        config = load_config()
        if not validate_config(config):
            sys.exit(1)
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            console.print(f"[red]No API key found.[/red] Edit {CONFIG_FILE} or set OPENAI_API_KEY.")
            sys.exit(1)
        session = independent_session()
        console.print(f"[dim]Independent session: {session[1]}[/dim]")
        client = OpenAI(api_key=api_key)
        run_heads_up_mode(config, client, session_override=session, independent=True)
        return

    if command == "update":
        cmd_update()
        return

    if command == "clear":
        cmd_clear()
        return

    if command == "history":
        cmd_history()
        return

    if command == "set":
        cmd_set(args[1:])
        return

    if command == "unset":
        cmd_unset(args[1:])
        return

    if command == "config":
        cmd_config()
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


def run_heads_up_mode(
    config: dict,
    client: OpenAI,
    session_override: tuple[str, str, str] | None = None,
    independent: bool = False,
) -> None:
    title = "jarv independent session" if independent else "jarv heads-up mode"
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print("[dim]Type a prompt and press Enter. Type 'exit' or press Ctrl+C to leave.[/dim]")
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

        try:
            run_agent(
                query,
                config,
                client,
                session_override=session_override,
                independent=independent,
                propagate_keyboard_interrupt=True,
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye.[/dim]")
            return


if __name__ == "__main__":
    main()
