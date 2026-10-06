from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    import threading

from . import __version__

STDIN_LABEL = "Input from stdin"


def load_config() -> dict:
    # --help and --version need neither storage nor the configuration schema.
    from .config import load_config as load

    return load()


def validate_config(config: dict) -> bool:
    from .config import validate_config as validate

    return validate(config)


def _console():
    from .display import console

    return console


def _print_previous_update_result() -> None:
    from rich.text import Text

    from .standalone import consume_windows_update_result

    result = consume_windows_update_result()
    if result is None:
        return
    console = _console()
    if result["status"] == "updated":
        message = result["message"] or "Updated successfully."
        console.print(Text.assemble(("✓ ", "bold green"), (message, "green")))
        return
    message = result["message"] or "The previous update failed."
    console.print(Text.assemble(("✗ ", "bold red"), (message, "red")))
    console.print("[dim]Run [bold]jarv /update[/bold] to retry.[/dim]")


def _print_previous_uninstall_result() -> None:
    from rich.text import Text

    from .paths import UNINSTALL_RESULT_FILE
    from .standalone import consume_windows_update_result

    result = consume_windows_update_result(UNINSTALL_RESULT_FILE)
    if result is None or result["status"] != "failed":
        return
    console = _console()
    message = result["message"] or "The previous uninstall did not complete."
    console.print(Text.assemble(("✗ ", "bold red"), (message, "red")))
    console.print("[dim]Run [bold]jarv /uninstall[/bold] to retry.[/dim]")


def _print_pending_results(*, quiet: bool = False) -> None:
    from .paths import CONFIG_DIR, UNINSTALL_RESULT_FILE

    if not quiet and (CONFIG_DIR / "update-result.json").is_file():
        _print_previous_update_result()
    if not quiet and UNINSTALL_RESULT_FILE.is_file():
        _print_previous_uninstall_result()


def _dispatch_command_query(query_parts: list[str]) -> None:
    if len(query_parts) == 1 and query_parts[0].lower() == "help":
        from .commands import print_help

        print_help(include_setup_nudge=False)
        return
    command = query_parts[0].lower()
    if not _run_slash_command(command, query_parts[1:], exit_on_error=True):
        console = _console()
        console.print(f"[red]Unknown command:[/red] {command}")
        _print_command_suggestions(command)
        console.print("[dim]Run [bold]jarv /help[/bold] for a list of commands.[/dim]")
        raise SystemExit(2)


def _command_entry(query_parts: list[str]) -> None:
    """The option-free command path has the same diagnostics as main()."""
    try:
        _console()
        _print_pending_results()
        _dispatch_command_query(query_parts)
    except (OSError, ValueError) as exc:
        print(f"jarv: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"jarv: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


def _setup_nudge() -> None:
    """Print a one-line nudge if the env key is missing."""
    from .config import is_setup_complete
    if not is_setup_complete():
        _console().print("[dim]Tip: run [bold cyan]jarv /setup[/bold cyan] to configure your API key and get started.[/dim]\n")


def _lazy_commands():
    from .command_registry import build_dispatch

    return build_dispatch()


def _run_slash_command(command: str, rest: list[str], *, exit_on_error: bool = False) -> bool:
    """Run a slash command. Returns True if handled, False if unknown."""
    dispatch = _lazy_commands()
    entry = dispatch.get(command)
    if entry is None:
        return False

    handler, needs_nudge, takes_rest = entry
    if needs_nudge:
        _setup_nudge()
    if rest and not takes_rest:
        _console().print(f"[red]{command} does not accept arguments.[/red]")
        if exit_on_error:
            raise SystemExit(2)
        return True
    outcome = handler(rest) if takes_rest else handler()
    status = outcome if type(outcome) is int else 0
    if status and exit_on_error:
        raise SystemExit(status)
    return True


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply one-run CLI flag overrides on top of a loaded config."""
    import copy
    config = copy.deepcopy(config)
    overrides = dict(getattr(args, "config", None) or [])
    config.update(overrides)
    if args.provider:
        config["provider"] = args.provider
    from .provider_catalog import PROVIDERS
    if config.get("provider", "openai") not in PROVIDERS:
        raise ValueError(f"Unknown provider: {config['provider']!r}")
    if args.model:
        config["model"] = args.model
    if args.effort is not None:
        config["reasoning_effort"] = args.effort
    elif (args.provider or args.model or "provider" in overrides or "model" in overrides) and "reasoning_effort" not in overrides:
        from .reasoning import reconcile_reasoning_effort

        reconcile_reasoning_effort(config)
    if args.timeout is not None:
        config["command_timeout"] = args.timeout
    if args.system is not None:
        config["system_prompt"] = args.system
    for argument, key in (
        ("base_url", "base_url"), ("command_safety", "command_safety"),
        ("system_file_text", "system_prompt"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            config[key] = value
    for argument, key in (("no_project_context", "project_context"),
                          ("no_update_check", "check_updates"), ("no_color", "colour")):
        if getattr(args, argument, False):
            config[key] = False
    tier = getattr(args, "service_tier", None)
    if tier is not None:
        from .provider_catalog import service_tier_choices
        provider = config.get("provider", "openai")
        if tier not in service_tier_choices(provider, config.get("model"), base_url=config.get("base_url")):
            raise ValueError(
                f"Service tier {tier!r} is not supported by {provider}/{config.get('model')} "
                "at this endpoint. Ultrafast requires gpt-6-astra through OpenAI's direct Responses API."
            )
        config.setdefault("service_tiers", {})[provider] = tier
    elif "service_tiers" not in overrides and (
        args.provider or args.model or getattr(args, "base_url", None) is not None
        or {"provider", "model", "base_url"}.intersection(overrides)
    ):
        from .provider_catalog import reconcile_service_tier

        if reconcile_service_tier(config) is not None:
            print("Processing tier reset to standard for this model/endpoint.", file=sys.stderr)
    allowed = getattr(args, "tools", None)
    if allowed is not None or getattr(args, "no_tools", False):
        from .config_schema import TOOL_NAMES
        config["disabled_tools"] = [name for name in TOOL_NAMES if name not in (allowed or [])]
    for argument in ("non_interactive", "max_turns", "run_timeout"):
        value = getattr(args, argument, None)
        if value:
            config["_" + argument] = value
    if getattr(args, "quiet", False):
        config["_quiet"] = True
    return config


def _client_needs_refresh(old: dict, new: dict) -> bool:
    keys = ("provider", "base_url", "api_key", "api_keys")
    return any(old.get(key) != new.get(key) for key in keys)


def _reload_heads_up_runtime(
    config: dict,
    client,
    args: argparse.Namespace,
) -> tuple[dict, object]:
    """Reload config and return a transport owned by the heads-up app.

    The app retires replaced clients after the turn using them has finished.
    """
    refreshed = _apply_cli_overrides(load_config(), args)
    if not validate_config(refreshed):
        return config, client
    # An unopened heads-up session has no transport yet. Keep config-only
    # changes cheap; the first submitted prompt will use the refreshed config.
    if client is not None and _client_needs_refresh(config, refreshed):
        from .provider import create_client

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
    from .command_registry import CONFIG_MUTATING_COMMANDS

    handled = _run_slash_command(command, rest)
    if not handled:
        console = _console()
        console.print(f"[red]Unknown command:[/red] {command}")
        _print_command_suggestions(command)
        if unknown_help_hint:
            console.print("[dim]Run [bold]/help[/bold] for a list of commands.[/dim]")
        return config, client
    if args is not None and command in CONFIG_MUTATING_COMMANDS:
        return _reload_heads_up_runtime(config, client, args)
    return config, client


def _print_command_suggestions(command: str) -> None:
    from .command_registry import suggest_commands

    suggestions = suggest_commands(command)
    if suggestions:
        formatted = ", ".join(f"[bold]/{name}[/bold]" for name in suggestions)
        _console().print(f"[dim]Did you mean {formatted}?[/dim]")


def _maybe_command(first_word: str, rest: list[str]) -> tuple[bool, str, list[str]] | None:
    """Check if the first word looks like a slash command without the slash.

    Only prompts when usage matches the command signature: commands that don't
    take arguments only match when used alone, commands that take arguments
    match regardless.

    Returns (is_command, command, rest) if the user confirms it's a command,
    None if they want to treat it as a regular message.
    """
    from .command_registry import command_takes_rest

    name = first_word.lower()
    takes_rest = command_takes_rest(name)
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


def cmd_setup(rest: list[str] | None = None) -> dict | int | None:
    """Run setup, returning its config or a command failure/cancellation status."""
    from .setup import run_setup_wizard, SETUP_STEPS
    console = _console()
    step = None
    if rest:
        if len(rest) > 1:
            console.print("[red]Usage: /setup [step][/red]")
            return 2
        step = rest[0].lower().lstrip("-")
        if step not in SETUP_STEPS:
            console.print(f"[red]Unknown setup step '{step}'.[/red]")
            console.print(f"[dim]Available: {', '.join(sorted(SETUP_STEPS))}[/dim]")
            return 2
    try:
        result = run_setup_wizard(step=step)
        return result if result is not None else 130
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Setup cancelled.[/dim]")
        return 130


def _positive_int(raw: str) -> int:
    import argparse
    try:
        value = int(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("must be a positive integer")


def _config_override(raw: str) -> tuple[str, object]:
    import argparse
    import json
    from .config_schema import CONFIG_FIELD_BY_KEY, parse_config_value, validate_config_fields

    key, separator, value = raw.partition("=")
    field = CONFIG_FIELD_BY_KEY.get(key)
    if not separator or field is None:
        raise argparse.ArgumentTypeError("expected a known configuration KEY=VALUE")
    if field.validator in ("string_map", "service_tiers", "disabled_tools"):
        try:
            parsed = json.loads(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{key} requires JSON") from None
    else:
        parsed = parse_config_value(key, value)
    errors = []
    candidate = {key: parsed}
    if not validate_config_fields(candidate, report=errors.append):
        raise argparse.ArgumentTypeError("; ".join(errors).replace("[red]", "").replace("[/red]", ""))
    return key, candidate[key]


def _tool_allowlist(raw: str) -> list[str]:
    import argparse
    from .config_schema import TOOL_NAMES
    names = [name.strip() for name in raw.split(",")]
    if not names or any(name not in TOOL_NAMES for name in names):
        raise argparse.ArgumentTypeError(f"tools must be a comma-separated list of: {', '.join(TOOL_NAMES)}")
    return list(dict.fromkeys(names))


def _build_parser() -> argparse.ArgumentParser:
    import argparse
    from .provider_catalog import PROVIDERS

    parser = argparse.ArgumentParser(
        prog="jarv",
        description="AI-powered CLI agent",
        add_help=True,
        allow_abbrev=False,
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
    parser.add_argument("--timeout", type=_positive_int, metavar="SECONDS", help="Override command timeout/check-in seconds")
    system = parser.add_mutually_exclusive_group()
    system.add_argument("-s", "--system", metavar="PROMPT", help="Override system prompt for this run")
    system.add_argument("--system-file", metavar="PATH", help="Read system prompt from a UTF-8 file")
    session = parser.add_mutually_exclusive_group()
    session.add_argument("--new", action="store_true", help="Start a fresh session (ignore prior history, but still save)")
    session.add_argument("--incognito", action="store_true", help="Don't load or save session history")
    session.add_argument("--session", metavar="ID", help="Use or create a named session for this invocation")
    parser.add_argument("-c", "--config", type=_config_override, action="append", metavar="KEY=VALUE", help="Override a setting for this run (repeatable; lists/maps use JSON)")
    parser.add_argument("-C", "--cwd", metavar="PATH", help="Working directory for this invocation")
    parser.add_argument("--base-url", metavar="URL", help="Override the provider API endpoint")
    parser.add_argument("--service-tier", choices=("standard", "flex", "priority", "ultrafast"), help="Override processing tier (ultrafast: direct OpenAI Astra, 6x standard token rates)")
    parser.add_argument("--command-safety", choices=("all", "risky", "none"), help="Override command/edit approval policy")
    tool_group = parser.add_mutually_exclusive_group()
    tool_group.add_argument("--tools", type=_tool_allowlist, metavar="LIST", help="Allow only these comma-separated tools (including for subagents)")
    tool_group.add_argument("--no-tools", action="store_true", help="Disable all agent tools")
    parser.add_argument("--max-turns", type=_positive_int, metavar="N", help="Maximum agent model turns, shared with subagents")
    parser.add_argument("--run-timeout", type=_positive_int, metavar="SECONDS", help="Cancel an entire agent run after this many seconds")
    parser.add_argument("--non-interactive", action="store_true", help="Never prompt; fail when required user input is unavailable")
    parser.add_argument("--output-format", choices=("text", "json", "jsonl"), help="One-shot answer format; progress goes to stderr")
    parser.add_argument("--prompt-file", metavar="PATH", help="Read prompt from a UTF-8 file (instead of positional prompt)")
    parser.add_argument("--no-project-context", action="store_true", help="Skip project instructions and git context")
    parser.add_argument("--no-update-check", action="store_true", help="Skip background update checks")
    parser.add_argument("--no-color", action="store_true", help="Disable output colour")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-q", "--quiet", action="store_true", help="One-shot answer only; suppress progress, preserve errors")
    verbosity.add_argument("--verbose", action="store_true", help="Show one-shot runtime details and progress on stderr")
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

    # The exact standalone invocation has no parsing or configuration work.
    # Mixed arguments still go through argparse, preserving its ordering and
    # validation semantics (including -- and options with missing values).
    if sys.argv[1:] == ["--version"]:
        print(f"jarv {__version__}")
        raise SystemExit(0)

    query_parts = sys.argv[1:]
    if query_parts and (
        query_parts[0].startswith("/") or
        (len(query_parts) == 1 and query_parts[0].lower() == "help")
    ) and not any(value.startswith("-") for value in query_parts):
        # With no option-looking words argparse would only copy positionals.
        # Mixed options still take its full validation/diagnostic path.
        _command_entry(query_parts)
        return

    parser = _build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        if args.query and args.query[0].lower() in {"/uninstall", "uninstall"}:
            args.query.extend(unknown)
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    # Resolve CLI paths before changing cwd, as shells do for other arguments.
    if args.prompt_file and args.query:
        parser.error("--prompt-file cannot be combined with a positional prompt")
    if args.session is not None and (not args.session.strip() or len(args.session) > 128):
        parser.error("--session must contain 1–128 characters and cannot be blank")
    if args.query and (args.query[0].startswith("/") or (len(args.query) == 1 and args.query[0].lower() == "help")):
        _reject_command_flags(parser, args)

    output = None
    if args.output_format or args.quiet or args.verbose:
        from .cli_output import CliOutput
        output = CliOutput(args.output_format or "text", quiet=args.quiet, verbose=args.verbose)
    from contextlib import ExitStack
    from pathlib import Path

    with ExitStack() as stack:
        if output:
            stack.enter_context(output.route_diagnostics())
        try:
            for flag in ("prompt_file", "system_file"):
                path = getattr(args, flag)
                if path is not None:
                    value = Path(path).expanduser().read_text(encoding="utf-8-sig")
                    if "\x00" in value:
                        raise ValueError(f"--{flag.replace('_', '-')} must contain text, not binary data")
                    setattr(args, flag + "_text", value)
            if args.prompt_file is not None:
                args.query = [args.prompt_file_text]
            if args.cwd:
                old_cwd = os.getcwd()
                from .shell import get_session_shell_state
                shell_state = get_session_shell_state()
                previous_cwd = shell_state.cwd
                os.chdir(Path(args.cwd).expanduser())
                stack.callback(os.chdir, old_cwd)
                shell_state.cwd = os.getcwd()
                stack.callback(setattr, shell_state, "cwd", previous_cwd)
            if args.session is not None:
                from .history import session_override
                stack.enter_context(session_override(args.session))
            _main(parser, args, output)
        except (OSError, ValueError) as exc:
            if output:
                output.finish(error=str(exc), status="error", exit_code=2)
            else:
                print(f"jarv: {exc}", file=sys.stderr)
            raise SystemExit(2) from None
        except SystemExit as exc:
            if output and not output.finished:
                output.finish(error="Invocation failed; see stderr for details.",
                              status="error", exit_code=exc.code or 0)
            raise
        except KeyboardInterrupt:
            if output:
                output.finish(error="Cancelled.", status="cancelled", exit_code=130)
            raise SystemExit(130) from None
        except Exception as exc:
            if output:
                output.finish(error=str(exc), status="error", exit_code=1)
            else:
                print(f"jarv: {exc}", file=sys.stderr)
            raise SystemExit(1) from None


def _reject_command_flags(parser, args):
    defaults = vars(parser.parse_args([]))
    changed = [name for name, value in vars(args).items()
               if name != "query" and name in defaults and value != defaults[name]]
    if changed:
        parser.error("runtime flags cannot be used with slash commands: " +
                     ", ".join("--" + name.replace("_", "-") for name in changed))


def _main(parser, args, output=None) -> None:
    query_parts: list[str] = args.query
    console = _console()
    _print_pending_results(quiet=args.quiet)

    # "jarv help" permanent alias (only when help is the sole argument)
    if args.prompt_file is None and len(query_parts) == 1 and query_parts[0].lower() == "help":
        _dispatch_command_query(query_parts)
        return

    # Prompt-file content is always a prompt, even if it begins with a slash.
    if query_parts and query_parts[0].startswith("/") and args.prompt_file is None:
        _dispatch_command_query(query_parts)
        return

    # Check if user typed a command name without the slash (e.g. "jarv set" instead of "jarv /set")
    if (query_parts and not query_parts[0].startswith("/") and not _stdin_is_piped()
            and not args.non_interactive and not args.prompt_file and output is None):
        result = _maybe_command(query_parts[0], query_parts[1:])
        if result is not None:
            _reject_command_flags(parser, args)
            _, command, rest = result
            if not _run_slash_command(command, rest, exit_on_error=True):
                console.print(f"[red]Unknown command:[/red] {command}")
                raise SystemExit(2)
            return

    # First-run: auto-trigger setup wizard if no config exists yet
    from .config import is_setup_complete

    config = _apply_cli_overrides(load_config(), args)
    if not args.provider and not is_setup_complete(config):
        if args.non_interactive:
            message = "Setup is incomplete. Configure a provider/API key before using --non-interactive."
            if output:
                output.finish(error=message, status="input_required", exit_code=3)
            else:
                print(message, file=sys.stderr)
            raise SystemExit(3)
        result = cmd_setup()
        if not isinstance(result, dict) or not is_setup_complete(result):
            sys.exit(result if type(result) is int else 1)
        config = result

        config = _apply_cli_overrides(config, args)

    if not validate_config(config):
        sys.exit(1)

    from .display import configure_monochrome, configure_output_display_lines

    configure_output_display_lines(config.get("tool_output_display_lines", "auto"))
    configure_monochrome(not config.get("colour", True))

    from .provider_auth import resolve_api_key
    from .provider_catalog import LOCAL_PROVIDERS

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
        if args.non_interactive or output is not None:
            raise ValueError("A prompt or non-empty stdin is required for one-shot execution.")
        run_heads_up_mode(config, None, args=args)
        return

    startup_wait = None
    try:
        if sys.stdout.isatty() and output is None:
            import time
            from .response_wait import start_response_wait

            started = time.perf_counter()
            indicator, live = start_response_wait(True, started, console=console)
            startup_wait = (started, indicator, live)

        if config.get("check_updates", True) and not args.quiet:
            from .update_check import _check_update_background, maybe_print_update_available

            maybe_print_update_available()
            import threading

            threading.Thread(target=_check_update_background, daemon=True).start()

        from .agent import run_agent

        wait_kwargs = {"startup_wait": startup_wait} if startup_wait is not None else {}
        if output is not None:
            wait_kwargs["ui"] = output
            config["_event_sink"] = output.event
        result = run_agent(query, config, client=None, new_session=args.new,
                           incognito=args.incognito, **wait_kwargs)
        if output is not None:
            status = getattr(result, "status", "success")
            # AgentRunResult fields are deliberately primitive protocol values.
            error = getattr(result, "error", None)
            error = error if isinstance(error, str) else None
            cancelled = getattr(result, "cancelled", False) is True
            exit_code = 130 if cancelled else (3 if status == "input_required" else 1 if error else 0)
            output.finish(text=getattr(result, "text", ""), error=error,
                          status="cancelled" if cancelled else status,
                          session_id=getattr(result, "session_id", None),
                          turns=getattr(result, "turns", 0), exit_code=exit_code)
        if getattr(result, "cancelled", False) is True:
            console.print("\n[dim]Cancelled.[/dim]")
            sys.exit(130)
        if isinstance(getattr(result, "error", None), str):
            if output is None and getattr(result, "status", None) in ("input_required", "limit"):
                console.print(f"[red]{result.error}[/red]")
            sys.exit(3 if getattr(result, "status", None) == "input_required" else 1)
    except KeyboardInterrupt:
        if output is not None:
            output.finish(error="Cancelled.", status="cancelled", exit_code=130)
        else:
            console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(130)
    finally:
        # Also restore the terminal if importing the agent fails or is cancelled.
        if startup_wait is not None:
            startup_wait[2].stop()


def run_heads_up_mode(
    config: dict,
    client,
    *,
    args: argparse.Namespace | None = None,
    agent_loader: tuple[dict, threading.Event] | None = None,
) -> None:
    from .headsup import run_heads_up_mode as run_fullscreen_heads_up

    def handle_slash(
        command: str,
        rest: list[str],
        current_config: dict,
        current_client,
        current_args: argparse.Namespace | None,
        unknown_help_hint: bool,
    ) -> tuple[dict, object]:
        return _handle_heads_up_slash_command(
            command,
            rest,
            config=current_config,
            client=current_client,
            args=current_args,
            unknown_help_hint=unknown_help_hint,
        )

    run_fullscreen_heads_up(
        config,
        client,
        args=args,
        agent_loader=agent_loader,
        handle_slash=handle_slash,
        maybe_command=_maybe_command,
    )


if __name__ == "__main__":
    main()
