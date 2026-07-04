"""Single source of truth for jarv slash-command metadata and dispatch."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class ArgChoice:
    """One completable value for a command's first argument.

    ``final`` says whether the command is runnable once this value is chosen
    (``/usage day``) or expects further input after it (``/set model `` still
    needs a value).
    """

    value: str
    summary: str = ""
    final: bool = True


def _usage_period_choices() -> list[ArgChoice]:
    return [
        ArgChoice("session", "This session"),
        ArgChoice("day", "Today"),
        ArgChoice("week", "Past week"),
        ArgChoice("month", "Past month"),
        ArgChoice("all", "All time"),
    ]


_SETUP_STEP_ORDER = ("provider", "key", "model", "base_url")
_SETUP_STEP_SUMMARIES = {
    "provider": "Choose the provider",
    "key": "Set the API key",
    "model": "Choose the model",
    "base_url": "Set a custom base URL",
}


def _setup_step_choices() -> list[ArgChoice]:
    from .setup import SETUP_STEPS

    ordered = [step for step in _SETUP_STEP_ORDER if step in SETUP_STEPS]
    ordered += sorted(SETUP_STEPS - set(ordered))
    return [ArgChoice(step, _SETUP_STEP_SUMMARIES.get(step, "")) for step in ordered]


def _config_key_choices(*, final: bool) -> Callable[[], list[ArgChoice]]:
    def choices() -> list[ArgChoice]:
        from .config import DEFAULT_CONFIG

        return [
            ArgChoice(key, f"default {default!r}", final=final)
            for key, default in DEFAULT_CONFIG.items()
        ]

    return choices


@dataclass(frozen=True)
class CommandMeta:
    takes_rest: bool
    needs_nudge: bool = False
    mutates_config: bool = False
    summary: str = ""
    arg_hint: str = ""
    menu: bool = True
    # Completable values for the *first* argument (None → free-form or none).
    # A callable so dynamic sources (config keys, setup steps) stay lazy.
    arg_choices: Callable[[], list[ArgChoice]] | None = field(default=None, compare=False)


# Declaration order is the slash-menu order: everyday commands first, one-time
# and reference commands last. /help groups its rows explicitly, so this order
# only drives the menu.
COMMANDS: dict[str, CommandMeta] = {
    "settings": CommandMeta(False, needs_nudge=True, mutates_config=True, summary="Open common controls"),
    "usage": CommandMeta(True, summary="Show token usage", arg_hint="[session|day|week|month|all]", arg_choices=_usage_period_choices),
    "new": CommandMeta(False, needs_nudge=True, summary="Start a fresh session"),
    "tree": CommandMeta(False, needs_nudge=True, summary="Browse the session as a tree — fork, edit, or resume any prompt"),
    "btw": CommandMeta(True, needs_nudge=True, summary="Ask an aside without derailing the main thread", arg_hint="<question>"),
    "history": CommandMeta(False, needs_nudge=True, summary="Show recent conversation history"),
    "undo": CommandMeta(True, needs_nudge=True, summary="Unsend the last n exchanges", arg_hint="[n]"),
    "redo": CommandMeta(True, needs_nudge=True, summary="Restore undone exchanges", arg_hint="[n]"),
    "sessions": CommandMeta(True, needs_nudge=True, summary="List sessions"),
    "session": CommandMeta(True, needs_nudge=True, summary="List sessions", menu=False),
    "archive": CommandMeta(False, needs_nudge=True, summary="Archive this session and start fresh"),
    "set": CommandMeta(True, needs_nudge=True, mutates_config=True, summary="Set a configuration value", arg_hint="<key> <value>", arg_choices=_config_key_choices(final=False)),
    "unset": CommandMeta(True, needs_nudge=True, mutates_config=True, summary="Reset or remove a configuration value", arg_hint="<key>", arg_choices=_config_key_choices(final=True)),
    "config": CommandMeta(False, summary="Show raw configuration values"),
    "setup": CommandMeta(True, mutates_config=True, summary="Run setup or jump to a step", arg_hint="[step]", arg_choices=_setup_step_choices),
    "help": CommandMeta(False, summary="Show help menu"),
    "about": CommandMeta(False, summary="Show detailed reference information"),
    "update": CommandMeta(False, needs_nudge=True, summary="Update jarv"),
}

CONFIG_MUTATING_COMMANDS = frozenset(
    f"/{name}" for name, meta in COMMANDS.items() if meta.mutates_config
)

HANDLER_SPECS: dict[str, tuple[str, str]] = {
    "setup": ("jarv.cli", "cmd_setup"),
    "help": ("jarv.commands", "print_help"),
    "about": ("jarv.commands", "print_about"),
    "update": ("jarv.commands", "cmd_update"),
    "new": ("jarv.commands", "cmd_new"),
    "archive": ("jarv.session_commands", "cmd_archive"),
    "session": ("jarv.session_browser", "cmd_sessions"),
    "sessions": ("jarv.session_browser", "cmd_sessions"),
    "history": ("jarv.session_commands", "cmd_history"),
    "usage": ("jarv.usage_command", "cmd_usage"),
    "set": ("jarv.commands", "cmd_set"),
    "unset": ("jarv.commands", "cmd_unset"),
    "config": ("jarv.commands", "cmd_config"),
    "settings": ("jarv.settings_command", "cmd_settings"),
    "undo": ("jarv.undo_commands", "cmd_undo"),
    "redo": ("jarv.undo_commands", "cmd_redo"),
    "tree": ("jarv.tree_command", "cmd_tree"),
    "btw": ("jarv.commands", "cmd_btw"),
}


def slash_name(name: str) -> str:
    return f"/{name.lower().lstrip('/')}"


def suggest_commands(name: str, limit: int = 3) -> list[str]:
    """Closest known commands to a mistyped ``name`` (without the slash).

    Prefix and substring matches lead (declaration order), then close spellings
    via difflib. Used for the "did you mean" hint on unknown slash commands.
    """
    import difflib

    needle = name.lower().lstrip("/")
    if not needle:
        return []
    candidates = list(COMMANDS) + ["exit", "quit"]
    ranked = [c for c in candidates if c.startswith(needle)]
    ranked += [c for c in candidates if needle in c and c not in ranked]
    for close in difflib.get_close_matches(needle, candidates, n=limit, cutoff=0.6):
        if close not in ranked:
            ranked.append(close)
    return ranked[:limit]


def command_takes_rest(name: str) -> bool | None:
    meta = COMMANDS.get(name.lower().lstrip("/"))
    return meta.takes_rest if meta is not None else None


def parse_command_alias(
    first_word: str,
    rest: list[str],
) -> tuple[str, list[str]] | None:
    """Return (slash_command, rest) when input matches a known command signature."""
    name = first_word.lower()
    takes_rest = command_takes_rest(name)
    if takes_rest is None:
        return None
    if not takes_rest and rest:
        return None
    return slash_name(name), rest


def load_handlers() -> dict[str, Callable]:
    def lazy_handler(module_name: str, attribute: str) -> Callable:
        def handler(*args, **kwargs):
            module = importlib.import_module(module_name)
            target = getattr(module, attribute)
            return target(*args, **kwargs)

        handler.__name__ = attribute
        handler.__qualname__ = attribute
        handler.__module__ = module_name
        return handler

    return {
        name: lazy_handler(module_name, attribute)
        for name, (module_name, attribute) in HANDLER_SPECS.items()
    }


def build_dispatch(
    handlers: dict[str, Callable] | None = None,
) -> dict[str, tuple[Callable, bool, bool]]:
    """Map slash commands to (handler, needs_nudge, takes_rest)."""
    handlers = handlers or load_handlers()
    dispatch: dict[str, tuple[Callable, bool, bool]] = {}
    for name, meta in COMMANDS.items():
        handler = handlers.get(name)
        if handler is None:
            continue
        dispatch[slash_name(name)] = (handler, meta.needs_nudge, meta.takes_rest)
    return dispatch
