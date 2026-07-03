"""Single source of truth for jarv slash-command metadata and dispatch."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CommandMeta:
    takes_rest: bool
    needs_nudge: bool = False
    mutates_config: bool = False
    summary: str = ""
    arg_hint: str = ""
    menu: bool = True


COMMANDS: dict[str, CommandMeta] = {
    "setup": CommandMeta(True, mutates_config=True, summary="Run setup or jump to a step", arg_hint="[step]"),
    "help": CommandMeta(False, summary="Show help menu"),
    "about": CommandMeta(False, summary="Show detailed reference information"),
    "update": CommandMeta(False, needs_nudge=True, summary="Update jarv"),
    "new": CommandMeta(False, needs_nudge=True, summary="Start a fresh session"),
    "archive": CommandMeta(False, needs_nudge=True, summary="Archive this session and start fresh"),
    "session": CommandMeta(True, needs_nudge=True, summary="List sessions", menu=False),
    "sessions": CommandMeta(True, needs_nudge=True, summary="List sessions"),
    "history": CommandMeta(False, needs_nudge=True, summary="Show recent conversation history"),
    "usage": CommandMeta(True, summary="Show token usage", arg_hint="[session|day|week|month|all]"),
    "set": CommandMeta(True, needs_nudge=True, mutates_config=True, summary="Set a configuration value", arg_hint="<key> <value>"),
    "unset": CommandMeta(True, needs_nudge=True, mutates_config=True, summary="Reset or remove a configuration value", arg_hint="<key>"),
    "config": CommandMeta(False, summary="Show raw configuration values"),
    "settings": CommandMeta(False, needs_nudge=True, mutates_config=True, summary="Open common controls"),
    "undo": CommandMeta(True, needs_nudge=True, summary="Unsend the last n exchanges", arg_hint="[n]"),
    "redo": CommandMeta(True, needs_nudge=True, summary="Restore undone exchanges", arg_hint="[n]"),
    "tree": CommandMeta(False, needs_nudge=True, summary="Browse the session as a tree — fork, edit, or resume any prompt"),
    "btw": CommandMeta(True, needs_nudge=True, summary="Ask an aside without derailing the main thread", arg_hint="<question>"),
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
