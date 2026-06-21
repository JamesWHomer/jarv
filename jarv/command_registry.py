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


COMMANDS: dict[str, CommandMeta] = {
    "setup": CommandMeta(True, needs_nudge=False, mutates_config=True),
    "help": CommandMeta(False),
    "about": CommandMeta(False),
    "update": CommandMeta(False, needs_nudge=True),
    "new": CommandMeta(False, needs_nudge=True),
    "archive": CommandMeta(False, needs_nudge=True),
    "session": CommandMeta(True, needs_nudge=True),
    "sessions": CommandMeta(True, needs_nudge=True),
    "history": CommandMeta(False, needs_nudge=True),
    "usage": CommandMeta(True),
    "set": CommandMeta(True, needs_nudge=True, mutates_config=True),
    "unset": CommandMeta(True, needs_nudge=True, mutates_config=True),
    "config": CommandMeta(False),
    "settings": CommandMeta(False, needs_nudge=True, mutates_config=True),
    "undo": CommandMeta(True, needs_nudge=True),
    "redo": CommandMeta(True, needs_nudge=True),
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
    "archive": ("jarv.commands", "cmd_archive"),
    "session": ("jarv.commands", "cmd_sessions"),
    "sessions": ("jarv.commands", "cmd_sessions"),
    "history": ("jarv.commands", "cmd_history"),
    "usage": ("jarv.commands", "cmd_usage"),
    "set": ("jarv.commands", "cmd_set"),
    "unset": ("jarv.commands", "cmd_unset"),
    "config": ("jarv.commands", "cmd_config"),
    "settings": ("jarv.commands", "cmd_settings"),
    "undo": ("jarv.commands", "cmd_undo"),
    "redo": ("jarv.commands", "cmd_redo"),
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
