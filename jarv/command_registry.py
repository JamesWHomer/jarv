"""Single source of truth for jarv slash-command metadata and dispatch."""

from __future__ import annotations

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
    from .cli import cmd_setup
    from .commands import (
        cmd_archive,
        cmd_config,
        cmd_history,
        cmd_new,
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
    from .settings_command import cmd_settings

    return {
        "setup": cmd_setup,
        "help": print_help,
        "about": print_about,
        "update": cmd_update,
        "new": cmd_new,
        "archive": cmd_archive,
        "session": cmd_sessions,
        "sessions": cmd_sessions,
        "history": cmd_history,
        "usage": cmd_usage,
        "set": cmd_set,
        "unset": cmd_unset,
        "config": cmd_config,
        "settings": cmd_settings,
        "undo": cmd_undo,
        "redo": cmd_redo,
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
