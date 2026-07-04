"""Pure enumeration and filtering for the heads-up slash-command menu.

No Rich and no I/O — just data derived from :mod:`jarv.command_registry`, so the
ordering and matching rules can be unit-tested in isolation. All rendering and
key handling lives in ``headsup.py``.

The menu has two modes, both expressed as lists of :class:`MenuEntry`:

- **command mode** (:func:`menu_entries`) — completing the command token itself
  ("/se" → /settings, /sessions, …);
- **argument mode** (:func:`argument_entries`) — completing the first argument
  of a command that declares ``arg_choices`` ("/usage d" → day).
"""

from __future__ import annotations

from dataclasses import dataclass

from .command_registry import COMMANDS


@dataclass(frozen=True)
class MenuEntry:
    name: str                     # match target: "settings" or an arg value "day"
    display: str                  # what the row shows: "/settings" or "day"
    summary: str                  # "Open common controls"
    arg_hint: str                 # "" | "<key> <value>" | "[n]" ...
    insert: str                   # buffer text on completion: "/settings", "/usage ", "/usage day"
    runs_on_enter: bool           # Enter runs insert.strip(); False → Enter completes like Tab
    aliases: tuple[str, ...] = ()  # extra names that match but keep the canonical display


# Hidden spellings that should still match while typing: the row shown (and the
# text inserted) stays the canonical command.
_COMMAND_ALIASES: dict[str, tuple[str, ...]] = {
    "sessions": ("session",),
    "exit": ("quit",),
}


# ``/exit`` (and its ``/quit`` twin) are handled directly in headsup rather than
# through COMMANDS, but they belong in the discovery menu. ``/exit`` is the
# canonical label.
_EXIT_ENTRY = MenuEntry(
    name="exit",
    display="/exit",
    summary="Leave heads-up mode",
    arg_hint="",
    insert="/exit",
    runs_on_enter=True,
    aliases=_COMMAND_ALIASES["exit"],
)


def menu_entries() -> list[MenuEntry]:
    """All commands surfaced in the menu, in registry declaration order.

    Commands with possible arguments complete to "/name " (trailing space) and
    defer running to the user; parameterless ones complete to the bare command
    and run on Enter.
    """
    entries = [
        MenuEntry(
            name=name,
            display=f"/{name}",
            summary=meta.summary,
            arg_hint=meta.arg_hint,
            insert=f"/{name} " if meta.takes_rest else f"/{name}",
            runs_on_enter=not meta.takes_rest,
            aliases=_COMMAND_ALIASES.get(name, ()),
        )
        for name, meta in COMMANDS.items()
        if meta.menu
    ]
    entries.append(_EXIT_ENTRY)
    return entries


def argument_entries(command: str) -> list[MenuEntry]:
    """First-argument choices for ``command``, or ``[]`` when it declares none.

    A final choice ("/usage day") runs on Enter; a non-final one ("/set model")
    completes with a trailing space so the user can keep typing the value.
    """
    meta = COMMANDS.get(command)
    if meta is None or meta.arg_choices is None:
        return []
    entries: list[MenuEntry] = []
    for choice in meta.arg_choices():
        completed = f"/{command} {choice.value}"
        entries.append(
            MenuEntry(
                name=choice.value,
                display=choice.value,
                summary=choice.summary,
                arg_hint="",
                insert=completed if choice.final else completed + " ",
                runs_on_enter=choice.final,
            )
        )
    return entries


def filter_entries(entries: list[MenuEntry], query: str) -> list[MenuEntry]:
    """Filter ``entries`` by ``query`` (the text after the leading ``/``).

    Case-insensitive. Prefix matches come before substring matches; declaration
    order is preserved within each tier. Hidden aliases match too but rank with
    their canonical entry. An empty query returns every entry. Deliberately
    simple and predictable — not fuzzy.
    """
    needle = query.lower()
    if not needle:
        return list(entries)
    prefix: list[MenuEntry] = []
    substring: list[MenuEntry] = []
    for entry in entries:
        names = (entry.name.lower(), *(alias.lower() for alias in entry.aliases))
        if any(name.startswith(needle) for name in names):
            prefix.append(entry)
        elif any(needle in name for name in names):
            substring.append(entry)
    return prefix + substring
