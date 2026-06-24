"""Pure enumeration and filtering for the heads-up slash-command menu.

No Rich and no I/O — just data derived from :mod:`jarv.command_registry`, so the
ordering and matching rules can be unit-tested in isolation. All rendering and
key handling lives in ``headsup.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .command_registry import COMMANDS


@dataclass(frozen=True)
class MenuEntry:
    name: str        # "settings"
    display: str     # "/settings"
    summary: str     # "Open common controls"
    arg_hint: str    # "" | "<key> <value>" | "[n]" ...
    takes_rest: bool # parameters possible?


# ``/exit`` (and its ``/quit`` twin) are handled directly in headsup rather than
# through COMMANDS, but they belong in the discovery menu. ``/exit`` is the
# canonical label.
_EXIT_ENTRY = MenuEntry(
    name="exit",
    display="/exit",
    summary="Leave heads-up mode",
    arg_hint="",
    takes_rest=False,
)


def menu_entries() -> list[MenuEntry]:
    """All commands surfaced in the menu, in registry declaration order."""
    entries = [
        MenuEntry(
            name=name,
            display=f"/{name}",
            summary=meta.summary,
            arg_hint=meta.arg_hint,
            takes_rest=meta.takes_rest,
        )
        for name, meta in COMMANDS.items()
        if meta.menu
    ]
    entries.append(_EXIT_ENTRY)
    return entries


def filter_entries(entries: list[MenuEntry], query: str) -> list[MenuEntry]:
    """Filter ``entries`` by ``query`` (the text after the leading ``/``).

    Case-insensitive. Prefix matches come before substring matches; registry
    order is preserved within each tier. An empty query returns every entry.
    Deliberately simple and predictable — not fuzzy.
    """
    needle = query.lower()
    if not needle:
        return list(entries)
    prefix: list[MenuEntry] = []
    substring: list[MenuEntry] = []
    for entry in entries:
        name = entry.name.lower()
        if name.startswith(needle):
            prefix.append(entry)
        elif needle in name:
            substring.append(entry)
    return prefix + substring
