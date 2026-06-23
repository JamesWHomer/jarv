"""Named terminal-capability quirks for the TUI layer.

Historically the heads-up and settings renderers each inlined magic constants and
ad-hoc checks to work around terminal redraw quirks (most visibly WSL/ConPTY
keeping stale border cells in the auto-wrap column). This module gives those
quirks a single named home so the workarounds are discoverable and individually
testable, instead of being scattered literals.

The wrap guard is applied universally on purpose: the stale right-edge artifact
shows up on Windows ConPTY (Windows Terminal) as well as WSL, and reserving a
couple of columns is cheap. ``is_wsl()`` is exposed as a detection seam for future
capability-specific tuning, but is intentionally *not* wired into the guard count
today -- doing so would change rendering on terminals that also need the guard.
"""

from __future__ import annotations

import functools
import os
import platform

# Columns left spare at the right edge so an erase-to-end-of-line can clear a
# stale border left by a previous, wider frame without pushing the cursor into
# the terminal's auto-wrap column.
_WRAP_GUARD_COLUMNS = 2


def wrap_guard_columns() -> int:
    """Number of right-edge columns to leave spare when sizing a full-width panel."""
    return _WRAP_GUARD_COLUMNS


@functools.lru_cache(maxsize=1)
def is_wsl() -> bool:
    """Best-effort detection of the Windows Subsystem for Linux.

    Detection seam for future capability tuning; not currently used to vary the
    wrap guard (see module docstring).
    """
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def supports_erase_eol() -> bool:
    """Whether emitting an erase-to-end-of-line control is safe.

    True for real terminals; the renderable that uses it already guards on
    ``console.is_terminal`` per frame, so this stays permissive.
    """
    return True
