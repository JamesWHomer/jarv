"""Named terminal-capability quirks for the TUI layer.

The fullscreen views (heads-up, settings, the session browser, ...) span the
full terminal width, so their right border sits flush against the screen edge.
The one redraw quirk that still needs handling is the stale right edge a
previous, wider frame can leave on Windows ConPTY (Windows Terminal) and WSL.
That is cleared by emitting an erase-to-end-of-line before every rendered row
(see ``jarv.tui_frame.EraseTrailingColumns``) -- not by reserving columns, which
would only re-introduce the visible gap the erase already makes unnecessary.

``is_wsl()`` is exposed as a detection seam for future capability-specific
tuning, but is intentionally *not* wired into rendering today.
"""

from __future__ import annotations

import functools
import os
import platform


@functools.lru_cache(maxsize=1)
def is_wsl() -> bool:
    """Best-effort detection of the Windows Subsystem for Linux.

    Detection seam for future capability tuning; not currently used to vary
    rendering (see module docstring).
    """
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def supports_erase_eol() -> bool:
    """Whether emitting an erase-to-end-of-line control is safe.

    True for real terminals; the renderable that uses it already guards on
    ``console.is_terminal`` per frame, so this stays permissive. Set
    ``JARV_NO_ERASE_EOL=1`` to opt out on a terminal that mishandles
    ``\\x1b[0K`` (the control then never reaches the wire and views fall back to
    plain full-width repaints).
    """
    return os.environ.get("JARV_NO_ERASE_EOL") != "1"
