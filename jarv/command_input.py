"""Keyboard input helpers shared by interactive command screens."""

import os
import sys
from collections import deque
from collections.abc import Iterable
from contextlib import contextmanager


MOUSE_CAPTURE_ENABLE = "\x1b[?1000h\x1b[?1006h"
MOUSE_CAPTURE_DISABLE = "\x1b[?1006l\x1b[?1000l"
_PENDING_KEYS: deque[str] = deque()
_REPEATABLE_NAV_KEYS = frozenset({"UP", "DOWN", "LEFT", "RIGHT", "PAGEUP", "PAGEDOWN"})
_POSIX_INPUT_POLL_INTERVAL = 0.1
_LAST_TERMINAL_SIZE: tuple[int, int] | None = None


@contextmanager
def mouse_capture():
    """Capture POSIX terminal mouse input while a full-screen view is active."""
    if sys.platform == "win32" or not sys.stdout.isatty():
        yield
        return

    sys.stdout.write(MOUSE_CAPTURE_ENABLE)
    sys.stdout.flush()
    try:
        yield
    finally:
        sys.stdout.write(MOUSE_CAPTURE_DISABLE)
        sys.stdout.flush()


def _read_until_any(chars: set[str]) -> str:
    data = ""
    while True:
        ch = sys.stdin.read(1)
        if not ch:
            return data
        data += ch
        if ch in chars:
            return data


def _parse_sgr_mouse(sequence: str) -> str:
    parts = sequence[:-1].split(";") if sequence and sequence[-1] in ("M", "m") else sequence.split(";")
    if len(parts) != 3:
        return "OTHER"
    try:
        button = int(parts[0])
    except ValueError:
        return "OTHER"

    if not button & 64:
        return "OTHER"

    wheel = button & 3
    if wheel == 0:
        return "UP"
    if wheel == 1:
        return "DOWN"
    if wheel == 2:
        return "PAGEUP"
    if wheel == 3:
        return "PAGEDOWN"
    return "OTHER"


def _terminal_size() -> tuple[int, int] | None:
    for stream in (0, 1, 2):
        try:
            size = os.get_terminal_size(stream)
        except OSError:
            continue
        return max(1, size.columns), max(1, size.lines)
    return None


def _terminal_size_changed() -> bool:
    global _LAST_TERMINAL_SIZE

    size = _terminal_size()
    if size is None:
        return False
    if _LAST_TERMINAL_SIZE is None:
        _LAST_TERMINAL_SIZE = size
        return False
    if size == _LAST_TERMINAL_SIZE:
        return False
    _LAST_TERMINAL_SIZE = size
    return True


def _read_posix_char() -> str | None:
    import select

    if _terminal_size_changed():
        return None

    while True:
        readable, _, _ = select.select([sys.stdin], [], [], _POSIX_INPUT_POLL_INTERVAL)
        if readable:
            return sys.stdin.read(1)
        if _terminal_size_changed():
            return None


def _key_available() -> bool:
    if _PENDING_KEYS:
        return True
    try:
        if sys.platform == "win32":
            import msvcrt
            return bool(msvcrt.kbhit())
        import select
        return bool(select.select([sys.stdin], [], [], 0)[0])
    except (ImportError, OSError, TypeError, ValueError):
        return False


def _read_key_with_repeats(
    text_mode: bool = False,
    *,
    repeatable: Iterable[str] = _REPEATABLE_NAV_KEYS,
    max_count: int = 128,
) -> tuple[str, int]:
    """Read one key and fold immediately queued identical navigation repeats.

    Full-screen Rich views are expensive enough that some terminals can queue
    key-repeat events faster than jarv can redraw. Coalescing identical queued
    arrows lets menus advance several rows per refresh while preserving the
    first different key for the next input loop.
    """
    key = _read_key(text_mode=text_mode)
    repeatable_keys = frozenset(repeatable)
    if key not in repeatable_keys or max_count <= 1:
        return key, 1

    count = 1
    while count < max_count and _key_available():
        next_key = _read_key(text_mode=text_mode)
        if next_key != key:
            _PENDING_KEYS.appendleft(next_key)
            break
        count += 1
    return key, count


def _read_key(text_mode: bool = False) -> str:
    """Read a single keypress and return a normalised token.

    Returns one of: UP, DOWN, LEFT, RIGHT, HOME, END, PAGEUP, PAGEDOWN,
    ENTER, ESC, TAB, CTRL_F, BACKSPACE, or the raw character.  Raises KeyboardInterrupt on
    Ctrl-C.  When ``text_mode`` is True, the convenience q/Q → ESC mapping is
    disabled so a search query can include those letters.
    """
    if _PENDING_KEYS:
        return _PENDING_KEYS.popleft()

    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            second = msvcrt.getwch()
            return {
                "H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT",
                "G": "HOME", "O": "END",
                "I": "PAGEUP", "Q": "PAGEDOWN",
            }.get(second, "OTHER")
        if ch == "\r":
            return "ENTER"
        if ch == "\t":
            return "TAB"
        if ch == "\x1b":
            return "ESC"
        if not text_mode and ch in ("q", "Q"):
            return "ESC"
        if ch == "\x06":
            return "CTRL_F"
        if ch in ("\x08", "\x7f"):
            return "BACKSPACE"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = _read_posix_char()
            if ch is None:
                return "RESIZE"
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "<":
                        return _parse_sgr_mouse(_read_until_any({"M", "m"}))
                    if ch3 in ("5", "6"):
                        sys.stdin.read(1)  # consume trailing ~
                    return {
                        "A": "UP", "B": "DOWN", "D": "LEFT", "C": "RIGHT",
                        "H": "HOME", "F": "END",
                        "5": "PAGEUP", "6": "PAGEDOWN",
                    }.get(ch3, "OTHER")
                return "ESC"
            if ch in ("\r", "\n"):
                return "ENTER"
            if ch == "\t":
                return "TAB"
            if not text_mode and ch in ("q", "Q"):
                return "ESC"
            if ch == "\x06":
                return "CTRL_F"
            if ch in ("\x7f", "\x08"):
                return "BACKSPACE"
            if ch == "\x03":
                raise KeyboardInterrupt
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

