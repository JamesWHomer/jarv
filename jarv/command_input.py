"""Keyboard input helpers shared by interactive command screens."""

import os
import re
import sys
from collections import deque
from collections.abc import Iterable
from contextlib import contextmanager

from rich.cells import cell_len, get_character_cell_size


MOUSE_CAPTURE_ENABLE = "\x1b[?1000h\x1b[?1006h"
MOUSE_CAPTURE_DISABLE = "\x1b[?1006l\x1b[?1000l"
CURSOR_HIDE = "\x1b[?25l"
CURSOR_SHOW = "\x1b[?25h"
ANSI_RESET = "\x1b[0m"
_PENDING_KEYS: deque[str] = deque()
_REPEATABLE_NAV_KEYS = frozenset({"UP", "DOWN", "LEFT", "RIGHT", "PAGEUP", "PAGEDOWN"})
_MOUSE_WHEEL_KEYS = {
    0: "MOUSE_WHEEL_UP",
    1: "MOUSE_WHEEL_DOWN",
    2: "MOUSE_WHEEL_PAGEUP",
    3: "MOUSE_WHEEL_PAGEDOWN",
}
_POSIX_INPUT_POLL_INTERVAL = 0.1
_LAST_TERMINAL_SIZE: tuple[int, int] | None = None
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class TextInput(str):
    """A queued group of printable characters from an editable input."""


@contextmanager
def mouse_capture():
    """Capture terminal mouse input while a full-screen view is active."""
    if not sys.stdout.isatty():
        yield
        return

    with _windows_virtual_terminal_input():
        sys.stdout.write(MOUSE_CAPTURE_ENABLE)
        sys.stdout.flush()
        try:
            yield
        finally:
            sys.stdout.write(MOUSE_CAPTURE_DISABLE)
            sys.stdout.flush()


@contextmanager
def _windows_virtual_terminal_input():
    if sys.platform != "win32":
        yield
        return

    try:
        import ctypes
    except ImportError:
        yield
        return

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-10)
    mode = ctypes.c_uint()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        yield
        return

    original_mode = mode.value
    enabled_mode = original_mode | 0x0200
    changed = enabled_mode != original_mode and kernel32.SetConsoleMode(handle, enabled_mode)
    try:
        yield
    finally:
        if changed:
            kernel32.SetConsoleMode(handle, original_mode)


def _read_until_any(chars: set[str]) -> str:
    data = ""
    while True:
        ch = sys.stdin.read(1)
        if not ch:
            return data
        data += ch
        if ch in chars:
            return data


def _windows_key_available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import msvcrt

        return bool(msvcrt.kbhit())
    except (ImportError, OSError):
        return False


def _read_windows_until_any(msvcrt, chars: set[str]) -> str:
    data = ""
    while True:
        ch = msvcrt.getwch()
        if not ch:
            return data
        data += ch
        if ch in chars:
            return data


def _parse_sgr_mouse(sequence: str, *, translate_wheel: bool = True) -> str:
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
    if not translate_wheel:
        return _MOUSE_WHEEL_KEYS.get(wheel, "OTHER")
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


def requeue_key(key: str) -> None:
    """Return a key read out-of-band to the foreground input loop."""
    _PENDING_KEYS.appendleft(key)


def _read_key_with_repeats(
    text_mode: bool = False,
    *,
    repeatable: Iterable[str] = _REPEATABLE_NAV_KEYS,
    max_count: int = 128,
    batch_text: bool = False,
    translate_mouse_wheel: bool = True,
) -> tuple[str, int]:
    """Read one key and fold immediately queued identical navigation repeats.

    Full-screen Rich views are expensive enough that some terminals can queue
    key-repeat events faster than jarv can redraw. Coalescing identical queued
    arrows lets menus advance several rows per refresh while preserving the
    first different key for the next input loop. Editable views can also batch
    queued printable characters so a paste triggers one redraw instead of one
    redraw per character.
    """
    def read_key() -> str:
        if translate_mouse_wheel:
            return _read_key(text_mode=text_mode)
        return _read_key(text_mode=text_mode, translate_mouse_wheel=False)

    key = read_key()
    if (
        batch_text
        and text_mode
        and isinstance(key, str)
        and len(key) == 1
        and key.isprintable()
    ):
        inserted = [key]
        while len(inserted) < max_count and _key_available():
            next_key = read_key()
            if (
                isinstance(next_key, str)
                and len(next_key) == 1
                and next_key.isprintable()
            ):
                inserted.append(next_key)
                continue
            _PENDING_KEYS.appendleft(next_key)
            break
        return TextInput("".join(inserted)), 1

    repeatable_keys = frozenset(repeatable)
    if key not in repeatable_keys or max_count <= 1:
        return key, 1

    count = 1
    while count < max_count and _key_available():
        next_key = read_key()
        if next_key != key:
            _PENDING_KEYS.appendleft(next_key)
            break
        count += 1
    return key, count


def _read_key(text_mode: bool = False, *, translate_mouse_wheel: bool = True) -> str:
    """Read a single keypress and return a normalised token.

    Returns one of: UP, DOWN, LEFT, RIGHT, HOME, END, PAGEUP, PAGEDOWN,
    ENTER, ESC, TAB, CTRL_F, CTRL_S, BACKSPACE, DELETE, or the raw character. Raises KeyboardInterrupt on
    Ctrl-C.  When ``text_mode`` is True, the convenience q/Q → ESC mapping is
    disabled so a search query can include those letters. When
    ``translate_mouse_wheel`` is False, SGR wheel input returns MOUSE_WHEEL_*
    tokens instead of arrow/page navigation tokens.
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
                "I": "PAGEUP", "Q": "PAGEDOWN", "S": "DELETE",
            }.get(second, "OTHER")
        if ch == "\r":
            return "ENTER"
        if ch == "\t":
            return "TAB"
        if ch == "\x1b":
            if _windows_key_available():
                ch2 = msvcrt.getwch()
                if ch2 == "[" and _windows_key_available():
                    ch3 = msvcrt.getwch()
                    if ch3 == "<":
                        return _parse_sgr_mouse(
                            _read_windows_until_any(msvcrt, {"M", "m"}),
                            translate_wheel=translate_mouse_wheel,
                        )
                    if ch3 in ("5", "6", "3") and _windows_key_available():
                        msvcrt.getwch()  # consume trailing ~
                    return {
                        "A": "UP", "B": "DOWN", "D": "LEFT", "C": "RIGHT",
                        "H": "HOME", "F": "END",
                        "5": "PAGEUP", "6": "PAGEDOWN", "3": "DELETE",
                    }.get(ch3, "OTHER")
            return "ESC"
        if not text_mode and ch in ("q", "Q"):
            return "ESC"
        if ch == "\x06":
            return "CTRL_F"
        if ch == "\x13":
            return "CTRL_S"
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
                        return _parse_sgr_mouse(
                            _read_until_any({"M", "m"}),
                            translate_wheel=translate_mouse_wheel,
                        )
                    if ch3 in ("5", "6"):
                        sys.stdin.read(1)  # consume trailing ~
                    if ch3 == "3":
                        sys.stdin.read(1)  # consume trailing ~
                    return {
                        "A": "UP", "B": "DOWN", "D": "LEFT", "C": "RIGHT",
                        "H": "HOME", "F": "END",
                        "5": "PAGEUP", "6": "PAGEDOWN", "3": "DELETE",
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
            if ch == "\x13":
                return "CTRL_S"
            if ch in ("\x7f", "\x08"):
                return "BACKSPACE"
            if ch == "\x03":
                raise KeyboardInterrupt
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _render_editable_line(
    prompt: str,
    text: str,
    cursor: int,
    *,
    text_style: str = "",
    write=None,
    columns: int | None = None,
) -> None:
    write = write or sys.stdout.write
    if columns is None:
        size = _terminal_size()
        columns = size[0] if size is not None else 80

    prompt_width = cell_len(_ANSI_ESCAPE_RE.sub("", prompt))
    # Keep one column unused so writing at the right edge cannot auto-wrap.
    available = max(1, columns - prompt_width - 1)

    start = cursor
    cursor_column = 0
    while start:
        width = get_character_cell_size(text[start - 1])
        if cursor_column + width > available:
            break
        start -= 1
        cursor_column += width

    end = cursor
    visible_width = cursor_column
    while end < len(text):
        width = get_character_cell_size(text[end])
        if visible_width + width > available:
            break
        end += 1
        visible_width += width
    visible = text[start:end]

    repaint = [CURSOR_HIDE, "\r\x1b[2K", prompt]
    if text_style:
        repaint.extend((text_style, visible, ANSI_RESET))
    else:
        repaint.append(visible)
    trailing = visible_width - cursor_column
    if trailing:
        repaint.append(f"\x1b[{trailing}D")
    repaint.append(CURSOR_SHOW)
    write("".join(repaint))
    sys.stdout.flush()


def read_editable_line(
    prompt: str,
    initial: str = "",
    *,
    text_style: str = "",
    read_key=None,
    key_available=None,
    write=None,
) -> str:
    """Read one editable line with cross-platform raw key handling."""
    if read_key is None:
        read_key = lambda: _read_key(text_mode=True)
        key_available = key_available or _key_available
    else:
        key_available = key_available or (lambda: False)
    write = write or sys.stdout.write
    chars = list(initial.replace("\r", " ").replace("\n", " "))
    cursor = len(chars)
    pending: deque[str] = deque()

    _render_editable_line(
        prompt,
        "".join(chars),
        cursor,
        text_style=text_style,
        write=write,
    )
    try:
        while True:
            try:
                key = pending.popleft() if pending else read_key()
            except KeyboardInterrupt:
                if chars:
                    chars.clear()
                    cursor = 0
                    _render_editable_line(
                        prompt,
                        "",
                        cursor,
                        text_style=text_style,
                        write=write,
                    )
                    continue
                raise

            if key == "ENTER":
                write("\n")
                return "".join(chars)
            if key == "LEFT":
                cursor = max(0, cursor - 1)
            elif key == "RIGHT":
                cursor = min(len(chars), cursor + 1)
            elif key == "HOME":
                cursor = 0
            elif key == "END":
                cursor = len(chars)
            elif key == "BACKSPACE":
                if cursor:
                    del chars[cursor - 1]
                    cursor -= 1
            elif key == "DELETE":
                if cursor < len(chars):
                    del chars[cursor]
            elif key == "\x04":
                if not chars:
                    raise EOFError
                if cursor < len(chars):
                    del chars[cursor]
            elif key in ("RESIZE", "OTHER", "ESC"):
                continue
            elif len(key) == 1 and key >= " ":
                inserted = [key]
                while key_available():
                    next_key = read_key()
                    if len(next_key) == 1 and next_key >= " ":
                        inserted.append(next_key)
                        continue
                    pending.append(next_key)
                    break
                chars[cursor:cursor] = inserted
                cursor += len(inserted)
            else:
                continue

            _render_editable_line(
                prompt,
                "".join(chars),
                cursor,
                text_style=text_style,
                write=write,
            )
    finally:
        sys.stdout.flush()

