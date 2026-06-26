"""Keyboard input helpers shared by interactive command screens."""

import atexit
import os
import re
import sys
import time
from collections import deque
from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

from rich.cells import cell_len, get_character_cell_size


MOUSE_CAPTURE_ENABLE = "\x1b[?1002l\x1b[?1003l\x1b[?1006h\x1b[?1000h\x1b[?1007l"
MOUSE_CAPTURE_DISABLE = "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1007h"
BRACKETED_PASTE_ENABLE = "\x1b[?2004h"
BRACKETED_PASTE_DISABLE = "\x1b[?2004l"
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
# A clipboard paste can reach the batcher in pieces -- Windows hands it to the
# console buffer chunk by chunk with sub-frame gaps. Once a burst is underway we
# wait this long for the next character before deciding the paste has ended, so a
# single paste stays one token instead of fragmenting into several lines (and
# several ``[Pasted text]`` markers). A lone keystroke never waits: the bridge
# only applies mid-burst.
_PASTE_BURST_GAP_SECONDS = 0.02
# When a burst hits a line break we have to decide: paste newline, or a typed
# line being submitted? A paste's next chunk can lag further than the burst gap
# (ConPTY scheduling), so we wait longer here before concluding "submit". This
# only ever delays an ENTER pressed within the burst gap of the previous char --
# a human pause before ENTER exits the batch loop before this point, so genuine
# submits stay instant.
_PASTE_RESUME_GAP_SECONDS = 0.1
# Hard cap on the *total* time a single batched read may block waiting for more
# of a chunked paste. The fallback batcher runs on the heads-up loop thread, so
# without a cap a large multi-line paste (each line break could wait a full
# resume gap) would freeze repaint/resize for seconds. Once this budget is spent
# the batch returns what it has; any remaining chunks are read on the next loop
# iteration -- so the screen repaints between chunks instead of stalling. Fast
# pastes never block at all (the next char is already available), so they still
# coalesce into one token. Bracketed pastes bypass this path entirely.
_PASTE_MAX_BLOCK_SECONDS = 0.1
_WINDOWS_ESCAPE_SEQUENCE_TIMEOUT = 0.03
_WINDOWS_CAPTURED_ESCAPE_SEQUENCE_TIMEOUT = 0.12
_LAST_TERMINAL_SIZE: tuple[int, int] | None = None
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SGR_MOUSE_TEXT_RE = re.compile(r"(?:\x1b)?\[<\d+;\d+;\d+[Mm]")
_MOUSE_CAPTURE_ACTIVE_DEPTH = 0
_WINDOWS_MOUSE_CAPTURE_DEPTH = 0
_WINDOWS_ENABLE_ECHO_INPUT = 0x0004
_WINDOWS_ENABLE_EXTENDED_FLAGS = 0x0080
_WINDOWS_ENABLE_LINE_INPUT = 0x0002
_WINDOWS_ENABLE_MOUSE_INPUT = 0x0010
_WINDOWS_ENABLE_QUICK_EDIT_MODE = 0x0040
_WINDOWS_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
_WINDOWS_ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
_WINDOWS_KEY_EVENT = 0x0001
_WINDOWS_MOUSE_EVENT = 0x0002
_WINDOWS_MOUSE_WHEELED = 0x0004
# Returned by the input readers when only non-key console events were pending
# (resize/focus/mouse-move). Mirrors the POSIX read returning ``None`` on a size
# change: the loop treats it as "no key this iteration" and repaints if needed.
_NO_ACTIONABLE_KEY = "RESIZE"
# Upper bound on characters folded into a single batched-text token. A paste can
# be tens of thousands of characters (Windows delivers it key-by-key, not as a
# bracketed-paste chunk), and it must coalesce into ONE token or it fragments
# into many; the bound only guards against a terminal that never stops reporting
# input as available.
_TEXT_BATCH_LIMIT = 100_000


class TextInput(str):
    """A queued group of printable characters from an editable input."""


@dataclass
class PasteRegistry:
    """Collapse bulky multi-line pastes into short inline placeholders.

    A paste that spans more than one line is replaced in the editable buffer
    with a ``[Pasted text #N +M lines]`` marker so the input stays readable,
    and the original text is restored by :meth:`expand` when the line is
    submitted. Single-line pastes are left untouched (``collapse`` returns
    ``None``) so short snippets stay visible and editable.

    The span helpers below let editors treat a marker as a single atomic token:
    one Backspace/Delete removes the whole placeholder (see
    :meth:`span_covering`), and pasting the same block again next to its marker
    can "unbox" it back to plain text (see :meth:`duplicate_span`).
    """

    _pastes: dict[str, str] = field(default_factory=dict)
    _count: int = 0

    def collapse(self, text: str) -> str | None:
        """Return a placeholder marker for a multi-line paste, else ``None``."""
        line_count = len(text.splitlines())
        if line_count < 2:
            return None
        self._count += 1
        marker = f"[Pasted text #{self._count} +{line_count} lines]"
        self._pastes[marker] = text
        return marker

    def expand(self, text: str) -> str:
        """Restore any placeholder markers in ``text`` to their pasted content."""
        for marker, original in self._pastes.items():
            if marker in text:
                text = text.replace(marker, original)
        return text

    def clear(self) -> None:
        """Forget every stored paste (call when the draft is sent or cleared)."""
        self._pastes.clear()
        self._count = 0

    def prune(self, text: str) -> None:
        """Forget markers that no longer appear in ``text`` (call after an edit)."""
        if not self._pastes:
            return
        self._pastes = {
            marker: original
            for marker, original in self._pastes.items()
            if marker in text
        }

    def marker_spans(self, text: str) -> list[tuple[int, int]]:
        """Return the ``(start, end)`` offsets of every known marker in ``text``."""
        spans: list[tuple[int, int]] = []
        for marker in self._pastes:
            start = text.find(marker)
            while start != -1:
                spans.append((start, start + len(marker)))
                start = text.find(marker, start + len(marker))
        spans.sort()
        return spans

    def span_covering(self, text: str, index: int) -> tuple[int, int] | None:
        """Return the span of the marker covering character ``index``, if any."""
        for start, end in self.marker_spans(text):
            if start <= index < end:
                return start, end
        return None

    def duplicate_span(
        self, text: str, cursor: int, content: str
    ) -> tuple[int, int] | None:
        """Return an adjacent marker's span when it already holds ``content``.

        "Adjacent" means the marker sits immediately before or after ``cursor``.
        Pasting the same block again next to its box uses this to unbox the
        placeholder rather than stacking a second, identical marker.
        """
        for marker, original in self._pastes.items():
            if original != content:
                continue
            width = len(marker)
            before = cursor - width
            if before >= 0 and text[before:cursor] == marker:
                return before, cursor
            if text[cursor : cursor + width] == marker:
                return cursor, cursor + width
        return None


def _delete_paste_aware(chars: list[str], pastes: PasteRegistry, index: int) -> int:
    """Delete the character at ``index`` -- or the whole marker covering it.

    Returns the new cursor position (the start of whatever was removed) so a
    single keypress erases a ``[Pasted text]`` placeholder atomically.
    """
    span = pastes.span_covering("".join(chars), index)
    if span is not None:
        start, end = span
        del chars[start:end]
        pastes.prune("".join(chars))
        return start
    del chars[index]
    return index


@contextmanager
def mouse_capture():
    """Capture terminal mouse input while a full-screen view is active."""
    global _MOUSE_CAPTURE_ACTIVE_DEPTH

    if not sys.stdout.isatty():
        with _windows_virtual_terminal_input():
            yield
        return

    with _windows_virtual_terminal_input():
        if _MOUSE_CAPTURE_ACTIVE_DEPTH == 0:
            disable_mouse_capture()
            sys.stdout.write(MOUSE_CAPTURE_ENABLE)
            sys.stdout.flush()
        _MOUSE_CAPTURE_ACTIVE_DEPTH += 1
        try:
            yield
        finally:
            _MOUSE_CAPTURE_ACTIVE_DEPTH = max(0, _MOUSE_CAPTURE_ACTIVE_DEPTH - 1)
            if _MOUSE_CAPTURE_ACTIVE_DEPTH == 0:
                disable_mouse_capture()


def disable_mouse_capture() -> None:
    """Best-effort reset for terminal mouse reporting modes."""
    if not sys.stdout.isatty():
        return
    sys.stdout.write(MOUSE_CAPTURE_DISABLE)
    sys.stdout.flush()


def enable_mouse_wheel_reporting() -> None:
    """Turn on SGR mouse reporting so the wheel arrives as MOUSE_WHEEL_* tokens.

    Writes the same DECSET enables as :func:`mouse_capture` (SGR ``1006h`` +
    button tracking ``1000h``, and crucially ``1007l`` to *disable* the terminal's
    alternate-scroll mode that would otherwise translate the wheel into arrow
    keys). Unlike :func:`mouse_capture` this only writes output escapes and touches
    no Windows console modes, so it composes with ``windows_vt_input`` -- letting a
    VT-input view (heads-up) read both bracketed pastes and SGR mouse wheel events
    from the same stream. Tear down with :func:`disable_mouse_capture`.
    """
    if not sys.stdout.isatty():
        return
    sys.stdout.write(MOUSE_CAPTURE_ENABLE)
    sys.stdout.flush()


def _restore_terminal_modes_atexit() -> None:
    """Backstop: undo terminal modes a hard exit could otherwise leak.

    The cursor-hide and bracketed-paste enables are written outside any single
    context manager (e.g. the editable-line renderer hides the cursor every
    keystroke), so an interrupt landing between an enable and its ``finally``
    could leave the user's shell with a hidden cursor or paste mode stuck on.
    Showing the cursor and disabling paste at exit is idempotent and safe.
    """
    if not sys.stdout.isatty():
        return
    try:
        sys.stdout.write(CURSOR_SHOW)
        sys.stdout.write(BRACKETED_PASTE_DISABLE)
        sys.stdout.flush()
    except Exception:
        pass


atexit.register(disable_mouse_capture)
atexit.register(_restore_terminal_modes_atexit)


@contextmanager
def bracketed_paste():
    """Ask terminals to wrap pasted text so editable views can preserve it."""
    if not sys.stdout.isatty():
        yield
        return
    sys.stdout.write(BRACKETED_PASTE_ENABLE)
    sys.stdout.flush()
    try:
        yield
    finally:
        sys.stdout.write(BRACKETED_PASTE_DISABLE)
        sys.stdout.flush()


def _stdin_isatty() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


@contextmanager
def raw_input_mode():
    r"""Hold the POSIX terminal in no-echo, non-canonical mode for a full-screen view.

    Full-screen views poll for input on the loop thread *between* repaints, so
    the terminal spends almost all of its time outside ``_read_key`` (which only
    sets raw mode for the duration of a single read). POSIX terminals start in
    cooked mode -- canonical line buffering plus echo -- so without this, every
    key typed between reads is echoed by the TTY driver at the cursor (flashing
    in the corner until the next repaint wipes it) and stays line-buffered out of
    reach of the non-blocking ``select`` read until Enter is pressed. Windows
    gets the same effect for free (``msvcrt.getwch`` never echoes and isn't line
    buffered), so this is a no-op there.

    The mode mirrors ``tty.setraw`` -- no echo, no canonical buffering, and no
    signal generation so Ctrl-C arrives as a ``\x03`` byte that the readers turn
    into ``KeyboardInterrupt`` -- but deliberately leaves output post-processing
    (``OPOST``) enabled, so the Live display's newlines still carry a carriage
    return and the frame doesn't stair-step down the screen.
    """
    if sys.platform == "win32" or not _stdin_isatty():
        yield
        return
    try:
        import termios
    except ImportError:
        yield
        return
    try:
        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
    except Exception:
        yield
        return
    try:
        mode = termios.tcgetattr(fd)
        mode[0] &= ~(
            termios.BRKINT | termios.ICRNL | termios.INPCK | termios.ISTRIP | termios.IXON
        )
        mode[2] &= ~(termios.CSIZE | termios.PARENB)
        mode[2] |= termios.CS8
        mode[3] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)
        mode[6][termios.VMIN] = 1
        mode[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSADRAIN, mode)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)
        except Exception:
            pass


@contextmanager
def windows_vt_input():
    """Enable VT input on Windows so pastes arrive as one bracketed-paste block.

    Without this, Windows hands console input over as plain characters and the
    terminal's ``\\x1b[?2004h`` paste markers never materialise, so a multi-line
    paste reaches the reader as raw chars (one ``\\r`` per line break) and has to
    be reassembled by the timing-based batcher. With VT input enabled the paste
    comes through wrapped in ``\\x1b[200~ ... \\x1b[201~`` and the existing decoder
    in :func:`_read_key` returns it atomically. No-op off win32 / without ctypes;
    only the keyboard layer is touched (mouse capture is handled separately).
    """
    if sys.platform != "win32":
        yield
        return

    try:
        import ctypes
    except ImportError:
        yield
        return

    try:
        kernel32 = ctypes.windll.kernel32
    except AttributeError:
        yield
        return

    handle = kernel32.GetStdHandle(-10)
    mode = ctypes.c_uint()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        yield
        return

    original_mode = mode.value
    enabled_mode = (
        original_mode | _WINDOWS_ENABLE_VIRTUAL_TERMINAL_INPUT
    ) & ~(
        _WINDOWS_ENABLE_LINE_INPUT | _WINDOWS_ENABLE_ECHO_INPUT
    )
    changed = enabled_mode != original_mode and bool(
        kernel32.SetConsoleMode(handle, enabled_mode)
    )
    try:
        yield
    finally:
        if changed:
            kernel32.SetConsoleMode(handle, original_mode)


@contextmanager
def windows_vt_input_suspended():
    r"""Temporarily clear VT input so a nested console view reads key records.

    The inverse of :func:`windows_vt_input`. A VT-input view (heads-up) holds the
    console in ``ENABLE_VIRTUAL_TERMINAL_INPUT`` for its whole lifetime, including
    while it suspends to run a nested full-screen view (an interactive slash
    command). Those nested views read keyboard input as console *records* and
    expect VT input *off* -- with it on, an arrow key or wheel scroll arrives as
    the raw byte run ``\x1b [ A``/``\x1b [ B``, whose leading ESC the view reads
    as a close key (so it exits) while the trailing ``[A``/``[B`` is left in the
    buffer to leak into the parent's input box on resume. Clearing the flag here
    for the duration of the nested view restores per-key console records; the
    original mode (VT input back on) is restored on exit. No-op off win32 /
    without ctypes / when VT input wasn't enabled.
    """
    if sys.platform != "win32":
        yield
        return

    try:
        import ctypes
    except ImportError:
        yield
        return

    try:
        kernel32 = ctypes.windll.kernel32
    except AttributeError:
        yield
        return

    handle = kernel32.GetStdHandle(-10)
    mode = ctypes.c_uint()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        yield
        return

    original_mode = mode.value
    cleared_mode = original_mode & ~_WINDOWS_ENABLE_VIRTUAL_TERMINAL_INPUT
    changed = cleared_mode != original_mode and bool(
        kernel32.SetConsoleMode(handle, cleared_mode)
    )
    try:
        yield
    finally:
        if changed:
            kernel32.SetConsoleMode(handle, original_mode)


@contextmanager
def _windows_virtual_terminal_input():
    global _WINDOWS_MOUSE_CAPTURE_DEPTH

    if sys.platform != "win32":
        yield
        return

    try:
        import ctypes
    except ImportError:
        yield
        return

    try:
        kernel32 = ctypes.windll.kernel32
    except AttributeError:
        yield
        return

    input_handle = kernel32.GetStdHandle(-10)
    output_handle = kernel32.GetStdHandle(-11)
    input_mode = ctypes.c_uint()
    output_mode = ctypes.c_uint()
    has_input_mode = bool(kernel32.GetConsoleMode(input_handle, ctypes.byref(input_mode)))
    has_output_mode = bool(kernel32.GetConsoleMode(output_handle, ctypes.byref(output_mode)))
    if not has_input_mode and not has_output_mode:
        yield
        return

    original_input_mode = input_mode.value
    original_output_mode = output_mode.value
    input_changed = False
    output_changed = False
    if has_input_mode:
        # Full-screen input must be unbuffered. Keep keyboard input as console
        # records; forcing VT input makes arrows look like ESC+[B in some
        # Windows terminals, so Esc can be mistaken for "exit".
        enabled_input_mode = (
            original_input_mode
            | _WINDOWS_ENABLE_MOUSE_INPUT
            | _WINDOWS_ENABLE_EXTENDED_FLAGS
        ) & ~(
            _WINDOWS_ENABLE_LINE_INPUT
            | _WINDOWS_ENABLE_ECHO_INPUT
        )
        input_changed = enabled_input_mode != original_input_mode and bool(
            kernel32.SetConsoleMode(input_handle, enabled_input_mode)
        )
    if has_output_mode:
        enabled_output_mode = original_output_mode | _WINDOWS_ENABLE_VIRTUAL_TERMINAL_PROCESSING
        output_changed = enabled_output_mode != original_output_mode and bool(
            kernel32.SetConsoleMode(output_handle, enabled_output_mode)
        )

    _WINDOWS_MOUSE_CAPTURE_DEPTH += 1
    try:
        yield
    finally:
        _WINDOWS_MOUSE_CAPTURE_DEPTH = max(0, _WINDOWS_MOUSE_CAPTURE_DEPTH - 1)
        if input_changed:
            kernel32.SetConsoleMode(input_handle, original_input_mode)
        if output_changed:
            kernel32.SetConsoleMode(output_handle, original_output_mode)


def _read_until_any(chars: set[str]) -> str:
    data = ""
    while True:
        ch = sys.stdin.read(1)
        if not ch:
            return data
        data += ch
        if ch in chars:
            return data


def _read_until_sequence(sequence: str) -> str:
    data = ""
    while True:
        ch = sys.stdin.read(1)
        if not ch:
            return data
        data += ch
        if data.endswith(sequence):
            return data[: -len(sequence)]


def _windows_key_available() -> bool:
    if sys.platform != "win32":
        return False
    if _WINDOWS_MOUSE_CAPTURE_DEPTH:
        # With mouse capture on, keys are read via ReadConsoleInputW, so judge
        # availability at that same Win32 layer (PeekConsoleInput) rather than
        # the CRT ``kbhit`` -- the two can disagree on focus/mouse/resize events.
        pending = _windows_actionable_key_pending()
        if pending is not None:
            return pending
        # Console-input API unusable: fall back to the CRT layer below.
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


def _read_windows_until_sequence(msvcrt, sequence: str) -> str:
    data = ""
    while True:
        ch = msvcrt.getwch()
        if not ch:
            return data
        data += ch
        if data.endswith(sequence):
            return data[: -len(sequence)]


def _read_windows_available_char(msvcrt, *, timeout: float = 0.0) -> str | None:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        if _windows_key_available():
            return msvcrt.getwch()
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.001)


def _windows_escape_sequence_timeout() -> float:
    if _WINDOWS_MOUSE_CAPTURE_DEPTH:
        return _WINDOWS_CAPTURED_ESCAPE_SEQUENCE_TIMEOUT
    return _WINDOWS_ESCAPE_SEQUENCE_TIMEOUT


def _windows_key_from_virtual_key(
    virtual_key: int,
    char: str,
    *,
    text_mode: bool,
) -> str | None:
    if char == "\r":
        return "ENTER"
    if char == "\n":
        return "CTRL_N"
    if char == "\t":
        return "TAB"
    if char == "\x1b":
        return "ESC"
    if not text_mode and char in ("q", "Q"):
        return "ESC"
    if char == "\x06":
        return "CTRL_F"
    if char == "\x13":
        return "CTRL_S"
    if char in ("\x08", "\x7f"):
        return "BACKSPACE"
    if char == "\x03":
        raise KeyboardInterrupt

    mapped = {
        0x26: "UP",
        0x28: "DOWN",
        0x25: "LEFT",
        0x27: "RIGHT",
        0x24: "HOME",
        0x23: "END",
        0x21: "PAGEUP",
        0x22: "PAGEDOWN",
        0x2E: "DELETE",
    }.get(virtual_key)
    if mapped is not None:
        return mapped
    if char and char != "\x00":
        return char
    return None


def _signed_high_word(value: int) -> int:
    high = (int(value) >> 16) & 0xFFFF
    return high - 0x10000 if high & 0x8000 else high


def _windows_console_input_reader():
    """Build the Win32 console-input access objects, or return ``None``.

    Returns ``(kernel32, handle, ctypes, wintypes, InputRecord)`` while a captured
    full-screen view is active on Windows and ctypes is available; otherwise
    ``None`` so callers fall back to the msvcrt path (or report "no key"). The
    record structs are rebuilt per call -- cheap, and it keeps the types bound to
    whatever ``ctypes`` is live (tests swap it), which is how the old inline reader
    behaved too.
    """
    if sys.platform != "win32" or not _WINDOWS_MOUSE_CAPTURE_DEPTH:
        return None

    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None

    try:
        kernel32 = ctypes.windll.kernel32
    except AttributeError:
        return None

    class Coord(ctypes.Structure):
        _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]

    class KeyEventRecord(ctypes.Structure):
        _fields_ = [
            ("bKeyDown", wintypes.BOOL),
            ("wRepeatCount", wintypes.WORD),
            ("wVirtualKeyCode", wintypes.WORD),
            ("wVirtualScanCode", wintypes.WORD),
            ("UnicodeChar", wintypes.WCHAR),
            ("dwControlKeyState", wintypes.DWORD),
        ]

    class MouseEventRecord(ctypes.Structure):
        _fields_ = [
            ("dwMousePosition", Coord),
            ("dwButtonState", wintypes.DWORD),
            ("dwControlKeyState", wintypes.DWORD),
            ("dwEventFlags", wintypes.DWORD),
        ]

    class EventUnion(ctypes.Union):
        _fields_ = [
            ("KeyEvent", KeyEventRecord),
            ("MouseEvent", MouseEventRecord),
        ]

    class InputRecord(ctypes.Structure):
        _anonymous_ = ("Event",)
        _fields_ = [
            ("EventType", wintypes.WORD),
            ("Event", EventUnion),
        ]

    handle = kernel32.GetStdHandle(-10)
    return kernel32, handle, ctypes, wintypes, InputRecord


def _windows_keydown_actionable(virtual_key: int, char: str) -> bool:
    """Whether a key-down record would yield a key for the loop to handle."""
    if char == "\x03":  # Ctrl-C: actionable (the read path raises KeyboardInterrupt)
        return True
    try:
        return (
            _windows_key_from_virtual_key(virtual_key, char, text_mode=True)
            is not None
        )
    except KeyboardInterrupt:  # pragma: no cover - guarded by the \x03 check above
        return True


def _windows_record_is_actionable(record) -> bool:
    """True only for a key-down or mouse-wheel record (something to act on)."""
    if record.EventType == _WINDOWS_KEY_EVENT:
        key = record.KeyEvent
        if not key.bKeyDown:
            return False
        return _windows_keydown_actionable(int(key.wVirtualKeyCode), key.UnicodeChar)
    if record.EventType == _WINDOWS_MOUSE_EVENT:
        mouse = record.MouseEvent
        if int(mouse.dwEventFlags) != _WINDOWS_MOUSE_WHEELED:
            return False
        return _signed_high_word(int(mouse.dwButtonState)) != 0
    return False


def _windows_actionable_key_pending() -> bool | None:
    """Report whether a key/wheel waits, draining non-actionable records first.

    Peeks the head of the Win32 console input buffer one record at a time. Returns
    ``True`` as soon as a key-down or mouse-wheel record is at the head, ``False``
    when only non-actionable records (key-up, mouse-move, FOCUS/MENU/
    WINDOW_BUFFER_SIZE) are queued -- those are consumed, since they would
    otherwise make the captured read block waiting for a real key -- and ``None``
    when the console-input API is unusable (e.g. redirected stdin), so callers can
    fall back to the CRT ``kbhit`` / ``getwch`` path.
    """
    reader = _windows_console_input_reader()
    if reader is None:
        return None
    kernel32, handle, ctypes, wintypes, InputRecord = reader
    record = InputRecord()
    read = wintypes.DWORD()
    while True:
        if not kernel32.PeekConsoleInputW(handle, ctypes.byref(record), 1, ctypes.byref(read)):
            return None
        if not read.value:
            return False
        if _windows_record_is_actionable(record):
            return True
        # Non-actionable head record: consume it and look at the next one.
        if not kernel32.ReadConsoleInputW(handle, ctypes.byref(record), 1, ctypes.byref(read)):
            return None
        if not read.value:
            return False


def _read_windows_console_input_key(
    *,
    text_mode: bool,
    translate_mouse_wheel: bool,
) -> str | None:
    reader = _windows_console_input_reader()
    if reader is None:
        return None
    kernel32, handle, ctypes, wintypes, InputRecord = reader

    record = InputRecord()
    read = wintypes.DWORD()
    while True:
        # Never block: only read once an actionable key/wheel is at the head.
        pending = _windows_actionable_key_pending()
        if pending is None:
            return None  # console API unusable -> fall back to msvcrt getwch
        if not pending:
            # Nothing actionable -> hand back the resize sentinel so the loop keeps
            # spinning and repaints (mirrors the POSIX read returning ``None``).
            return _NO_ACTIONABLE_KEY
        if not kernel32.ReadConsoleInputW(handle, ctypes.byref(record), 1, ctypes.byref(read)):
            return None
        if not read.value:
            return _NO_ACTIONABLE_KEY

        if record.EventType == _WINDOWS_KEY_EVENT:
            key = record.KeyEvent
            if not key.bKeyDown:
                continue
            token = _windows_key_from_virtual_key(
                int(key.wVirtualKeyCode),
                key.UnicodeChar,
                text_mode=text_mode,
            )
            if token is None:
                continue
            for _ in range(max(0, int(key.wRepeatCount) - 1)):
                _PENDING_KEYS.append(token)
            return token

        if record.EventType == _WINDOWS_MOUSE_EVENT:
            mouse = record.MouseEvent
            if int(mouse.dwEventFlags) != _WINDOWS_MOUSE_WHEELED:
                continue
            delta = _signed_high_word(int(mouse.dwButtonState))
            if delta == 0:
                continue
            if translate_mouse_wheel:
                return "UP" if delta > 0 else "DOWN"
            return "MOUSE_WHEEL_UP" if delta > 0 else "MOUSE_WHEEL_DOWN"


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


# Final byte -> token for the common letter-terminated CSI navigation keys
# (``ESC [ <params> <letter>``). The ``~``-terminated variants key off the first
# numeric parameter instead.
_CSI_LETTER_TOKENS = {
    "A": "UP",
    "B": "DOWN",
    "C": "RIGHT",
    "D": "LEFT",
    "H": "HOME",
    "F": "END",
}
_CSI_TILDE_TOKENS = {
    "1": "HOME",
    "3": "DELETE",
    "4": "END",
    "5": "PAGEUP",
    "6": "PAGEDOWN",
    "7": "HOME",
    "8": "END",
}


def _csi_token(params: str, final: str) -> str:
    """Map a parsed CSI sequence (``params`` plus terminating ``final``) to a token.

    Handles xterm-style *modified* navigation keys of the form
    ``ESC [ 1 ; <mod> X`` (and the ``~``-terminated ``ESC [ N ; <mod> ~``
    variants). The modifier lives in the second parameter and encodes
    ``Shift``/``Alt``/``Ctrl`` as ``(mod - 1)`` bit flags. For Left/Right we fold
    Ctrl and Shift into ``CTRL_``/``SHIFT_`` prefixes so the editable input can
    map them to word-wise motion and text selection; every other key ignores the
    modifier and returns its plain token (so e.g. Ctrl+Home is still HOME).
    """
    parts = params.split(";")
    code = parts[0]
    modifier = 1
    if len(parts) >= 2 and parts[1].isdigit():
        modifier = int(parts[1])
    if final == "~":
        base = _CSI_TILDE_TOKENS.get(code, "OTHER")
    else:
        base = _CSI_LETTER_TOKENS.get(final, "OTHER")
    if base in ("LEFT", "RIGHT") and modifier >= 2:
        bits = modifier - 1
        prefix = ("CTRL_" if bits & 4 else "") + ("SHIFT_" if bits & 1 else "")
        if prefix:
            return prefix + base
    return base


def _read_posix_csi_tail(first: str) -> tuple[str, str]:
    """Read a CSI parameter/final tail on POSIX after its first byte.

    ``first`` is the already-read first parameter byte (e.g. ``"1"`` for a
    modified arrow ``ESC [ 1 ; 5 D``). Returns ``(params, final)`` where ``final``
    is the terminating byte (``""`` if the stream ran dry first).
    """
    params = first
    while True:
        nxt = sys.stdin.read(1)
        if not nxt:
            return params, ""
        if nxt.isdigit() or nxt == ";":
            params += nxt
            continue
        return params, nxt


def _read_windows_csi_tail(msvcrt, first: str, timeout: float) -> tuple[str, str]:
    """Windows counterpart of :func:`_read_posix_csi_tail` using the timed reader."""
    params = first
    while True:
        nxt = _read_windows_available_char(msvcrt, timeout=timeout)
        if nxt is None:
            return params, ""
        if nxt.isdigit() or nxt == ";":
            params += nxt
            continue
        return params, nxt


def _terminal_size() -> tuple[int, int] | None:
    for stream in (1, 2, 0):
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
            if _WINDOWS_MOUSE_CAPTURE_DEPTH:
                pending = _windows_actionable_key_pending()
                if pending is not None:
                    return pending
            import msvcrt
            return bool(msvcrt.kbhit())
        import select
        return bool(select.select([sys.stdin], [], [], 0)[0])
    except (ImportError, OSError, TypeError, ValueError):
        return False


def requeue_key(key: str) -> None:
    """Return a key read out-of-band to the foreground input loop."""
    _PENDING_KEYS.appendleft(key)


def strip_sgr_mouse_sequences(text: str) -> str:
    """Remove printable SGR mouse reports that leaked past ESC handling."""
    return _SGR_MOUSE_TEXT_RE.sub("", text)


def _is_batched_text_key(key: str) -> bool:
    return (
        isinstance(key, str)
        and len(key) == 1
        and (key.isprintable() or key == "\t")
    )


def _await_more_input(
    in_burst: bool,
    *,
    timeout: float = _PASTE_BURST_GAP_SECONDS,
    deadline: float | None = None,
) -> bool:
    """Return whether another character is (or becomes) available to batch.

    Reports immediately when input is already waiting. Otherwise, only while a
    burst is in progress, it polls for up to ``timeout`` seconds so a chunked
    paste's next piece can arrive before the batch is closed -- bridging the
    sub-frame gaps that would otherwise split one paste into several tokens. A
    longer ``timeout`` is used at a line break, where a premature decision would
    wrongly submit a paste's first line.

    ``deadline`` (an absolute ``time.monotonic`` instant) caps the wait so the
    caller can bound the *total* time a batch blocks the loop: once it passes,
    this only reports input that is already available and never sleeps.
    """
    if _key_available():
        return True
    if not in_burst:
        return False
    wait_until = time.monotonic() + timeout
    if deadline is not None:
        wait_until = min(wait_until, deadline)
    while time.monotonic() < wait_until:
        if _key_available():
            return True
        time.sleep(0.001)
    return False


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
        and _is_batched_text_key(key)
    ):
        # Coalesce a burst of printable characters (a paste, or held/queued
        # typing) into one TextInput so the view redraws once. Windows hands a
        # paste to a console-record reader key-by-key -- including a carriage
        # return per line break -- so newlines are folded into the token rather
        # than leaking out as separate ENTER submits. A newline only ends the
        # burst as a submit when the burst is a single typed line; a multi-line
        # paste keeps its content (trailing newline dropped) and stays in the
        # editor so the user can keep editing, mirroring bracketed paste.
        inserted = [key]
        # Bound the total blocking of this batch so the loop thread can't be
        # starved by a slowly-arriving paste (see _PASTE_MAX_BLOCK_SECONDS).
        block_deadline = time.monotonic() + _PASTE_MAX_BLOCK_SECONDS
        while len(inserted) < _TEXT_BATCH_LIMIT and _await_more_input(
            len(inserted) >= 2, deadline=block_deadline
        ):
            next_key = read_key()
            if _is_batched_text_key(next_key):
                inserted.append(next_key)
                continue
            if next_key == "ENTER":
                if _await_more_input(
                    len(inserted) >= 2,
                    timeout=_PASTE_RESUME_GAP_SECONDS,
                    deadline=block_deadline,
                ):
                    inserted.append("\n")
                    continue
                if "\n" not in inserted:
                    # Single typed line: hand ENTER back so the caller submits.
                    _PENDING_KEYS.appendleft("ENTER")
                break
            _PENDING_KEYS.appendleft(next_key)
            break
        text = strip_sgr_mouse_sequences("".join(inserted))
        if not text:
            return "OTHER", 1
        return TextInput(text), 1

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
    ENTER, ESC, TAB, CTRL_F, CTRL_N, CTRL_S, BACKSPACE, DELETE, the modified
    arrows CTRL_LEFT/CTRL_RIGHT (word-wise) and SHIFT_LEFT/SHIFT_RIGHT/
    CTRL_SHIFT_LEFT/CTRL_SHIFT_RIGHT (selection), or the raw character. Raises
    KeyboardInterrupt on Ctrl-C.  When ``text_mode`` is True, the convenience q/Q → ESC mapping is
    disabled so a search query can include those letters. When
    ``translate_mouse_wheel`` is False, SGR wheel input returns MOUSE_WHEEL_*
    tokens instead of arrow/page navigation tokens.
    """
    if _PENDING_KEYS:
        return _PENDING_KEYS.popleft()

    if sys.platform == "win32":
        captured_key = _read_windows_console_input_key(
            text_mode=text_mode,
            translate_mouse_wheel=translate_mouse_wheel,
        )
        if captured_key is not None:
            return captured_key

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
        if ch == "\n":
            return "CTRL_N"
        if ch == "\t":
            return "TAB"
        if ch == "\x1b":
            sequence_timeout = _windows_escape_sequence_timeout()
            ch2 = _read_windows_available_char(
                msvcrt,
                timeout=sequence_timeout,
            )
            if ch2 is not None:
                if ch2 == "[":
                    ch3 = _read_windows_available_char(
                        msvcrt,
                        timeout=sequence_timeout,
                    )
                    if ch3 is None:
                        _PENDING_KEYS.appendleft("[")
                        return "ESC"
                    if ch3 == "<":
                        return _parse_sgr_mouse(
                            _read_windows_until_any(msvcrt, {"M", "m"}),
                            translate_wheel=translate_mouse_wheel,
                        )
                    if ch3 == "2":
                        sequence = ch3 + _read_windows_until_any(msvcrt, {"~"})
                        if sequence == "200~":
                            return TextInput(
                                strip_sgr_mouse_sequences(
                                    _read_windows_until_sequence(msvcrt, "\x1b[201~")
                                )
                            )
                        return "OTHER"
                    if ch3 == "1":
                        # Modified navigation key: ESC [ 1 ; <mod> <final>
                        # (Ctrl/Shift + arrow). Read the rest of the sequence so
                        # the modifier byte and final letter aren't left to leak
                        # back as literal characters.
                        params, final = _read_windows_csi_tail(
                            msvcrt, ch3, sequence_timeout
                        )
                        return _csi_token(params, final)
                    if ch3 in ("5", "6", "3") and _windows_key_available():
                        msvcrt.getwch()  # consume trailing ~
                    return {
                        "A": "UP", "B": "DOWN", "D": "LEFT", "C": "RIGHT",
                        "H": "HOME", "F": "END",
                        "5": "PAGEUP", "6": "PAGEDOWN", "3": "DELETE",
                    }.get(ch3, "OTHER")
                _PENDING_KEYS.appendleft(ch2)
            return "ESC"
        if ch == "[":
            ch2 = _read_windows_available_char(
                msvcrt,
                timeout=_WINDOWS_ESCAPE_SEQUENCE_TIMEOUT,
            )
            if ch2 == "<":
                return _parse_sgr_mouse(
                    _read_windows_until_any(msvcrt, {"M", "m"}),
                    translate_wheel=translate_mouse_wheel,
                )
            if ch2 is not None:
                _PENDING_KEYS.appendleft(ch2)
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
            # TCSANOW, not setraw's default TCSAFLUSH: the poll loop calls this
            # only once a key is already queued (``_key_available`` saw it), and
            # TCSAFLUSH would discard that unread byte before the read, dropping
            # the keystroke. TCSANOW applies raw mode without flushing input.
            tty.setraw(fd, termios.TCSANOW)
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
                    if ch3 == "2":
                        sequence = ch3 + _read_until_any({"~"})
                        if sequence == "200~":
                            return TextInput(
                                strip_sgr_mouse_sequences(
                                    _read_until_sequence("\x1b[201~")
                                )
                            )
                        return "OTHER"
                    if ch3 == "1":
                        # Modified navigation key: ESC [ 1 ; <mod> <final>
                        # (Ctrl/Shift + arrow). Consume the full sequence so its
                        # modifier byte and final letter don't leak as raw input.
                        params, final = _read_posix_csi_tail("1")
                        return _csi_token(params, final)
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
            if ch == "\r":
                return "ENTER"
            if ch == "\n":
                return "CTRL_N"
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
    """Read one editable line with cross-platform raw key handling.

    Multi-line pastes are collapsed to a ``[Pasted text #N +M lines]`` marker
    (this view is a single line) and restored when the line is submitted.
    """
    manage_terminal = read_key is None
    if read_key is None:
        # Batch a paste into one TextInput. POSIX delivers it as a bracketed
        # chunk; Windows delivers it key-by-key, so the batcher folds the
        # newlines together instead of letting the first one submit the line.
        read_key = lambda: _read_key_with_repeats(text_mode=True, batch_text=True)[0]
        key_available = key_available or _key_available
    else:
        key_available = key_available or (lambda: False)
    write = write or sys.stdout.write
    pastes = PasteRegistry()
    chars = list(initial.replace("\r", " ").replace("\n", " "))
    cursor = len(chars)
    pending: deque[str] = deque()

    def insert(text: str) -> None:
        nonlocal cursor
        printable = [ch for ch in text if ch >= " "]
        chars[cursor:cursor] = printable
        cursor += len(printable)

    # On POSIX, ask the terminal to wrap pastes so they arrive as one chunk.
    # Windows isn't put in VT-input mode here, so it can't emit the markers (and
    # the enable sequence could echo as raw text); the batching reader coalesces
    # the key-by-key paste there instead.
    enable_paste_wrap = manage_terminal and sys.platform != "win32"
    paste_terminal = bracketed_paste() if enable_paste_wrap else nullcontext()
    with paste_terminal:
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
                        pastes.clear()
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
                    return pastes.expand("".join(chars))
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
                        cursor = _delete_paste_aware(chars, pastes, cursor - 1)
                elif key == "DELETE":
                    if cursor < len(chars):
                        cursor = _delete_paste_aware(chars, pastes, cursor)
                elif key == "\x04":
                    if not chars:
                        raise EOFError
                    if cursor < len(chars):
                        del chars[cursor]
                elif key in ("RESIZE", "OTHER", "ESC"):
                    continue
                elif isinstance(key, TextInput):
                    text = strip_sgr_mouse_sequences(str(key))
                    span = pastes.duplicate_span("".join(chars), cursor, text)
                    if span is not None:
                        # The same block pasted again next to its box: unbox it
                        # to one plain (flattened) copy instead of a second box.
                        start, end = span
                        flattened = [ch for ch in text.replace("\n", " ") if ch >= " "]
                        chars[start:end] = flattened
                        cursor = start + len(flattened)
                        pastes.prune("".join(chars))
                    else:
                        marker = pastes.collapse(text)
                        insert(marker if marker is not None else text.replace("\n", " "))
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

