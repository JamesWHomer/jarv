import sys
from collections import deque
from types import SimpleNamespace

from jarv import command_input


class FakeStdin:
    def __init__(self, text: str):
        self._chars = list(text)

    def fileno(self):
        return 0

    def read(self, count: int):
        if count != 1 or not self._chars:
            return ""
        return self._chars.pop(0)

    @property
    def remaining(self):
        return "".join(self._chars)


def _install_posix_input(monkeypatch, text: str) -> FakeStdin:
    stdin = FakeStdin(text)

    def no_terminal_size(_fd):
        raise OSError

    command_input._LAST_TERMINAL_SIZE = None
    monkeypatch.setattr(command_input.sys, "platform", "linux")
    monkeypatch.setattr(command_input.sys, "stdin", stdin)
    monkeypatch.setattr(command_input.os, "get_terminal_size", no_terminal_size)
    monkeypatch.setitem(
        sys.modules,
        "select",
        SimpleNamespace(select=lambda read, _write, _error, _timeout: (read if stdin.remaining else [], [], [])),
    )
    monkeypatch.setitem(
        sys.modules, "tty", SimpleNamespace(setraw=lambda _fd, _when=None: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "termios",
        SimpleNamespace(
            TCSADRAIN=1,
            TCSANOW=0,
            tcgetattr=lambda _fd: "old",
            tcsetattr=lambda _fd, _when, _old: None,
        ),
    )
    return stdin


class FakeStdout:
    def __init__(self, is_tty=True):
        self.writes = []
        self.is_tty = is_tty

    def isatty(self):
        return self.is_tty

    def write(self, text):
        self.writes.append(text)

    def flush(self):
        pass


class _FakeTermios:
    """Minimal termios stand-in recording attribute round-trips."""

    # Flag bit masks (arbitrary but distinct powers of two).
    BRKINT = 0x01
    ICRNL = 0x02
    INPCK = 0x04
    ISTRIP = 0x08
    IXON = 0x10
    CSIZE = 0x20
    PARENB = 0x40
    CS8 = 0x80
    ECHO = 0x100
    ICANON = 0x200
    IEXTEN = 0x400
    ISIG = 0x800
    VMIN = 6
    VTIME = 5
    TCSADRAIN = 1

    def __init__(self, mode):
        self.mode = mode
        self.applied: list[list] = []
        self.restored_to = None

    def tcgetattr(self, _fd):
        # Return a deep-ish copy so the caller's edits don't mutate our baseline.
        return [list(part) if isinstance(part, list) else part for part in self.mode]

    def tcsetattr(self, _fd, _when, mode):
        self.applied.append(mode)
        self.restored_to = mode


def _cooked_mode():
    iflag = _FakeTermios.BRKINT | _FakeTermios.ICRNL | _FakeTermios.IXON
    lflag = _FakeTermios.ECHO | _FakeTermios.ICANON | _FakeTermios.ISIG
    cc = [0] * 8
    return [iflag, 0, 0, lflag, 0, 0, cc]


def test_raw_input_mode_disables_echo_and_canonical_then_restores(monkeypatch):
    baseline = _cooked_mode()
    fake = _FakeTermios(baseline)
    monkeypatch.setattr(command_input.sys, "platform", "linux")
    monkeypatch.setattr(command_input.sys, "stdin", FakeStdin(""))
    monkeypatch.setattr(command_input, "_stdin_isatty", lambda: True)
    monkeypatch.setitem(sys.modules, "termios", fake)

    with command_input.raw_input_mode():
        applied = fake.applied[-1]
        # Echo, canonical and signal generation are off inside the block...
        assert not applied[3] & _FakeTermios.ECHO
        assert not applied[3] & _FakeTermios.ICANON
        assert not applied[3] & _FakeTermios.ISIG
        # ...but output post-processing (OFLAG, index 1) is left untouched so the
        # Live frame doesn't stair-step.
        assert applied[1] == baseline[1]
        assert applied[6][_FakeTermios.VMIN] == 1

    # On exit the original cooked attributes are restored verbatim.
    assert fake.restored_to == baseline


def test_raw_input_mode_is_noop_on_windows(monkeypatch):
    monkeypatch.setattr(command_input.sys, "platform", "win32")
    fake = _FakeTermios(_cooked_mode())
    monkeypatch.setitem(sys.modules, "termios", fake)

    with command_input.raw_input_mode():
        pass

    assert fake.applied == []


def test_raw_input_mode_is_noop_without_a_tty(monkeypatch):
    monkeypatch.setattr(command_input.sys, "platform", "linux")
    monkeypatch.setattr(command_input, "_stdin_isatty", lambda: False)
    fake = _FakeTermios(_cooked_mode())
    monkeypatch.setitem(sys.modules, "termios", fake)

    with command_input.raw_input_mode():
        pass

    assert fake.applied == []


def test_read_key_maps_sgr_mouse_wheel_up_to_up(monkeypatch):
    stdin = _install_posix_input(monkeypatch, "\x1b[<64;1;1M")

    assert command_input._read_key() == "UP"
    assert stdin.remaining == ""


def test_read_key_maps_sgr_mouse_wheel_down_to_down(monkeypatch):
    stdin = _install_posix_input(monkeypatch, "\x1b[<65;1;1M")

    assert command_input._read_key(text_mode=True) == "DOWN"
    assert stdin.remaining == ""


def test_read_key_maps_page_like_sgr_wheel_variants(monkeypatch):
    stdin = _install_posix_input(monkeypatch, "\x1b[<66;1;1M\x1b[<67;1;1M")

    assert command_input._read_key() == "PAGEUP"
    assert command_input._read_key() == "PAGEDOWN"
    assert stdin.remaining == ""


def test_read_key_can_preserve_sgr_mouse_wheel_tokens(monkeypatch):
    stdin = _install_posix_input(monkeypatch, "\x1b[<64;1;1M\x1b[<65;1;1M")

    assert command_input._read_key(translate_mouse_wheel=False) == "MOUSE_WHEEL_UP"
    assert command_input._read_key(translate_mouse_wheel=False) == "MOUSE_WHEEL_DOWN"
    assert stdin.remaining == ""


def test_mouse_capture_enables_windows_vt_and_mouse_modes(monkeypatch):
    stdout = FakeStdout()
    set_modes = []

    class FakeUInt:
        def __init__(self, value=0):
            self.value = value

    class FakeKernel32:
        def GetStdHandle(self, handle):
            return handle

        def GetConsoleMode(self, handle, mode):
            mode.value = 0x0047 if handle == -10 else 0
            return True

        def SetConsoleMode(self, handle, mode):
            set_modes.append((handle, mode))
            return True

    fake_ctypes = SimpleNamespace(
        c_uint=FakeUInt,
        byref=lambda value: value,
        windll=SimpleNamespace(kernel32=FakeKernel32()),
    )

    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.setattr(command_input.sys, "stdout", stdout)
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    command_input._MOUSE_CAPTURE_ACTIVE_DEPTH = 0
    command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0

    with command_input.mouse_capture():
        assert command_input._MOUSE_CAPTURE_ACTIVE_DEPTH == 1
        assert command_input._WINDOWS_MOUSE_CAPTURE_DEPTH == 1

    assert stdout.writes == [
        command_input.MOUSE_CAPTURE_DISABLE,
        command_input.MOUSE_CAPTURE_ENABLE,
        command_input.MOUSE_CAPTURE_DISABLE,
    ]
    input_mode = set_modes[0][1]
    output_mode = set_modes[1][1]
    assert set_modes[-2:] == [(-10, 0x0047), (-11, 0)]
    assert input_mode & 0x0010
    assert input_mode & 0x0080
    assert not input_mode & 0x0200
    assert input_mode & 0x0001
    assert not input_mode & 0x0002
    assert not input_mode & 0x0004
    assert input_mode & 0x0040
    assert output_mode & 0x0004
    assert command_input._MOUSE_CAPTURE_ACTIVE_DEPTH == 0
    assert command_input._WINDOWS_MOUSE_CAPTURE_DEPTH == 0


def test_mouse_capture_keeps_windows_input_mode_when_stdout_is_not_tty(monkeypatch):
    stdout = FakeStdout(is_tty=False)
    set_modes = []

    class FakeUInt:
        def __init__(self, value=0):
            self.value = value

    class FakeKernel32:
        def GetStdHandle(self, handle):
            return handle

        def GetConsoleMode(self, handle, mode):
            mode.value = 0x0047 if handle == -10 else 0
            return True

        def SetConsoleMode(self, handle, mode):
            set_modes.append((handle, mode))
            return True

    fake_ctypes = SimpleNamespace(
        c_uint=FakeUInt,
        byref=lambda value: value,
        windll=SimpleNamespace(kernel32=FakeKernel32()),
    )

    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.setattr(command_input.sys, "stdout", stdout)
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    command_input._MOUSE_CAPTURE_ACTIVE_DEPTH = 0
    command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0

    with command_input.mouse_capture():
        assert command_input._MOUSE_CAPTURE_ACTIVE_DEPTH == 0
        assert command_input._WINDOWS_MOUSE_CAPTURE_DEPTH == 1

    assert stdout.writes == []
    input_mode = set_modes[0][1]
    assert not input_mode & 0x0200
    assert not input_mode & 0x0002
    assert not input_mode & 0x0004
    assert set_modes[-2:] == [(-10, 0x0047), (-11, 0)]
    assert command_input._WINDOWS_MOUSE_CAPTURE_DEPTH == 0


def test_mouse_capture_disables_mouse_motion_tracking():
    assert "\x1b[?1002l" in command_input.MOUSE_CAPTURE_ENABLE
    assert "\x1b[?1003l" in command_input.MOUSE_CAPTURE_ENABLE
    assert "\x1b[?1002l" in command_input.MOUSE_CAPTURE_DISABLE
    assert "\x1b[?1003l" in command_input.MOUSE_CAPTURE_DISABLE


class FakeRichFileProxy:
    """Mimic ``rich.file_proxy.FileProxy``: an active Live installs this over
    ``sys.stdout``. It exposes the wrapped file as ``rich_proxied_file`` and
    swallows raw control writes into a line buffer (they never reach the
    terminal), which is exactly why the mode writers must unwrap it."""

    def __init__(self, file):
        self.rich_proxied_file = file
        self.writes = []

    def write(self, text):
        self.writes.append(text)

    def flush(self):
        pass

    def isatty(self):
        return self.rich_proxied_file.isatty()


def test_mouse_mode_writers_bypass_rich_stdout_proxy(monkeypatch):
    real = FakeStdout(is_tty=True)
    proxy = FakeRichFileProxy(real)
    monkeypatch.setattr(command_input.sys, "stdout", proxy)

    command_input.enable_mouse_wheel_reporting()
    command_input.disable_mouse_capture()

    assert proxy.writes == []
    assert real.writes == [
        command_input.MOUSE_CAPTURE_ENABLE,
        command_input.MOUSE_CAPTURE_DISABLE,
    ]


def test_bracketed_paste_bypasses_rich_stdout_proxy(monkeypatch):
    real = FakeStdout(is_tty=True)
    proxy = FakeRichFileProxy(real)
    monkeypatch.setattr(command_input.sys, "stdout", proxy)

    with command_input.bracketed_paste():
        pass

    assert proxy.writes == []
    assert real.writes == [
        command_input.BRACKETED_PASTE_ENABLE,
        command_input.BRACKETED_PASTE_DISABLE,
    ]


def test_windows_console_mouse_wheel_record_returns_mouse_token(monkeypatch):
    import ctypes

    class FakeKernel32:
        def GetStdHandle(self, handle):
            return handle

        def _fill(self, record, read):
            target = record._obj
            target.EventType = 0x0002
            target.MouseEvent.dwEventFlags = 0x0004
            target.MouseEvent.dwButtonState = 120 << 16
            read._obj.value = 1
            return True

        def PeekConsoleInputW(self, _handle, record, _count, read):
            return self._fill(record, read)

        def ReadConsoleInputW(self, _handle, record, _count, read):
            return self._fill(record, read)

    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=FakeKernel32()), raising=False)
    command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 1

    try:
        assert (
            command_input._read_windows_console_input_key(
                text_mode=True,
                translate_mouse_wheel=False,
            )
            == "MOUSE_WHEEL_UP"
        )
    finally:
        command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0


def test_windows_terminal_console_key_record_returns_arrow(monkeypatch):
    import ctypes

    class FakeKernel32:
        def GetStdHandle(self, handle):
            return handle

        def _fill(self, record, read):
            target = record._obj
            target.EventType = 0x0001
            target.KeyEvent.bKeyDown = True
            target.KeyEvent.wRepeatCount = 1
            target.KeyEvent.wVirtualKeyCode = 0x28
            target.KeyEvent.UnicodeChar = "\x00"
            read._obj.value = 1
            return True

        def PeekConsoleInputW(self, _handle, record, _count, read):
            return self._fill(record, read)

        def ReadConsoleInputW(self, _handle, record, _count, read):
            return self._fill(record, read)

    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.setenv("WT_SESSION", "test")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=FakeKernel32()), raising=False)
    command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 1

    try:
        assert (
            command_input._read_windows_console_input_key(
                text_mode=False,
                translate_mouse_wheel=True,
            )
            == "DOWN"
        )
    finally:
        command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0


def _make_console_records_kernel32(records):
    """FakeKernel32 backed by a queue of record dicts.

    ``PeekConsoleInputW`` reports the head without consuming it; ``ReadConsoleInputW``
    pops it. Both report ``0`` events read when the queue is empty (never blocking),
    mirroring the real Win32 calls the freeze fix relies on.
    """
    pending = deque(records)

    def _apply(target, rec):
        target.EventType = rec["type"]
        if rec["type"] == 0x0001:
            target.KeyEvent.bKeyDown = rec.get("down", True)
            target.KeyEvent.wRepeatCount = rec.get("repeat", 1)
            target.KeyEvent.wVirtualKeyCode = rec.get("vk", 0)
            target.KeyEvent.UnicodeChar = rec.get("char", "\x00")
        elif rec["type"] == 0x0002:
            target.MouseEvent.dwEventFlags = rec.get("flags", 0)
            target.MouseEvent.dwButtonState = rec.get("button", 0)

    class FakeKernel32:
        def GetStdHandle(self, handle):
            return handle

        def PeekConsoleInputW(self, _handle, record, _count, read):
            if not pending:
                read._obj.value = 0
                return True
            _apply(record._obj, pending[0])
            read._obj.value = 1
            return True

        def ReadConsoleInputW(self, _handle, record, _count, read):
            if not pending:
                read._obj.value = 0
                return True
            _apply(record._obj, pending.popleft())
            read._obj.value = 1
            return True

    return FakeKernel32(), pending


# Win32 console EventType / mouse-flag values used to script the records above.
_FOCUS_RECORD = {"type": 0x0010}
_MOUSE_MOVE_RECORD = {"type": 0x0002, "flags": 0x0001, "button": 0}
_KEY_DOWN_A = {"type": 0x0001, "down": True, "vk": 0x41, "char": "a"}


def test_windows_actionable_pending_drains_non_keys_then_reads_key(monkeypatch):
    import ctypes

    kernel32, pending = _make_console_records_kernel32(
        [_FOCUS_RECORD, _MOUSE_MOVE_RECORD, _KEY_DOWN_A]
    )
    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False)
    command_input._PENDING_KEYS.clear()
    command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 1

    try:
        # Focus + mouse-move are drained; the key-down stays queued at the head.
        assert command_input._windows_actionable_key_pending() is True
        assert len(pending) == 1
        # The read consumes only the key and returns it without ever blocking.
        assert (
            command_input._read_windows_console_input_key(
                text_mode=True,
                translate_mouse_wheel=False,
            )
            == "a"
        )
        assert not pending
    finally:
        command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0
        command_input._PENDING_KEYS.clear()


def test_windows_read_returns_sentinel_when_only_non_key_events(monkeypatch):
    import ctypes

    kernel32, pending = _make_console_records_kernel32([_FOCUS_RECORD])
    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False)
    command_input._PENDING_KEYS.clear()
    command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 1

    try:
        assert command_input._windows_actionable_key_pending() is False
        assert not pending  # focus record drained, buffer left clean
        # Nothing actionable pending -> the read returns the resize sentinel
        # instead of blocking on the empty console buffer.
        assert (
            command_input._read_windows_console_input_key(
                text_mode=True,
                translate_mouse_wheel=False,
            )
            == command_input._NO_ACTIONABLE_KEY
        )
    finally:
        command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0
        command_input._PENDING_KEYS.clear()


def test_key_available_peeks_when_mouse_capture_active(monkeypatch):
    import ctypes

    kernel32, pending = _make_console_records_kernel32([_FOCUS_RECORD, _KEY_DOWN_A])
    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False)
    command_input._PENDING_KEYS.clear()
    command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 1

    try:
        assert command_input._key_available() is True
        assert len(pending) == 1  # focus drained, key remains for the read
    finally:
        command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0
        command_input._PENDING_KEYS.clear()


def test_windows_sgr_mouse_motion_sequence_is_ignored(monkeypatch):
    chars = deque("\x1b[<32;27;29M")

    fake_msvcrt = SimpleNamespace(
        getwch=chars.popleft,
        kbhit=lambda: bool(chars),
    )

    command_input._PENDING_KEYS.clear()
    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.setenv("WT_SESSION", "test")
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    command_input._MOUSE_CAPTURE_ACTIVE_DEPTH = 1
    command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 1

    try:
        assert command_input._read_key(text_mode=True) == "OTHER"
        assert not chars
        assert not command_input._PENDING_KEYS
    finally:
        command_input._MOUSE_CAPTURE_ACTIVE_DEPTH = 0
        command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0


def test_windows_orphaned_sgr_mouse_motion_sequence_is_ignored(monkeypatch):
    chars = deque("[<32;27;29M")

    fake_msvcrt = SimpleNamespace(
        getwch=chars.popleft,
        kbhit=lambda: bool(chars),
    )

    command_input._PENDING_KEYS.clear()
    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.setenv("WT_SESSION", "test")
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    command_input._MOUSE_CAPTURE_ACTIVE_DEPTH = 0
    command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0

    try:
        assert command_input._read_key(text_mode=True) == "OTHER"
        assert not chars
        assert not command_input._PENDING_KEYS
    finally:
        command_input._MOUSE_CAPTURE_ACTIVE_DEPTH = 0
        command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0


def test_windows_vt_arrow_waits_for_delayed_escape_sequence(monkeypatch):
    chars = deque("\x1b[B")
    clock = {"now": 0.0}

    def kbhit():
        if not chars:
            return False
        if chars[0] == "[":
            return clock["now"] >= 0.05
        return True

    fake_msvcrt = SimpleNamespace(
        getwch=chars.popleft,
        kbhit=kbhit,
    )

    command_input._PENDING_KEYS.clear()
    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.setenv("WT_SESSION", "test")
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(command_input.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        command_input.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 1

    try:
        assert command_input._read_key() == "DOWN"
        assert not chars
    finally:
        command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0
        command_input._PENDING_KEYS.clear()


def test_read_key_with_repeats_coalesces_identical_navigation(monkeypatch):
    command_input._PENDING_KEYS.clear()
    keys = ["DOWN", "DOWN", "DOWN", "ENTER"]

    monkeypatch.setattr(command_input, "_read_key", lambda text_mode=False: keys.pop(0))
    monkeypatch.setattr(command_input, "_key_available", lambda: bool(keys))

    assert command_input._read_key_with_repeats() == ("DOWN", 3)
    assert list(command_input._PENDING_KEYS) == ["ENTER"]
    command_input._PENDING_KEYS.clear()


def test_read_key_with_repeats_does_not_drain_non_repeatable_key(monkeypatch):
    command_input._PENDING_KEYS.clear()
    keys = ["x", "DOWN"]

    monkeypatch.setattr(command_input, "_read_key", lambda text_mode=False: keys.pop(0))
    monkeypatch.setattr(command_input, "_key_available", lambda: bool(keys))

    assert command_input._read_key_with_repeats() == ("x", 1)
    assert keys == ["DOWN"]
    command_input._PENDING_KEYS.clear()


def test_read_key_with_repeats_batches_queued_text(monkeypatch):
    command_input._PENDING_KEYS.clear()
    keys = [*"openai/gpt-5.5", "ENTER"]

    monkeypatch.setattr(command_input, "_read_key", lambda text_mode=False: keys.pop(0))
    monkeypatch.setattr(command_input, "_key_available", lambda: bool(keys))

    assert command_input._read_key_with_repeats(
        text_mode=True,
        batch_text=True,
    ) == (command_input.TextInput("openai/gpt-5.5"), 1)
    assert list(command_input._PENDING_KEYS) == ["ENTER"]
    command_input._PENDING_KEYS.clear()


def test_read_key_with_repeats_batches_pasted_lines_without_bracketed_paste(monkeypatch):
    command_input._PENDING_KEYS.clear()
    keys = [*"first", "ENTER", *"second", "ENTER"]

    monkeypatch.setattr(command_input, "_read_key", lambda text_mode=False: keys.pop(0))
    monkeypatch.setattr(command_input, "_key_available", lambda: bool(keys))

    # A multi-line paste coalesces into one token; its trailing newline is part
    # of the paste, so it does NOT leak out as a submit.
    assert command_input._read_key_with_repeats(
        text_mode=True,
        batch_text=True,
    ) == (command_input.TextInput("first\nsecond"), 1)
    assert not command_input._PENDING_KEYS
    command_input._PENDING_KEYS.clear()


def test_read_key_with_repeats_keeps_blank_lines_in_pasted_block(monkeypatch):
    command_input._PENDING_KEYS.clear()
    # "para1\n\npara2" -- the blank line between paragraphs must survive.
    keys = [*"para1", "ENTER", "ENTER", *"para2"]

    monkeypatch.setattr(command_input, "_read_key", lambda text_mode=False: keys.pop(0))
    monkeypatch.setattr(command_input, "_key_available", lambda: bool(keys))

    assert command_input._read_key_with_repeats(
        text_mode=True,
        batch_text=True,
    ) == (command_input.TextInput("para1\n\npara2"), 1)
    assert not command_input._PENDING_KEYS
    command_input._PENDING_KEYS.clear()


def test_read_key_with_repeats_submits_single_typed_line(monkeypatch):
    command_input._PENDING_KEYS.clear()
    keys = [*"hello", "ENTER"]

    monkeypatch.setattr(command_input, "_read_key", lambda text_mode=False: keys.pop(0))
    monkeypatch.setattr(command_input, "_key_available", lambda: bool(keys))

    # A single typed line keeps ENTER as a submit (no embedded newline).
    assert command_input._read_key_with_repeats(
        text_mode=True,
        batch_text=True,
    ) == (command_input.TextInput("hello"), 1)
    assert list(command_input._PENDING_KEYS) == ["ENTER"]
    command_input._PENDING_KEYS.clear()


def test_read_key_with_repeats_uses_resume_window_at_newline(monkeypatch):
    command_input._PENDING_KEYS.clear()
    keys = [*"first", "ENTER", *"second"]
    timeouts: list[float] = []

    def recording_await(in_burst, *, timeout=command_input._PASTE_BURST_GAP_SECONDS, deadline=None):
        timeouts.append(timeout)
        return bool(keys)

    monkeypatch.setattr(command_input, "_read_key", lambda text_mode=False: keys.pop(0))
    monkeypatch.setattr(command_input, "_key_available", lambda: bool(keys))
    monkeypatch.setattr(command_input, "_await_more_input", recording_await)

    # The newline decision waits with the longer resume window, not the 20ms burst
    # gap -- this is what stops a chunked paste's first line being mis-submitted.
    result, _ = command_input._read_key_with_repeats(text_mode=True, batch_text=True)
    assert result == command_input.TextInput("first\nsecond")
    assert command_input._PASTE_RESUME_GAP_SECONDS in timeouts
    assert not command_input._PENDING_KEYS
    command_input._PENDING_KEYS.clear()


def test_await_more_input_returns_immediately_when_key_waiting(monkeypatch):
    monkeypatch.setattr(command_input, "_key_available", lambda: True)
    slept: list[bool] = []
    monkeypatch.setattr(command_input.time, "sleep", lambda *_: slept.append(True))

    assert command_input._await_more_input(False) is True
    assert slept == []


def test_await_more_input_skips_wait_for_lone_keystroke(monkeypatch):
    monkeypatch.setattr(command_input, "_key_available", lambda: False)
    slept: list[bool] = []
    monkeypatch.setattr(command_input.time, "sleep", lambda *_: slept.append(True))

    # A single keystroke (not mid-burst) never pays the bridge latency.
    assert command_input._await_more_input(False) is False
    assert slept == []


def test_await_more_input_deadline_caps_blocking(monkeypatch):
    monkeypatch.setattr(command_input, "_key_available", lambda: False)
    slept: list[bool] = []
    monkeypatch.setattr(command_input.time, "sleep", lambda *_: slept.append(True))

    # A deadline already passed means no sleeping even mid-burst with the long
    # resume-gap timeout -- this is what keeps a slow chunked paste from freezing
    # the heads-up loop thread; the batch just ends and resumes next iteration.
    past = command_input.time.monotonic() - 1.0
    assert (
        command_input._await_more_input(
            True, timeout=command_input._PASTE_RESUME_GAP_SECONDS, deadline=past
        )
        is False
    )
    assert slept == []


def test_await_more_input_bridges_burst_gap(monkeypatch):
    # The next chunk of a chunked paste shows up on the second poll.
    polls = iter([False, True])
    monkeypatch.setattr(command_input, "_key_available", lambda: next(polls))
    monkeypatch.setattr(command_input.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(command_input.time, "sleep", lambda *_: None)

    assert command_input._await_more_input(True) is True


def test_await_more_input_gives_up_after_gap_window(monkeypatch):
    monkeypatch.setattr(command_input, "_key_available", lambda: False)
    monkeypatch.setattr(command_input.time, "sleep", lambda *_: None)
    times = iter([0.0, 0.0, 1.0])  # deadline anchor, one poll, then past deadline
    monkeypatch.setattr(command_input.time, "monotonic", lambda: next(times))

    assert command_input._await_more_input(True) is False


def test_read_key_with_repeats_drops_batched_sgr_mouse_text(monkeypatch):
    command_input._PENDING_KEYS.clear()
    keys = [*"[<35;62;15M"]

    monkeypatch.setattr(command_input, "_read_key", lambda text_mode=False: keys.pop(0))
    monkeypatch.setattr(command_input, "_key_available", lambda: bool(keys))

    assert command_input._read_key_with_repeats(
        text_mode=True,
        batch_text=True,
    ) == ("OTHER", 1)
    assert not command_input._PENDING_KEYS
    command_input._PENDING_KEYS.clear()


def test_read_key_with_repeats_strips_embedded_sgr_mouse_text(monkeypatch):
    command_input._PENDING_KEYS.clear()
    keys = [*"hi[<35;62;15Mthere"]

    monkeypatch.setattr(command_input, "_read_key", lambda text_mode=False: keys.pop(0))
    monkeypatch.setattr(command_input, "_key_available", lambda: bool(keys))

    assert command_input._read_key_with_repeats(
        text_mode=True,
        batch_text=True,
    ) == (command_input.TextInput("hithere"), 1)
    assert not command_input._PENDING_KEYS
    command_input._PENDING_KEYS.clear()


def test_read_key_returns_resize_when_posix_terminal_size_changes(monkeypatch):
    _install_posix_input(monkeypatch, "")
    sizes = [
        command_input.os.terminal_size((80, 24)),
        command_input.os.terminal_size((120, 40)),
    ]

    def terminal_size(_fd):
        return sizes.pop(0) if len(sizes) > 1 else sizes[0]

    monkeypatch.setattr(command_input.os, "get_terminal_size", terminal_size)

    assert command_input._read_key() == "RESIZE"


def test_read_key_maps_posix_delete(monkeypatch):
    stdin = _install_posix_input(monkeypatch, "\x1b[3~")

    assert command_input._read_key(text_mode=True) == "DELETE"
    assert stdin.remaining == ""


def test_read_key_maps_posix_modified_arrows(monkeypatch):
    cases = {
        "\x1b[1;5D": "CTRL_LEFT",
        "\x1b[1;5C": "CTRL_RIGHT",
        "\x1b[1;2D": "SHIFT_LEFT",
        "\x1b[1;2C": "SHIFT_RIGHT",
        "\x1b[1;6D": "CTRL_SHIFT_LEFT",
        "\x1b[1;6C": "CTRL_SHIFT_RIGHT",
    }
    for sequence, expected in cases.items():
        stdin = _install_posix_input(monkeypatch, sequence)
        assert command_input._read_key(text_mode=True) == expected
        assert stdin.remaining == ""


def test_read_key_maps_posix_modified_non_arrow_to_base(monkeypatch):
    # A modifier on a non-arrow nav key falls back to its plain token, and the
    # whole sequence is consumed (no leftover bytes leak as literal input).
    stdin = _install_posix_input(monkeypatch, "\x1b[1;5H")
    assert command_input._read_key(text_mode=True) == "HOME"
    assert stdin.remaining == ""


def test_read_key_maps_posix_bracketed_paste_to_text_input(monkeypatch):
    stdin = _install_posix_input(monkeypatch, "\x1b[200~first\nsecond\x1b[201~")

    key = command_input._read_key(text_mode=True)

    assert key == command_input.TextInput("first\nsecond")
    assert stdin.remaining == ""


def test_read_key_maps_windows_bracketed_paste_to_text_input(monkeypatch):
    # With VT input enabled, Windows delivers a paste wrapped in the bracketed
    # paste markers and the whole block must come back as one TextInput -- never
    # fragmenting into per-line ENTER submits.
    chars = deque("\x1b[200~first\nsecond\x1b[201~")

    fake_msvcrt = SimpleNamespace(
        getwch=chars.popleft,
        kbhit=lambda: bool(chars),
    )

    command_input._PENDING_KEYS.clear()
    monkeypatch.setattr(command_input.sys, "platform", "win32")
    monkeypatch.setenv("WT_SESSION", "test")
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0

    try:
        assert command_input._read_key(text_mode=True) == command_input.TextInput(
            "first\nsecond"
        )
        assert not chars
        assert not command_input._PENDING_KEYS
    finally:
        command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0
        command_input._PENDING_KEYS.clear()


def test_read_key_maps_posix_ctrl_v_and_alt_v(monkeypatch):
    stdin = _install_posix_input(monkeypatch, "\x16")
    assert command_input._read_key(text_mode=True) == "CTRL_V"
    assert stdin.remaining == ""

    stdin = _install_posix_input(monkeypatch, "\x1bv")
    assert command_input._read_key(text_mode=True) == "ALT_V"
    assert stdin.remaining == ""


def test_read_key_maps_windows_ctrl_v_and_alt_v(monkeypatch):
    # Ctrl+V arrives as a raw \x16 when the terminal doesn't own the paste
    # binding; Alt+V arrives ESC-prefixed under VT input and is the fallback
    # for terminals (Windows Terminal) whose paste action swallows Ctrl+V.
    for sequence, expected in {"\x16": "CTRL_V", "\x1bv": "ALT_V"}.items():
        chars = deque(sequence)
        fake_msvcrt = SimpleNamespace(
            getwch=chars.popleft,
            kbhit=lambda chars=chars: bool(chars),
        )
        command_input._PENDING_KEYS.clear()
        monkeypatch.setattr(command_input.sys, "platform", "win32")
        monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
        command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0
        try:
            assert command_input._read_key(text_mode=True) == expected
            assert not chars
            assert not command_input._PENDING_KEYS
        finally:
            command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0
            command_input._PENDING_KEYS.clear()


def test_read_key_maps_windows_modified_arrows(monkeypatch):
    # With VT input enabled, Windows delivers modified arrows as xterm-style
    # CSI sequences (ESC [ 1 ; <mod> <final>) which must decode to the same
    # CTRL_/SHIFT_ tokens as POSIX -- and consume the whole sequence.
    cases = {
        "\x1b[1;5D": "CTRL_LEFT",
        "\x1b[1;5C": "CTRL_RIGHT",
        "\x1b[1;2D": "SHIFT_LEFT",
        "\x1b[1;6C": "CTRL_SHIFT_RIGHT",
    }
    for sequence, expected in cases.items():
        chars = deque(sequence)
        fake_msvcrt = SimpleNamespace(
            getwch=chars.popleft,
            kbhit=lambda chars=chars: bool(chars),
        )
        command_input._PENDING_KEYS.clear()
        monkeypatch.setattr(command_input.sys, "platform", "win32")
        monkeypatch.setenv("WT_SESSION", "test")
        monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
        command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0
        try:
            assert command_input._read_key(text_mode=True) == expected
            assert not chars
            assert not command_input._PENDING_KEYS
        finally:
            command_input._WINDOWS_MOUSE_CAPTURE_DEPTH = 0
            command_input._PENDING_KEYS.clear()


def test_read_editable_line_edits_prefilled_text():
    keys = iter(["LEFT", "LEFT", "X", "DELETE", "ENTER"])
    output = []

    result = command_input.read_editable_line(
        "jarv> ",
        initial="hello",
        read_key=lambda: next(keys),
        write=output.append,
    )

    assert result == "helXo"
    assert output[-1] == "\n"


def test_render_editable_line_keeps_long_input_on_one_terminal_row():
    output = []

    command_input._render_editable_line(
        "\x1b[1;36mjarv>\x1b[0m ",
        "abcdefghijklmnopqrstuvwxyz",
        26,
        write=output.append,
        columns=16,
    )

    assert output == [
        "\x1b[?25l\r\x1b[2K\x1b[1;36mjarv>\x1b[0m rstuvwxyz\x1b[?25h"
    ]
    rendered = output[0]
    assert "abcdefghijklmnopqr" not in rendered


def test_render_editable_line_accounts_for_wide_unicode_characters():
    output = []

    command_input._render_editable_line(
        "jarv> ",
        "ab界cd",
        5,
        write=output.append,
        columns=11,
    )

    assert output == ["\x1b[?25l\r\x1b[2Kjarv> 界cd\x1b[?25h"]


def test_render_editable_line_restores_cursor_after_positioning():
    output = []

    command_input._render_editable_line(
        "jarv> ",
        "hello",
        2,
        write=output.append,
        columns=20,
    )

    assert output == ["\x1b[?25l\r\x1b[2Kjarv> hello\x1b[3D\x1b[?25h"]


def test_render_editable_line_applies_text_style_without_coloring_prompt():
    output = []

    command_input._render_editable_line(
        "\x1b[1;36m>\x1b[0m ",
        "yes",
        3,
        text_style="\x1b[97m",
        write=output.append,
        columns=20,
    )

    assert output == [
        "\x1b[?25l\r\x1b[2K\x1b[1;36m>\x1b[0m "
        "\x1b[97myes\x1b[0m\x1b[?25h"
    ]


def test_read_editable_line_batches_queued_paste_before_redrawing():
    keys = deque([*"a long pasted prompt", "ENTER"])
    output = []

    result = command_input.read_editable_line(
        "jarv> ",
        read_key=keys.popleft,
        key_available=lambda: bool(keys),
        write=output.append,
    )

    assert result == "a long pasted prompt"
    repaints = [item for item in output if item.startswith(command_input.CURSOR_HIDE)]
    assert len(repaints) == 2
    assert all(item.endswith(command_input.CURSOR_SHOW) for item in repaints)
    assert output[-1] == "\n"


def test_read_editable_line_ctrl_c_clears_before_exiting():
    actions = iter([KeyboardInterrupt(), "h", "i", "ENTER"])

    def read_key():
        action = next(actions)
        if isinstance(action, BaseException):
            raise action
        return action

    result = command_input.read_editable_line(
        "jarv> ",
        initial="restore me",
        read_key=read_key,
        write=lambda _text: None,
    )

    assert result == "hi"


def test_read_editable_line_collapses_and_expands_multiline_paste():
    keys = iter([command_input.TextInput("first\nsecond\nthird"), *" go", "ENTER"])
    captured: list[str] = []

    result = command_input.read_editable_line(
        "jarv> ",
        read_key=lambda: next(keys),
        write=captured.append,
    )

    # The buffer carries the short marker; the submitted line restores the paste.
    assert "[Pasted text #1 +3 lines] go" in "".join(captured)
    assert result == "first\nsecond\nthird go"


def test_read_editable_line_inserts_single_line_paste_inline():
    keys = iter([command_input.TextInput("hello world"), "ENTER"])

    result = command_input.read_editable_line(
        "jarv> ",
        read_key=lambda: next(keys),
        write=lambda _text: None,
    )

    assert result == "hello world"


def test_read_editable_line_backspace_removes_whole_paste_marker():
    # The chip is the only thing in the line; one Backspace clears it entirely.
    keys = iter([command_input.TextInput("first\nsecond\nthird"), "BACKSPACE", "ENTER"])

    result = command_input.read_editable_line(
        "jarv> ",
        read_key=lambda: next(keys),
        write=lambda _text: None,
    )

    assert result == ""


def test_read_editable_line_backspace_marker_keeps_neighbouring_text():
    keys = iter(
        [
            command_input.TextInput("first\nsecond"),
            *"ok",
            "LEFT",
            "LEFT",
            "BACKSPACE",  # cursor sits just after the chip -> remove the chip only
            "ENTER",
        ]
    )

    result = command_input.read_editable_line(
        "jarv> ",
        read_key=lambda: next(keys),
        write=lambda _text: None,
    )

    assert result == "ok"


def test_read_editable_line_duplicate_paste_unboxes_to_plain_text():
    block = command_input.TextInput("first\nsecond\nthird")
    keys = iter([block, block, "ENTER"])

    result = command_input.read_editable_line(
        "jarv> ",
        read_key=lambda: next(keys),
        write=lambda _text: None,
    )

    # One plain (flattened) copy, no markers left to expand.
    assert result == "first second third"


def test_paste_registry_span_helpers_locate_markers():
    registry = command_input.PasteRegistry()
    marker = registry.collapse("a\nb\nc")
    buffer = f"x{marker}y"

    assert registry.marker_spans(buffer) == [(1, 1 + len(marker))]
    assert registry.span_covering(buffer, 1 + len(marker) - 1) == (1, 1 + len(marker))
    assert registry.span_covering(buffer, 0) is None


def test_paste_registry_duplicate_span_matches_adjacent_marker():
    registry = command_input.PasteRegistry()
    marker = registry.collapse("a\nb")
    buffer = marker

    # Same content, cursor just after the marker -> adjacency hit.
    assert registry.duplicate_span(buffer, len(marker), "a\nb") == (0, len(marker))
    # Cursor just before the marker is adjacent too.
    assert registry.duplicate_span(buffer, 0, "a\nb") == (0, len(marker))
    # Different content never matches.
    assert registry.duplicate_span(buffer, len(marker), "x\ny") is None


def test_paste_registry_attach_registers_synthetic_chip():
    registry = command_input.PasteRegistry()
    marker = registry.attach("Image", "[attached image: /tmp/shot.png]")

    assert marker == "[Image #1]"
    assert registry.expand(f"see {marker}") == "see [attached image: /tmp/shot.png]"
    assert registry.marker_spans(marker) == [(0, len(marker))]
    # The counter is shared with collapsed pastes so markers never collide.
    assert registry.collapse("a\nb") == "[Pasted text #2 +2 lines]"


def test_paste_registry_prune_drops_absent_markers():
    registry = command_input.PasteRegistry()
    marker = registry.collapse("a\nb")

    registry.prune("nothing here")
    assert registry.expand(marker) == marker  # mapping forgotten


def test_read_editable_line_ctrl_c_exits_when_empty():
    def interrupt():
        raise KeyboardInterrupt

    try:
        command_input.read_editable_line(
            "jarv> ",
            read_key=interrupt,
            write=lambda _text: None,
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("empty Ctrl+C should exit the line editor")


def test_requeue_key_returns_key_to_pending_queue():
    command_input._PENDING_KEYS.clear()
    command_input.requeue_key("ENTER")
    assert command_input._read_key() == "ENTER"
    assert not command_input._PENDING_KEYS
