import sys
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
    monkeypatch.setitem(sys.modules, "tty", SimpleNamespace(setraw=lambda _fd: None))
    monkeypatch.setitem(
        sys.modules,
        "termios",
        SimpleNamespace(
            TCSADRAIN=1,
            tcgetattr=lambda _fd: "old",
            tcsetattr=lambda _fd, _when, _old: None,
        ),
    )
    return stdin


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
