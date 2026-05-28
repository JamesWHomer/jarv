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
    monkeypatch.setattr(command_input.sys, "platform", "linux")
    monkeypatch.setattr(command_input.sys, "stdin", stdin)
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
