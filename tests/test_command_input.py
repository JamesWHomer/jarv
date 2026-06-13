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


def test_read_key_maps_posix_delete(monkeypatch):
    stdin = _install_posix_input(monkeypatch, "\x1b[3~")

    assert command_input._read_key(text_mode=True) == "DELETE"
    assert stdin.remaining == ""


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
