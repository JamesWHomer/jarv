import base64
from unittest.mock import patch

from jarv import clipboard
from jarv.clipboard import copy_to_clipboard


def _decode_osc52(sequence: str) -> str:
    assert sequence.startswith("\x1b]52;c;")
    assert sequence.endswith("\x07")
    payload = sequence[len("\x1b]52;c;") : -1]
    return base64.b64decode(payload).decode("utf-8")


def test_osc52_sequence_round_trips_unicode():
    text = "café \U0001f600 你好"
    assert _decode_osc52(clipboard._osc52_sequence(text)) == text


def test_copy_prefers_native_backend():
    captured = []
    with patch.object(clipboard, "_native_copy", return_value=True) as native:
        assert copy_to_clipboard("hello", write=captured.append) is True
    native.assert_called_once_with("hello")
    # Native success short-circuits, so nothing is emitted to the terminal.
    assert captured == []


def test_copy_falls_back_to_osc52_when_native_unavailable():
    captured = []
    with patch.object(clipboard, "_native_copy", return_value=False):
        assert copy_to_clipboard("hello", write=captured.append) is True
    assert _decode_osc52("".join(captured)) == "hello"


def test_empty_text_is_a_noop_success():
    captured = []
    with patch.object(clipboard, "_native_copy") as native:
        assert copy_to_clipboard("", write=captured.append) is True
    native.assert_not_called()
    assert captured == []


def test_osc52_copy_skips_when_not_a_tty():
    with patch("jarv.clipboard.sys.stdout") as stdout:
        stdout.isatty.return_value = False
        assert clipboard._osc52_copy("hello") is False
