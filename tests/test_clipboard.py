import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jarv import clipboard
from jarv.clipboard import ClipboardImage, copy_to_clipboard


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


class _NonTtyStdout:
    # A plain fake, not MagicMock: real stdout objects have no
    # rich_proxied_file attribute, and auto-created mock attributes would
    # falsely trigger the Live-proxy unwrap in _osc52_copy.
    def __init__(self):
        self.writes = []

    def isatty(self):
        return False

    def write(self, text):
        self.writes.append(text)

    def flush(self):
        pass


def test_osc52_copy_skips_when_not_a_tty():
    stdout = _NonTtyStdout()
    with patch("jarv.clipboard.sys.stdout", stdout):
        assert clipboard._osc52_copy("hello") is False
    assert stdout.writes == []


def test_osc52_copy_unwraps_rich_stdout_proxy():
    class TtyStdout(_NonTtyStdout):
        def isatty(self):
            return True

    real = TtyStdout()

    class Proxy(_NonTtyStdout):
        rich_proxied_file = real

    proxy = Proxy()
    with patch("jarv.clipboard.sys.stdout", proxy):
        assert clipboard._osc52_copy("hello") is True
    assert proxy.writes == []
    assert _decode_osc52("".join(real.writes)) == "hello"


# --------------------------------------------------------------------------- #
# Paste direction: read_clipboard_image / read_clipboard_text
# --------------------------------------------------------------------------- #


def _completed(stdout: bytes, returncode: int = 0):
    return SimpleNamespace(stdout=stdout, returncode=returncode)


def test_first_file_uri_parses_uri_list():
    assert clipboard._first_file_uri("file:///home/me/a%20b.png\r\n") == "/home/me/a b.png"
    assert clipboard._first_file_uri("# comment\nfile:///x.png") == "/x.png"
    # A non-file URI at the head means the clipboard isn't a local file copy.
    assert clipboard._first_file_uri("https://example.com/a.png") is None
    assert clipboard._first_file_uri("") is None


def test_image_file_reference_accepts_only_supported_image_files(tmp_path):
    image = tmp_path / "shot.PNG"
    image.write_bytes(b"fake")
    ref = clipboard._image_file_reference(str(image))
    assert ref == ClipboardImage(image, "image/png")

    text_file = tmp_path / "notes.txt"
    text_file.write_bytes(b"fake")
    assert clipboard._image_file_reference(str(text_file)) is None
    assert clipboard._image_file_reference(str(tmp_path / "missing.png")) is None
    assert clipboard._image_file_reference("  ") is None


def test_windows_clipboard_image_uses_copied_file_reference(tmp_path):
    copied = tmp_path / "photo.jpg"
    copied.write_bytes(b"fake")
    stdout = f"file:{tmp_path / 'ignore.txt'}\nfile:{copied}\n".encode("utf-8")
    with patch.object(clipboard, "_run_paste", return_value=_completed(stdout)):
        image = clipboard._windows_clipboard_image(tmp_path)
    assert image == ClipboardImage(copied, "image/jpeg")


def test_windows_clipboard_image_saves_bitmap_to_target(tmp_path):
    def fake_run(cmd, *, env=None):
        target = Path(env["JARV_CLIPBOARD_TARGET"])
        target.write_bytes(b"png-bytes")
        return _completed(b"saved\n")

    with patch.object(clipboard, "_run_paste", fake_run):
        image = clipboard._windows_clipboard_image(tmp_path)
    assert image is not None
    assert image.media_type == "image/png"
    assert image.path.parent == tmp_path
    assert image.path.read_bytes() == b"png-bytes"


def test_windows_clipboard_image_returns_none_without_image(tmp_path):
    with patch.object(clipboard, "_run_paste", return_value=_completed(b"")):
        assert clipboard._windows_clipboard_image(tmp_path) is None
    with patch.object(clipboard, "_run_paste", return_value=None):
        assert clipboard._windows_clipboard_image(tmp_path) is None


def test_linux_image_from_targets_prefers_bitmap(tmp_path):
    reads = []

    def read_bytes(cmd):
        reads.append(cmd)
        return b"png-bytes"

    with patch.object(clipboard, "_run_paste_bytes", read_bytes):
        image = clipboard._linux_image_from_targets(
            ["text/uri-list", "image/png"],
            lambda t: ["wl-paste", "--type", t],
            tmp_path,
        )
    assert reads == [["wl-paste", "--type", "image/png"]]
    assert image is not None
    assert image.media_type == "image/png"
    assert image.path.suffix == ".png"
    assert image.path.read_bytes() == b"png-bytes"


def test_linux_image_from_targets_follows_uri_list_file_copy(tmp_path):
    copied = tmp_path / "photo.webp"
    copied.write_bytes(b"fake")
    uri = copied.as_uri() + "\n"

    with patch.object(
        clipboard, "_run_paste_bytes", return_value=uri.encode("utf-8")
    ):
        image = clipboard._linux_image_from_targets(
            ["text/uri-list", "TARGETS"],
            lambda t: ["xclip", "-t", t, "-o"],
            tmp_path,
        )
    assert image == ClipboardImage(copied, "image/webp")


def test_linux_image_from_targets_without_image_types(tmp_path):
    with patch.object(clipboard, "_run_paste_bytes", return_value=b"whatever"):
        assert (
            clipboard._linux_image_from_targets(
                ["UTF8_STRING", "text/plain"], lambda t: [t], tmp_path
            )
            is None
        )


def test_read_clipboard_text_normalises_newlines():
    with patch.object(clipboard.sys, "platform", "win32"), patch.object(
        clipboard, "_windows_clipboard_text", return_value="a\r\nb\rc"
    ):
        assert clipboard.read_clipboard_text() == "a\nb\nc"


def test_read_clipboard_text_empty_is_none():
    with patch.object(clipboard.sys, "platform", "win32"), patch.object(
        clipboard, "_windows_clipboard_text", return_value=""
    ):
        assert clipboard.read_clipboard_text() is None


def test_run_paste_swallows_launch_failures():
    def boom(*_args, **_kwargs):
        raise OSError("missing tool")

    with patch.object(clipboard.subprocess, "run", boom):
        assert clipboard._run_paste(["nope"]) is None


def test_subprocess_paste_requires_success():
    with patch.object(
        clipboard, "_run_paste", return_value=_completed(b"out", returncode=1)
    ):
        assert clipboard._subprocess_paste(["cmd"]) is None
    with patch.object(clipboard, "_run_paste", return_value=_completed(b"out")):
        assert clipboard._subprocess_paste(["cmd"]) == "out"
