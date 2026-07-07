"""Best-effort clipboard access for the terminal UI.

There is no GUI toolkit here, so "copy" has two possible backends:

* a **native** clipboard tool (Win32 ``CF_UNICODETEXT`` via ctypes, ``pbcopy``
  on macOS, ``wl-copy``/``xclip``/``xsel`` on Linux) -- reliable locally and
  Unicode-correct, and
* an **OSC 52** escape sequence that asks the *terminal* to store the text --
  the only thing that works over SSH, when the emulator supports it.

:func:`copy_to_clipboard` tries the native tool first (so the common local case
populates the real OS clipboard without risking stray escape bytes on terminals
that ignore OSC 52) and falls back to OSC 52 when no native tool is available.

The *paste* direction has no OSC-52-style terminal fallback: terminals only
ever deliver clipboard **text** through the input stream, so an image on the
OS clipboard is invisible to them. :func:`read_clipboard_image` therefore goes
to the OS clipboard directly -- PowerShell on Windows, ``osascript`` (or
``pngpaste``) on macOS, ``wl-paste``/``xclip`` on Linux -- and materialises the
image as a file: either the file the user copied (a Finder/Explorer copy) or a
PNG saved from the raw bitmap. :func:`read_clipboard_text` is the matching text
reader, used when a terminal passes Ctrl+V through without handling it.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

__all__ = [
    "ClipboardImage",
    "copy_to_clipboard",
    "read_clipboard_image",
    "read_clipboard_text",
]

# Only the image types every provider path in read_tool accepts as message
# content; a copied .bmp/.tiff would attach a file the model then can't view.
_IMAGE_SUFFIX_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_IMAGE_MEDIA_TYPE_SUFFIXES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_PASTE_SUBPROCESS_TIMEOUT = 15.0


def copy_to_clipboard(text: str, *, write=None) -> bool:
    """Copy ``text`` to the clipboard. Returns True on a best-effort success.

    ``write`` overrides the OSC 52 output sink (used by tests); production code
    leaves it ``None`` so the sequence goes to the live terminal on stdout.
    """
    if not text:
        return True
    if _native_copy(text):
        return True
    return _osc52_copy(text, write=write)


def _osc52_sequence(text: str) -> str:
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"\x1b]52;c;{payload}\x07"


def _osc52_copy(text: str, *, write=None) -> bool:
    stream = None
    if write is None:
        # Unwrap Rich Live's stdout proxy: copies are triggered from inside
        # full-screen views, and the proxy line-buffers/interprets raw escapes
        # instead of passing them to the terminal.
        stream = sys.stdout
        proxied = getattr(stream, "rich_proxied_file", None)
        if proxied is not None:
            stream = proxied
        if not stream.isatty():
            return False
        write = stream.write
    try:
        write(_osc52_sequence(text))
    except Exception:
        return False
    try:
        (stream if stream is not None else sys.stdout).flush()
    except Exception:
        pass
    return True


def _native_copy(text: str) -> bool:
    if sys.platform == "win32":
        return _windows_copy(text)
    if sys.platform == "darwin":
        return _subprocess_copy(["pbcopy"], text)
    # Linux / BSD: Wayland first, then the X11 helpers in likelihood order.
    for cmd in (
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
    ):
        if _subprocess_copy(cmd, text):
            return True
    return False


def _subprocess_copy(cmd: list[str], text: str) -> bool:
    try:
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, ValueError):
        return False
    return proc.returncode == 0


def _windows_copy(text: str) -> bool:
    """Set ``CF_UNICODETEXT`` via the Win32 clipboard API (no dependencies)."""
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return False

    CF_UNICODETEXT = 13
    GMEM_MOVEABLE = 0x0002

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except Exception:
        return False

    # Declare arg/return types so 64-bit handles are not truncated to int.
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype = wintypes.HGLOBAL

    # The Windows clipboard expects CRLF line endings.
    data = text.replace("\r\n", "\n").replace("\n", "\r\n")
    buffer = ctypes.create_unicode_buffer(data)  # NUL-terminated UTF-16
    size = ctypes.sizeof(buffer)

    if not user32.OpenClipboard(None):
        return False
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            return False
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            kernel32.GlobalFree(handle)
            return False
        try:
            ctypes.memmove(pointer, buffer, size)
        finally:
            kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
            return False
        # Ownership of the moveable block transfers to the clipboard on success.
        return True
    finally:
        user32.CloseClipboard()


@dataclass(frozen=True)
class ClipboardImage:
    """An image found on the OS clipboard, materialised as a local file."""

    path: Path
    media_type: str


def clipboard_image_dir() -> Path:
    """Directory pasted clipboard bitmaps are saved into (temp, per-OS-cleanup)."""
    return Path(tempfile.gettempdir()) / "jarv-clipboard"


def read_clipboard_image(save_dir: Path | None = None) -> ClipboardImage | None:
    """Return the image on the OS clipboard as a file, or ``None``.

    A copied *file* (Finder/Explorer) is referenced in place; raw bitmap data
    is saved as a PNG under ``save_dir``. Only image types the providers accept
    as message content are returned -- see ``_IMAGE_SUFFIX_MEDIA_TYPES``.
    """
    if save_dir is None:
        save_dir = clipboard_image_dir()
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    if sys.platform == "win32":
        return _windows_clipboard_image(save_dir)
    if sys.platform == "darwin":
        return _mac_clipboard_image(save_dir)
    return _linux_clipboard_image(save_dir)


def read_clipboard_text() -> str | None:
    """Return the text on the OS clipboard (newlines normalised), or ``None``."""
    if sys.platform == "win32":
        text = _windows_clipboard_text()
    elif sys.platform == "darwin":
        text = _subprocess_paste(["pbpaste"])
    else:
        text = None
        for cmd in (
            ["wl-paste", "--no-newline"],
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "--clipboard", "--output"],
        ):
            text = _subprocess_paste(cmd)
            if text is not None:
                break
    if not text:
        return None
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _image_target(save_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return save_dir / f"image-{stamp}-{uuid.uuid4().hex[:8]}.png"


def _image_file_reference(path_text: str) -> ClipboardImage | None:
    """A copied file, referenced in place -- image extensions only."""
    path_text = path_text.strip()
    if not path_text:
        return None
    path = Path(path_text)
    media_type = _IMAGE_SUFFIX_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        return None
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    return ClipboardImage(path, media_type)


def _run_paste(cmd: list[str], *, env: dict | None = None):
    """Run a clipboard helper, returning the CompletedProcess or ``None``."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            env=env,
            check=False,
            timeout=_PASTE_SUBPROCESS_TIMEOUT,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return None


def _subprocess_paste(cmd: list[str]) -> str | None:
    proc = _run_paste(cmd)
    if proc is None or proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace") or None


# Copied files come through as a FileDropList; raw bitmaps via Get-Clipboard
# -Format Image (a System.Drawing bitmap), saved straight to PNG. The target
# path travels in an env var so no quoting can break the -Command string.
# powershell.exe (Windows PowerShell 5.1) is used explicitly: pwsh 7 dropped
# -Format Image, and 5.1 ships on every Windows install.
_WINDOWS_IMAGE_PASTE_SCRIPT = """\
$ErrorActionPreference = 'SilentlyContinue'
$files = Get-Clipboard -Format FileDropList
if ($files) { $files | ForEach-Object { Write-Output ('file:' + $_.FullName) }; exit 0 }
Add-Type -AssemblyName System.Drawing | Out-Null
$img = Get-Clipboard -Format Image
if ($img -ne $null) {
    $img.Save($env:JARV_CLIPBOARD_TARGET, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Output 'saved'
}
"""


def _windows_clipboard_image(save_dir: Path) -> ClipboardImage | None:
    target = _image_target(save_dir)
    env = dict(os.environ, JARV_CLIPBOARD_TARGET=str(target))
    proc = _run_paste(
        [
            "powershell",
            "-NoProfile",
            "-STA",
            "-NonInteractive",
            "-Command",
            _WINDOWS_IMAGE_PASTE_SCRIPT,
        ],
        env=env,
    )
    if proc is None or proc.returncode != 0:
        return None
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line == "saved":
            if target.is_file():
                return ClipboardImage(target, "image/png")
            return None
        if line.startswith("file:"):
            image = _image_file_reference(line[len("file:"):])
            if image is not None:
                return image
    return None


def _windows_clipboard_text() -> str | None:
    """Read ``CF_UNICODETEXT`` via the Win32 clipboard API (no dependencies)."""
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    CF_UNICODETEXT = 13

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except Exception:
        return None

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL

    if not user32.OpenClipboard(None):
        return None
    try:
        if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return None
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return None
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return None
        try:
            return ctypes.wstring_at(pointer)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _mac_clipboard_image(save_dir: Path) -> ClipboardImage | None:
    # A Finder copy puts a file URL on the clipboard (often alongside the
    # file's *icon* as image data), so the file reference must win: pasting
    # the icon bitmap instead of the copied photo would be wrong.
    proc = _run_paste(
        ["osascript", "-e", "POSIX path of (the clipboard as «class furl»)"]
    )
    if proc is not None and proc.returncode == 0:
        image = _image_file_reference(proc.stdout.decode("utf-8", errors="replace"))
        # A non-image file was copied; there is no bitmap behind it.
        return image

    target = _image_target(save_dir)
    proc = _run_paste(["pngpaste", str(target)])
    if proc is not None and proc.returncode == 0 and target.is_file():
        return ClipboardImage(target, "image/png")

    # The first line raises when the clipboard has no image, so no file is
    # created in that case; a partial write is cleaned up below.
    script = (
        "set imgData to the clipboard as «class PNGf»\n"
        f'set f to open for access POSIX file "{target}" with write permission\n'
        "write imgData to f\n"
        "close access f\n"
    )
    proc = _run_paste(["osascript", "-e", script])
    if proc is not None and proc.returncode == 0 and target.is_file():
        return ClipboardImage(target, "image/png")
    try:
        target.unlink()
    except OSError:
        pass
    return None


def _first_file_uri(text: str) -> str | None:
    """The local path of the first ``file://`` URI in a ``text/uri-list`` body."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("file://"):
            path = unquote(urlsplit(line).path)
            # A file:///C:/... URI keeps a leading slash before the drive letter.
            if len(path) >= 3 and path[0] == "/" and path[2] == ":":
                path = path[1:]
            return path
        return None
    return None


def _run_paste_bytes(cmd: list[str]) -> bytes | None:
    proc = _run_paste(cmd)
    if proc is None or proc.returncode != 0:
        return None
    return proc.stdout or None


def _linux_image_from_targets(
    types: list[str],
    read_cmd_for,
    save_dir: Path,
) -> ClipboardImage | None:
    available = set(types)
    # Prefer the actual bitmap: an image copied from a browser advertises
    # image/png, while a file-manager copy has only text/uri-list.
    for media_type, suffix in _IMAGE_MEDIA_TYPE_SUFFIXES.items():
        if media_type not in available:
            continue
        data = _run_paste_bytes(read_cmd_for(media_type))
        if not data:
            return None
        target = _image_target(save_dir).with_suffix(suffix)
        try:
            target.write_bytes(data)
        except OSError:
            return None
        return ClipboardImage(target, media_type)
    if "text/uri-list" in available:
        raw = _run_paste_bytes(read_cmd_for("text/uri-list"))
        if raw:
            path_text = _first_file_uri(raw.decode("utf-8", errors="replace"))
            if path_text:
                return _image_file_reference(path_text)
    return None


def _linux_clipboard_image(save_dir: Path) -> ClipboardImage | None:
    for list_cmd, read_cmd_for in (
        (
            ["wl-paste", "--list-types"],
            lambda t: ["wl-paste", "--type", t],
        ),
        (
            ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"],
            lambda t: ["xclip", "-selection", "clipboard", "-t", t, "-o"],
        ),
    ):
        proc = _run_paste(list_cmd)
        if proc is None or proc.returncode != 0:
            continue
        types = proc.stdout.decode("utf-8", errors="replace").split()
        # This backend owns the display's clipboard: an empty answer means
        # there is no image, not that the other tool should be consulted.
        return _linux_image_from_targets(types, read_cmd_for, save_dir)
    return None
