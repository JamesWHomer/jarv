"""Best-effort clipboard copy for the terminal UI.

There is no GUI toolkit here, so "copy" has two possible backends:

* a **native** clipboard tool (Win32 ``CF_UNICODETEXT`` via ctypes, ``pbcopy``
  on macOS, ``wl-copy``/``xclip``/``xsel`` on Linux) -- reliable locally and
  Unicode-correct, and
* an **OSC 52** escape sequence that asks the *terminal* to store the text --
  the only thing that works over SSH, when the emulator supports it.

:func:`copy_to_clipboard` tries the native tool first (so the common local case
populates the real OS clipboard without risking stray escape bytes on terminals
that ignore OSC 52) and falls back to OSC 52 when no native tool is available.
"""

from __future__ import annotations

import base64
import subprocess
import sys

__all__ = ["copy_to_clipboard"]


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
    if write is None:
        if not sys.stdout.isatty():
            return False
        write = sys.stdout.write
    try:
        write(_osc52_sequence(text))
    except Exception:
        return False
    try:
        sys.stdout.flush()
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
