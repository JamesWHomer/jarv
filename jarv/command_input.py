"""Keyboard input helpers shared by interactive command screens."""

import sys


def _read_key(text_mode: bool = False) -> str:
    """Read a single keypress and return a normalised token.

    Returns one of: UP, DOWN, LEFT, RIGHT, HOME, END, PAGEUP, PAGEDOWN,
    ENTER, ESC, TAB, CTRL_F, BACKSPACE, or the raw character.  Raises KeyboardInterrupt on
    Ctrl-C.  When ``text_mode`` is True, the convenience q/Q → ESC mapping is
    disabled so a search query can include those letters.
    """
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            second = msvcrt.getwch()
            return {
                "H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT",
                "G": "HOME", "O": "END",
                "I": "PAGEUP", "Q": "PAGEDOWN",
            }.get(second, "OTHER")
        if ch == "\r":
            return "ENTER"
        if ch == "\t":
            return "TAB"
        if ch == "\x1b":
            return "ESC"
        if not text_mode and ch in ("q", "Q"):
            return "ESC"
        if ch == "\x06":
            return "CTRL_F"
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
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 in ("5", "6"):
                        sys.stdin.read(1)  # consume trailing ~
                    return {
                        "A": "UP", "B": "DOWN", "D": "LEFT", "C": "RIGHT",
                        "H": "HOME", "F": "END",
                        "5": "PAGEUP", "6": "PAGEDOWN",
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
            if ch in ("\x7f", "\x08"):
                return "BACKSPACE"
            if ch == "\x03":
                raise KeyboardInterrupt
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

