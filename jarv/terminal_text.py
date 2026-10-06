"""Render untrusted terminal text visibly without changing its stored value."""

_CONTROLS = {
    code: ("\\r" if code == 13 else f"\\x{code:02x}")
    for code in (*range(32), *range(127, 160))
    if code not in (9, 10)
}


def safe_terminal_text(text: str) -> str:
    """Escape terminal controls, preserving tabs and ordinary line endings.

    This is idempotent, so both renderable constructors and the final Console
    boundary can use it. The caller retains the original text for storage/API
    requests; visible escapes are exclusively a presentation concern.
    """
    return text.replace("\r\n", "\n").translate(_CONTROLS)


def safe_terminal_link(target: str) -> bool:
    """OSC hyperlink targets must not contain any terminal controls."""
    return not any(ord(char) < 32 or 127 <= ord(char) < 160 for char in target)
