"""Small layout primitives shared by fullscreen TUI views."""

from __future__ import annotations

from typing import Any


def clip_text(value: str, width: int, *, ellipsis: str = "...") -> str:
    if width <= 0:
        return ""
    if len(value) <= width:
        return value
    if width <= len(ellipsis):
        return value[:width]
    return value[: width - len(ellipsis)] + ellipsis


def append_bottom_footer(
    parts: list[Any],
    height: int,
    footer: Any,
    *,
    border_rows: int = 2,
    footer_rows: int = 2,
    crop: bool = False,
) -> None:
    target_rows_before_footer = max(0, height - border_rows - footer_rows)
    if crop and len(parts) > target_rows_before_footer:
        del parts[target_rows_before_footer:]
    while len(parts) < target_rows_before_footer:
        parts.append("")
    parts.append("")
    parts.append(footer)
