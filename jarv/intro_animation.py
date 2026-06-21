"""Idle intro animation for heads-up mode.

Renders a self-contained, colourful welcome animation into the empty
transcript area of heads-up mode until the user sends their first message.

The public entry point :func:`render_intro` is pure: it takes the available
``width``/``height`` (in cells) plus an ``elapsed`` time in seconds and returns
a list of exactly ``height`` Rich ``Text`` rows. No global state, no I/O, no
threads -- the caller is responsible for repainting on a timer, so this adds
zero startup cost to heads-up mode.
"""

from __future__ import annotations

import colorsys
import math
import random
from functools import lru_cache

from rich.style import Style
from rich.text import Text

# Block-letter wordmark. Each glyph is 6 cells wide and 5 rows tall, made of
# full-block (U+2588) cells against spaces.
_LOGO: dict[str, tuple[str, ...]] = {
    "J": (
        "  ████",
        "    ██",
        "    ██",
        "██  ██",
        " ████ ",
    ),
    "A": (
        " ████ ",
        "██  ██",
        "██████",
        "██  ██",
        "██  ██",
    ),
    "R": (
        "█████ ",
        "██  ██",
        "█████ ",
        "██ ██ ",
        "██  ██",
    ),
    "V": (
        "██  ██",
        "██  ██",
        "██  ██",
        " ████ ",
        "  ██  ",
    ),
}
_LOGO_ORDER = "JARV"
_GLYPH_W = 6
_LOGO_H = 5
_GLYPH_GAP = 2
_LOGO_W = len(_LOGO_ORDER) * _GLYPH_W + (len(_LOGO_ORDER) - 1) * _GLYPH_GAP

_WAVE_BLOCKS = "▁▂▃▄▅▆▇█"
# All width-1 across common terminals; kept deliberately ASCII-friendly.
_STAR_CHARS = ("·", "·", "·", "•", "⋆", "*", "+")

_TAGLINE = "your always-on terminal copilot"
_HINT_WIDE = "type a message to begin   ·   /help for commands"
_HINT_NARROW = "type to begin · /help"

_TAG_BASE = (86, 166, 214)
_TAG_HILITE = (236, 246, 255)


def _hex(r: int, g: int, b: int) -> str:
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


def _hsv_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return _hex(int(r * 255), int(g * 255), int(b * 255))


def _mix(c1: tuple[int, int, int], c2: tuple[int, int, int], f: float) -> str:
    f = max(0.0, min(1.0, f))
    return _hex(
        int(c1[0] + (c2[0] - c1[0]) * f),
        int(c1[1] + (c2[1] - c1[1]) * f),
        int(c1[2] + (c2[2] - c1[2]) * f),
    )


@lru_cache(maxsize=16)
def _starfield(width: int, height: int) -> tuple[tuple[int, int, float, float, str, float], ...]:
    """Deterministic star layout for a given size (cached, never mutated)."""
    rng = random.Random((width * 73856093) ^ (height * 19349663))
    count = max(6, (width * height) // 26)
    stars = []
    for _ in range(count):
        x = rng.randrange(width)
        y = rng.randrange(height)
        phase = rng.uniform(0.0, math.tau)
        speed = rng.uniform(1.2, 3.6)
        char = rng.choice(_STAR_CHARS)
        tint = rng.uniform(0.0, 1.0)
        stars.append((x, y, phase, speed, char, tint))
    return tuple(stars)


def _place(
    chars: list[list[str]],
    colors: list[list[str | None]],
    y: int,
    x: int,
    ch: str,
    color: str | None,
) -> None:
    if 0 <= y < len(chars) and 0 <= x < len(chars[0]):
        chars[y][x] = ch
        colors[y][x] = color


def _place_text(
    chars: list[list[str]],
    colors: list[list[str | None]],
    y: int,
    text: str,
    color_fn,
) -> None:
    width = len(chars[0])
    if len(text) > width:
        text = text[:width]
    x0 = (width - len(text)) // 2
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        _place(chars, colors, y, x0 + i, ch, color_fn(i, ch))


def _clear_band(chars, colors, y0: int, y1: int) -> None:
    width = len(chars[0])
    for y in range(max(0, y0), min(len(chars), y1)):
        for x in range(width):
            chars[y][x] = " "
            colors[y][x] = None


def _draw_starfield(chars, colors, t: float, height: int, width: int) -> None:
    for (x, y, phase, speed, char, tint) in _starfield(width, height):
        b = 0.5 + 0.5 * math.sin(t * speed + phase)
        if b < 0.2:
            continue
        val = int(60 + 175 * b)
        r = int(val * (0.5 + 0.28 * tint))
        g = int(val * (0.68 + 0.22 * tint))
        _place(chars, colors, y, x, char, _hex(r, g, val))


def _draw_logo(chars, colors, top: int, t: float, width: int) -> None:
    col_start = (width - _LOGO_W) // 2
    for gi, letter in enumerate(_LOGO_ORDER):
        glyph = _LOGO[letter]
        gx = gi * (_GLYPH_W + _GLYPH_GAP)
        for ry in range(_LOGO_H):
            row = glyph[ry]
            for cx in range(_GLYPH_W):
                if row[cx] == " ":
                    continue
                abs_cx = gx + cx
                hue = (abs_cx / _LOGO_W) * 0.85 + t * 0.13 + ry * 0.02
                val = 0.78 + 0.22 * math.sin(t * 2.6 + abs_cx * 0.22)
                _place(
                    chars,
                    colors,
                    top + ry,
                    col_start + abs_cx,
                    "█",
                    _hsv_hex(hue, 0.88, val),
                )


def _draw_wave(chars, colors, y: int, t: float, width: int) -> None:
    col_start = (width - _LOGO_W) // 2
    for c in range(_LOGO_W):
        level = (math.sin(c * 0.36 + t * 3.3) + 1.0) / 2.0
        idx = int(level * (len(_WAVE_BLOCKS) - 1))
        hue = (c / _LOGO_W) * 0.85 + t * 0.13
        _place(
            chars,
            colors,
            y,
            col_start + c,
            _WAVE_BLOCKS[idx],
            _hsv_hex(hue, 0.7, 0.5 + 0.32 * level),
        )


def _tag_color_fn(t: float, length: int):
    hl = (t * 9.0) % (length + 16) - 8.0

    def fn(i: int, ch: str) -> str:
        return _mix(_TAG_BASE, _TAG_HILITE, 1.0 - abs(i - hl) / 5.0)

    return fn


def _hint_color_fn(t: float):
    pulse = 0.5 + 0.5 * math.sin(t * 1.6)
    color = _mix((70, 86, 104), (150, 176, 198), pulse)

    def fn(i: int, ch: str) -> str:
        return color

    return fn


def _rows_to_text(chars, colors, width: int, height: int) -> list[Text]:
    lines: list[Text] = []
    for y in range(height):
        line = Text(no_wrap=True, overflow="crop")
        row_c = chars[y]
        row_col = colors[y]
        i = 0
        while i < width:
            col = row_col[i]
            j = i
            while j < width and row_col[j] == col:
                j += 1
            seg = "".join(row_c[i:j])
            if col is None:
                line.append(seg)
            else:
                line.append(seg, style=Style(color=col))
            i = j
        lines.append(line)
    return lines


def render_intro(width: int, height: int, elapsed: float) -> list[Text] | None:
    """Render the idle intro animation as ``height`` Rich ``Text`` rows.

    Returns ``None`` when the area is too small to draw anything meaningful,
    signalling the caller to fall back to blank padding.
    """
    if width < 18 or height < 5:
        return None

    t = elapsed
    chars = [[" "] * width for _ in range(height)]
    colors: list[list[str | None]] = [[None] * width for _ in range(height)]

    _draw_starfield(chars, colors, t, height, width)

    big = width >= _LOGO_W + 2 and height >= 11
    hint = _HINT_WIDE if width >= len(_HINT_WIDE) else _HINT_NARROW

    if big:
        block_h = _LOGO_H + 1 + 1 + 1 + 1 + 1 + 1  # logo, gap, wave, gap, tag, gap, hint
        top = max(0, (height - block_h) // 2)
        _clear_band(chars, colors, top - 1, top + block_h + 1)
        _draw_logo(chars, colors, top, t, width)
        wave_y = top + _LOGO_H + 1
        _draw_wave(chars, colors, wave_y, t, width)
        tag_y = wave_y + 2
        _place_text(chars, colors, tag_y, _TAGLINE, _tag_color_fn(t, len(_TAGLINE)))
        _place_text(chars, colors, tag_y + 2, hint, _hint_color_fn(t))
    else:
        title = "J A R V"
        block_h = 1 + 1 + 1 + 1 + 1
        top = max(0, (height - block_h) // 2)
        _clear_band(chars, colors, top - 1, top + block_h + 1)

        def title_fn(i: int, ch: str) -> str:
            return _hsv_hex(i / max(1, len(title)) * 0.8 + t * 0.2, 0.9, 0.95)

        _place_text(chars, colors, top, title, title_fn)
        _place_text(chars, colors, top + 2, _TAGLINE, _tag_color_fn(t, len(_TAGLINE)))
        _place_text(chars, colors, top + 4, hint, _hint_color_fn(t))

    return _rows_to_text(chars, colors, width, height)
