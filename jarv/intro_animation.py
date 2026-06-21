"""Idle intro animation for heads-up mode.

Renders a self-contained, colourful welcome animation into the empty
transcript area of heads-up mode until the user sends their first message.

The public entry point :func:`render_intro` is pure: it takes the available
``width``/``height`` (in cells) plus an ``elapsed`` time in seconds and returns
a list of exactly ``height`` Rich ``Text`` rows. No global state, no I/O, no
threads -- the caller is responsible for repainting on a timer, so this adds
zero startup cost to heads-up mode.

The composition, back to front:

* a parallax, drifting, twinkling starfield;
* occasional shooting stars that streak past and dive behind the wordmark;
* a staged entrance (logo wipes in -> wave grows -> tagline/hint type in);
* the ``JARV`` block wordmark with a sweeping rainbow gradient, a soft glow
  aura, and intermittent sparkle glints;
* an animated equalizer wave bar, a shimmering tagline, and a pulsing hint.
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
_GLOW_CHAR = "░"

_TAGLINE = "your always-on terminal copilot"
_HINT_WIDE = "type a message to begin   ·   /help for commands"
_HINT_NARROW = "type to begin · /help"

_TAG_BASE = (86, 166, 214)
_TAG_HILITE = (236, 246, 255)
_TYPE_COLOR = (120, 200, 235)
_CARET_COLOR = (236, 246, 255)
_WHITE = (236, 246, 255)

# Entrance stage windows, in seconds.
_LOGO_IN = (0.0, 1.1)
_WAVE_IN = (0.7, 1.5)
_TAG_IN = (1.35, 2.15)
_HINT_IN = (1.9, 2.7)
_WIPE_EDGE = 6.0

_N_COMETS = 3


def _hex(r: int, g: int, b: int) -> str:
    return f"#{max(0, min(255, int(r))):02x}{max(0, min(255, int(g))):02x}{max(0, min(255, int(b))):02x}"


def _hsv_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _hsv_hex(h: float, s: float, v: float) -> str:
    r, g, b = _hsv_rgb(h, s, v)
    return _hex(r, g, b)


def _mix(c1: tuple[int, int, int], c2: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    f = max(0.0, min(1.0, f))
    return (
        int(c1[0] + (c2[0] - c1[0]) * f),
        int(c1[1] + (c2[1] - c1[1]) * f),
        int(c1[2] + (c2[2] - c1[2]) * f),
    )


def _ease_out(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1.0 - (1.0 - x) ** 3


def _stage(t: float, span: tuple[float, float]) -> float:
    a, b = span
    if b <= a:
        return 1.0
    return max(0.0, min(1.0, (t - a) / (b - a)))


@lru_cache(maxsize=16)
def _starfield(width: int, height: int):
    """Deterministic star layout for a given size (cached, never mutated).

    Each star carries a ``depth`` in ``[0, 1]`` driving parallax drift speed and
    brightness, so nearer stars sweep faster and shine brighter.
    """
    rng = random.Random((width * 73856093) ^ (height * 19349663))
    count = max(6, (width * height) // 24)
    stars = []
    for _ in range(count):
        x = rng.uniform(0, width)
        y = rng.randrange(height)
        phase = rng.uniform(0.0, math.tau)
        speed = rng.uniform(1.2, 3.6)
        depth = rng.random() ** 1.5
        char = rng.choice(_STAR_CHARS)
        tint = rng.uniform(0.0, 1.0)
        stars.append((x, y, phase, speed, depth, char, tint))
    return tuple(stars)


@lru_cache(maxsize=16)
def _comets(width: int, height: int):
    """Deterministic shooting-star schedule for a given size."""
    rng = random.Random((width * 2654435761) ^ (height * 40503))
    comets = []
    for _ in range(_N_COMETS):
        y_start = rng.randrange(0, max(1, height // 2))
        y_end = y_start + rng.randrange(max(2, height // 3), max(3, height))
        period = rng.uniform(7.0, 13.0)
        offset = rng.uniform(0.0, 1.0)
        active = rng.uniform(0.1, 0.18)
        length = rng.randint(4, 7)
        comets.append((y_start, y_end, period, offset, active, length))
    return tuple(comets)


def _place(chars, colors, y: int, x: int, ch: str, color: str | None) -> None:
    if 0 <= y < len(chars) and 0 <= x < len(chars[0]):
        chars[y][x] = ch
        colors[y][x] = color


def _place_text(chars, colors, y: int, text: str, color_fn, *, center_len: int | None = None) -> None:
    width = len(chars[0])
    span = center_len if center_len is not None else len(text)
    x0 = (width - span) // 2
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        col = color_fn(i, ch)
        _place(chars, colors, y, x0 + i, ch, _hex(*col) if isinstance(col, tuple) else col)


def _clear_band(chars, colors, y0: int, y1: int) -> None:
    width = len(chars[0])
    for y in range(max(0, y0), min(len(chars), y1)):
        for x in range(width):
            chars[y][x] = " "
            colors[y][x] = None


def _draw_starfield(chars, colors, t: float, height: int, width: int) -> None:
    for (x0, y, phase, speed, depth, char, tint) in _starfield(width, height):
        twinkle = 0.5 + 0.5 * math.sin(t * speed + phase)
        b = twinkle * (0.4 + 0.6 * depth)
        if b < 0.16:
            continue
        drift = 0.4 + depth * 2.4
        x = int((x0 - t * drift) % width)
        val = int(55 + 185 * b)
        r = int(val * (0.5 + 0.28 * tint))
        g = int(val * (0.68 + 0.20 * tint))
        _place(chars, colors, y, x, char, _hex(r, g, val))


def _draw_comets(chars, colors, t: float, height: int, width: int) -> None:
    span = width + 12
    for (y_start, y_end, period, offset, active, length) in _comets(width, height):
        phase = ((t / period) + offset) % 1.0
        if phase >= active:
            continue
        lp = phase / active
        hx = -length + lp * span
        hy = y_start + lp * (y_end - y_start)
        dx = span
        dy = y_end - y_start
        mag = math.hypot(dx, dy) or 1.0
        ux, uy = dx / mag, dy / mag
        for i in range(length):
            px = int(round(hx - ux * i))
            py = int(round(hy - uy * i))
            f = 1.0 - i / length
            color = _mix((36, 54, 86), _WHITE, f * f)
            ch = "*" if i == 0 else ("·" if i > length // 2 else "•")
            _place(chars, colors, py, px, ch, _hex(*color))


def _draw_logo(chars, colors, top: int, t: float, width: int, reveal: float) -> None:
    col_start = (width - _LOGO_W) // 2
    wipe_pos = _ease_out(reveal) * (_LOGO_W + _WIPE_EDGE)
    settled = reveal >= 1.0
    for gi, letter in enumerate(_LOGO_ORDER):
        glyph = _LOGO[letter]
        gx = gi * (_GLYPH_W + _GLYPH_GAP)
        for ry in range(_LOGO_H):
            row = glyph[ry]
            for cx in range(_GLYPH_W):
                if row[cx] == " ":
                    continue
                abs_cx = gx + cx
                dist = wipe_pos - abs_cx
                if dist <= 0:
                    continue
                hue = (abs_cx / _LOGO_W) * 0.85 + t * 0.13 + ry * 0.02
                val = 0.78 + 0.22 * math.sin(t * 2.6 + abs_cx * 0.22)
                rgb = _hsv_rgb(hue, 0.88, val)
                lead = max(0.0, 1.0 - dist / _WIPE_EDGE)
                if lead > 0:
                    rgb = _mix(rgb, _WHITE, lead * 0.85)
                elif settled:
                    glint = math.sin(t * 3.0 + abs_cx * 12.9 + ry * 7.3)
                    if glint > 0.972:
                        rgb = _mix(rgb, _WHITE, (glint - 0.972) / 0.028)
                _place(chars, colors, top + ry, col_start + abs_cx, "█", _hex(*rgb))


def _draw_glow(chars, colors, top: int, t: float, width: int) -> None:
    col_start = (width - _LOGO_W) // 2
    y0, y1 = top - 1, top + _LOGO_H
    x0, x1 = col_start - 1, col_start + _LOGO_W
    for y in range(y0, y1 + 1):
        if not (0 <= y < len(chars)):
            continue
        for x in range(x0, x1 + 1):
            if not (0 <= x < width) or chars[y][x] != " ":
                continue
            near = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < len(chars) and 0 <= nx < width and chars[ny][nx] == "█":
                        near = True
                        break
                if near:
                    break
            if near:
                hue = ((x - col_start) / _LOGO_W) * 0.85 + t * 0.13
                pulse = 0.16 + 0.07 * math.sin(t * 2.2 + x * 0.3)
                _place(chars, colors, y, x, _GLOW_CHAR, _hsv_hex(hue, 0.65, pulse + 0.12))


def _draw_wave(chars, colors, y: int, t: float, width: int, grow: float) -> None:
    col_start = (width - _LOGO_W) // 2
    center = (_LOGO_W - 1) / 2.0
    reach = grow * (_LOGO_W / 2.0) + 0.5
    for c in range(_LOGO_W):
        if abs(c - center) > reach:
            continue
        level = (math.sin(c * 0.36 + t * 3.3) + 1.0) / 2.0
        idx = int(level * (len(_WAVE_BLOCKS) - 1))
        hue = (c / _LOGO_W) * 0.85 + t * 0.13
        _place(chars, colors, y, col_start + c, _WAVE_BLOCKS[idx], _hsv_hex(hue, 0.7, 0.5 + 0.32 * level))


def _tag_color_fn(t: float, length: int):
    hl = (t * 9.0) % (length + 16) - 8.0

    def fn(i: int, ch: str):
        return _mix(_TAG_BASE, _TAG_HILITE, 1.0 - abs(i - hl) / 5.0)

    return fn


def _hint_color_fn(t: float):
    pulse = 0.5 + 0.5 * math.sin(t * 1.6)
    color = _mix((70, 86, 104), (150, 176, 198), pulse)

    def fn(i: int, ch: str):
        return color

    return fn


def _caret(t: float) -> str:
    return "▌" if int(t * 3) % 2 == 0 else " "


def _draw_typed_line(chars, colors, y: int, full: str, t: float, stage: float, base_color, shimmer_fn) -> None:
    if stage >= 1.0:
        _place_text(chars, colors, y, full, shimmer_fn)
        return
    shown = full[: int(stage * len(full))]
    text = shown + _caret(t)

    def fn(i: int, ch: str):
        return _CARET_COLOR if ch == "▌" else base_color

    _place_text(chars, colors, y, text, fn, center_len=len(full) + 1)


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
    _draw_comets(chars, colors, t, height, width)

    big = width >= _LOGO_W + 2 and height >= 11
    hint = _HINT_WIDE if width >= len(_HINT_WIDE) else _HINT_NARROW

    if big:
        block_h = _LOGO_H + 1 + 1 + 1 + 1 + 1 + 1  # logo, gap, wave, gap, tag, gap, hint
        top = max(0, (height - block_h) // 2)
        _clear_band(chars, colors, top - 1, top + block_h + 1)

        _draw_logo(chars, colors, top, t, width, _stage(t, _LOGO_IN))
        _draw_glow(chars, colors, top, t, width)

        wave_y = top + _LOGO_H + 1
        _draw_wave(chars, colors, wave_y, t, width, _stage(t, _WAVE_IN))

        tag_y = wave_y + 2
        _draw_typed_line(
            chars, colors, tag_y, _TAGLINE, t, _stage(t, _TAG_IN),
            _TYPE_COLOR, _tag_color_fn(t, len(_TAGLINE)),
        )
        _draw_typed_line(
            chars, colors, tag_y + 2, hint, t, _stage(t, _HINT_IN),
            (110, 140, 165), _hint_color_fn(t),
        )
    else:
        title = "J A R V"
        block_h = 5
        top = max(0, (height - block_h) // 2)
        _clear_band(chars, colors, top - 1, top + block_h + 1)

        reveal = _ease_out(_stage(t, _LOGO_IN))

        def title_fn(i: int, ch: str):
            return _hsv_rgb(i / max(1, len(title)) * 0.8 + t * 0.2, 0.9, 0.95)

        shown = title[: max(0, int(reveal * len(title)))] if reveal < 1.0 else title
        _place_text(chars, colors, top, shown, title_fn, center_len=len(title))
        _draw_typed_line(
            chars, colors, top + 2, _TAGLINE, t, _stage(t, _TAG_IN),
            _TYPE_COLOR, _tag_color_fn(t, len(_TAGLINE)),
        )
        _draw_typed_line(
            chars, colors, top + 4, hint, t, _stage(t, _HINT_IN),
            (110, 140, 165), _hint_color_fn(t),
        )

    return _rows_to_text(chars, colors, width, height)
