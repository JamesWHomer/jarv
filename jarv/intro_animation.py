"""Idle intro animation for heads-up mode.

Renders a self-contained, minimalistic welcome animation into the empty
transcript area of heads-up mode until the user sends their first message.

The public entry point :func:`render_intro` is pure: it takes the available
``width``/``height`` (in cells) plus an ``elapsed`` time in seconds and returns
a list of exactly ``height`` Rich ``Text`` rows. No global state, no I/O, no
threads -- the caller is responsible for repainting on a timer, so this adds
zero startup cost to heads-up mode.

The composition, back to front:

* a quiet, static starfield that gently twinkles across the whole canvas;
* the ``JARV`` block wordmark with a smooth colour gradient and a soft sheen
  that sweeps across it;
* a slim animated wave bar and a single pulsing hint line.
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
# Floor for the waveform's centre-weighted envelope: the edges keep this much of
# their amplitude so the wave reaches the full logo width instead of fading out.
_WAVE_EDGE_FLOOR = 0.55
# Star glyphs grouped by depth tier; all width-1 across common terminals. The
# glyph now tracks the star's magnitude (dust → mid → bright) instead of being
# chosen at random, so size reads as apparent brightness rather than noise.
_STAR_DUST = "·"
_STAR_MID = ("•", "+", "⋆")
_STAR_BRIGHT = ("*", "✦")
# Per tier: (twinkle amplitude, peak brightness factor). Dust fades fully in and
# out; bright accents only shimmer slightly so they read as steady anchor stars.
_STAR_TIERS = (
    (0.50, 0.62),  # 0 — faint dust
    (0.34, 0.86),  # 1 — mid stars
    (0.16, 1.00),  # 2 — rare bright accents
)

_HINT_WIDE = "type a message to begin   ·   /help for commands"
_HINT_NARROW = "type to begin · /help"

# Sentinel for ``render_intro(hint=...)``: keep the default heads-up hint unless a
# caller (e.g. the /setup welcome screen) overrides it. Passing ``hint=""`` draws
# the brand mark with no hint line at all.
_DEFAULT_HINT = object()

_WHITE = (236, 246, 255)

# Entrance stage windows, in seconds.
_STARS_IN = (0.0, 1.8)
_LOGO_IN = (0.0, 1.1)
_WAVE_IN = (0.7, 1.6)
_HINT_IN = (1.4, 2.2)
_WIPE_EDGE = 6.0


def _hex(r: int, g: int, b: int) -> str:
    return f"#{max(0, min(255, int(r))):02x}{max(0, min(255, int(g))):02x}{max(0, min(255, int(b))):02x}"


def _parse_hex(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


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
    """Deterministic, static star layout for a given size (cached, never mutated).

    Each star carries a twinkle phase/speed so it pulses in place; there is no
    drift, so the field stays put while individual stars flicker on and off.

    Placement is edge-biased (a soft vignette) so the centre stays sparse and
    frames the wordmark, and a small fixed number of bright accent stars are
    promoted to the top tier so they stay scarce regardless of canvas size.
    """
    rng = random.Random((width * 73856093) ^ (height * 19349663))
    count = max(6, (width * height) // 42)
    n_bright = min(6, max(3, count // 45))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    norm = math.hypot(cx, cy) or 1.0
    stars = []
    for i in range(count):
        # Reject-sample toward the edges: acceptance probability rises with
        # distance from centre, thinning the field behind the logo.
        x = y = 0
        for _ in range(8):
            x = rng.randrange(width)
            y = rng.randrange(height)
            dist = math.hypot(x - cx, y - cy) / norm
            if rng.random() <= 0.12 + 0.88 * dist ** 1.6:
                break
        phase = rng.uniform(0.0, math.tau)
        speed = rng.uniform(0.35, 0.95)
        tint = rng.uniform(0.0, 1.0)
        if i < n_bright:
            tier = 2
            char = rng.choice(_STAR_BRIGHT)
        elif rng.random() < 0.24:
            tier = 1
            char = rng.choice(_STAR_MID)
        else:
            tier = 0
            char = _STAR_DUST
        stars.append((x, y, phase, speed, char, tint, tier))
    return tuple(stars)


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


def _clear_box(chars, colors, y0: int, y1: int, x0: int, x1: int) -> None:
    width = len(chars[0])
    for y in range(max(0, y0), min(len(chars), y1)):
        for x in range(max(0, x0), min(width, x1)):
            chars[y][x] = " "
            colors[y][x] = None


def _exit_noise(x: int, y: int) -> float:
    """Stable per-cell value in [0, 1) used to stagger the dissolve order."""
    h = (x * 73856093) ^ (y * 19349663)
    h ^= h >> 13
    return (h & 0xFFFF) / 0xFFFF


def _apply_exit(chars, colors, e: float, width: int, height: int) -> None:
    """Dissolve the rendered frame as ``e`` runs 0 -> 1.

    Each painted cell holds at full brightness until a dissolve wave (advancing
    through per-cell noise space) reaches it, then fades over a short band and
    clears -- so the wordmark, wave and stars disperse organically into the dark
    rather than blinking off all at once. A mild global dim pulls the whole
    composition back as it goes, keeping the hand-off to the transcript smooth.
    """
    e = max(0.0, min(1.0, e))
    # Smoothstep so the dissolve eases in and out instead of marching linearly.
    wave = e * e * (3.0 - 2.0 * e)
    band = 0.22
    global_dim = 1.0 - 0.4 * wave
    for y in range(height):
        row_c = chars[y]
        row_col = colors[y]
        for x in range(width):
            col = row_col[x]
            if col is None:
                continue
            ahead = _exit_noise(x, y) - wave
            if ahead <= 0.0:
                row_c[x] = " "
                row_col[x] = None
                continue
            fade = min(1.0, ahead / band) * global_dim
            r, g, b = _parse_hex(col)
            row_col[x] = _hex(r * fade, g * fade, b * fade)


def _draw_starfield(chars, colors, t: float, height: int, width: int, entrance: float = 1.0) -> None:
    staggered = entrance < 1.0
    for (x, y, phase, speed, char, tint, tier) in _starfield(width, height):
        amp, peak = _STAR_TIERS[tier]
        # Twinkle around a tier-dependent baseline: faint dust swings all the way
        # to black and back (long, gentle fades), while bright accents oscillate
        # only slightly so they sit steady against the field.
        osc = (1.0 - amp) + amp * math.sin(t * speed + phase)
        val = 235.0 * peak * osc
        if staggered:
            # Each star gets a stable, position-derived appearance delay so the
            # field kindles in scattered over the entrance window instead of all
            # blinking on at once. Once its delay passes it eases up over a short
            # ramp -- so stars twinkle into existence rather than popping in.
            appear = _exit_noise(x, y) * 0.74
            ramp = max(0.0, min(1.0, (entrance - appear) / 0.26))
            val *= ramp * ramp * (3.0 - 2.0 * ramp)
        val = int(val)
        if val < 6:
            continue
        r = int(val * (0.5 + 0.28 * tint))
        g = int(val * (0.68 + 0.22 * tint))
        _place(chars, colors, y, x, char, _hex(r, g, val))


def _draw_logo(chars, colors, top: int, t: float, width: int, reveal: float) -> None:
    col_start = (width - _LOGO_W) // 2
    wipe_pos = _ease_out(reveal) * (_LOGO_W + _WIPE_EDGE)
    settled = reveal >= 1.0
    # A soft highlight that sweeps left-to-right across the wordmark.
    sheen = (t * 6.0) % (_LOGO_W + 26.0) - 13.0
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
                # Smooth, slowly drifting gradient across the width.
                hue = (abs_cx / _LOGO_W) * 0.74 + 0.55 + t * 0.045
                rgb = _hsv_rgb(hue, 0.8, 0.92)
                lead = max(0.0, 1.0 - dist / _WIPE_EDGE)
                if lead > 0:
                    rgb = _mix(rgb, _WHITE, lead * 0.9)
                elif settled:
                    s = 1.0 - abs(abs_cx - sheen) / 3.5
                    if s > 0:
                        rgb = _mix(rgb, _WHITE, s * 0.45)
                _place(chars, colors, top + ry, col_start + abs_cx, "█", _hex(*rgb))


def _draw_wave(chars, colors, y: int, t: float, width: int, grow: float) -> None:
    col_start = (width - _LOGO_W) // 2
    center = (_LOGO_W - 1) / 2.0
    half = _LOGO_W / 2.0
    reach = grow * half + 0.5
    for c in range(_LOGO_W):
        if abs(c - center) > reach:
            continue
        # Raised-cosine envelope with a floor: the scrolling sine swells under the
        # wordmark's centre but the edges keep a healthy amplitude (rather than
        # taper to an invisible baseline), so the wave fills the full logo width
        # and still reads as a centred pulse echoing the wordmark.
        env = _WAVE_EDGE_FLOOR + (1.0 - _WAVE_EDGE_FLOOR) * (
            0.5 + 0.5 * math.cos(math.pi * (c - center) / half)
        )
        level = (math.sin(c * 0.36 + t * 2.6) + 1.0) / 2.0 * env
        idx = int(level * (len(_WAVE_BLOCKS) - 1))
        hue = (c / _LOGO_W) * 0.74 + 0.55 + t * 0.045
        _place(chars, colors, y, col_start + c, _WAVE_BLOCKS[idx], _hsv_hex(hue, 0.62, 0.4 + 0.28 * level))


def _hint_color_fn(t: float):
    pulse = 0.5 + 0.5 * math.sin(t * 1.5)
    color = _mix((68, 84, 102), (148, 174, 196), pulse)

    def fn(i: int, ch: str):
        return color

    return fn


def _caret(t: float) -> str:
    return "▌" if int(t * 3) % 2 == 0 else " "


def _draw_typed_line(chars, colors, y: int, full: str, t: float, stage: float, shimmer_fn) -> None:
    center_len = len(full) + 1

    def fn(i: int, ch: str):
        return _WHITE if ch == "▌" else shimmer_fn(i, ch)

    if stage >= 1.0:
        _place_text(chars, colors, y, full, fn, center_len=center_len)
        return
    shown = full[: int(stage * len(full))]
    text = shown + _caret(t)
    _place_text(chars, colors, y, text, fn, center_len=center_len)


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


def render_intro(
    width: int,
    height: int,
    elapsed: float,
    exit: float = 0.0,
    *,
    hint=_DEFAULT_HINT,
) -> list[Text] | None:
    """Render the idle intro animation as ``height`` Rich ``Text`` rows.

    Returns ``None`` when the area is too small to draw anything meaningful,
    signalling the caller to fall back to blank padding.

    ``exit`` runs from 0 (fully present) to 1 (fully gone) to play a quick
    dissolve when the user dismisses the intro by sending their first message.

    ``hint`` overrides the single hint line beneath the wordmark so the same
    brand mark can front other screens (the /setup welcome). The default keeps
    the heads-up "type a message to begin" hint; pass ``""`` to draw no hint.
    """
    if width < 18 or height < 5:
        return None
    if exit >= 1.0:
        return [Text("") for _ in range(height)]

    t = elapsed
    chars = [[" "] * width for _ in range(height)]
    colors: list[list[str | None]] = [[None] * width for _ in range(height)]

    # Stars fill the whole canvas; the wordmark and hint are drawn on top, with
    # only their tight bounding boxes cleared so stars remain at the sides.
    _draw_starfield(chars, colors, t, height, width, _ease_out(_stage(t, _STARS_IN)))

    big = width >= _LOGO_W + 2 and height >= 11
    if hint is _DEFAULT_HINT:
        hint = _HINT_WIDE if width >= len(_HINT_WIDE) else _HINT_NARROW

    if big:
        block_h = _LOGO_H + 1 + 1 + 1 + 1  # logo, gap, wave, gap, hint
        top = max(0, (height - block_h) // 2)
        col_start = (width - _LOGO_W) // 2
        wave_y = top + _LOGO_H + 1

        # Carve a padded halo around the wordmark + waveform so the twinkling
        # field never crowds the glyph edges, then draw the brand on top.
        _clear_box(chars, colors, top - 1, wave_y + 2, col_start - 2, col_start + _LOGO_W + 2)
        _draw_logo(chars, colors, top, t, width, _stage(t, _LOGO_IN))
        _draw_wave(chars, colors, wave_y, t, width, _stage(t, _WAVE_IN))

        if hint:
            hint_y = wave_y + 2
            hx = (width - len(hint)) // 2
            _clear_box(chars, colors, hint_y, hint_y + 1, hx - 1, hx + len(hint) + 1)
            _draw_typed_line(
                chars, colors, hint_y, hint, t, _stage(t, _HINT_IN), _hint_color_fn(t),
            )
    else:
        title = "J A R V"
        block_h = 3
        top = max(0, (height - block_h) // 2)
        tx = (width - len(title)) // 2
        _clear_box(chars, colors, top, top + 1, tx - 1, tx + len(title) + 1)

        reveal = _ease_out(_stage(t, _LOGO_IN))

        def title_fn(i: int, ch: str):
            return _hsv_rgb(i / max(1, len(title)) * 0.74 + 0.55 + t * 0.045, 0.8, 0.95)

        shown = title[: max(0, int(reveal * len(title)))] if reveal < 1.0 else title
        _place_text(chars, colors, top, shown, title_fn, center_len=len(title))

        if hint:
            hint_y = top + 2
            hx = (width - len(hint)) // 2
            _clear_box(chars, colors, hint_y, hint_y + 1, hx - 1, hx + len(hint) + 1)
            _draw_typed_line(
                chars, colors, hint_y, hint, t, _stage(t, _HINT_IN), _hint_color_fn(t),
            )

    if exit > 0.0:
        _apply_exit(chars, colors, exit, width, height)

    return _rows_to_text(chars, colors, width, height)
