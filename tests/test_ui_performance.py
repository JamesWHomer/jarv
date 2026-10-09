"""Output equivalence and bounded work for large TUI content."""

import pytest
from rich.text import Text

from jarv import intro_animation, settings_command
from jarv.headsup import TranscriptEntry
from jarv.tui_frame import window_transcript


@pytest.mark.parametrize("spacer", [False, True])
@pytest.mark.parametrize("scroll_offset", [0, 31, 9997, 100_000])
def test_cached_transcript_copies_only_visible_rows(
    headsup_app_factory, spacer, scroll_offset,
):
    class VisibleSlicesOnly(list):
        def __iter__(self):
            raise AssertionError("cached offscreen rows must not be iterated")

        def __getitem__(self, index):
            assert isinstance(index, slice)
            start, end, step = index.indices(len(self))
            assert step == 1 and end - start <= 6
            return super().__getitem__(index)

    source = [Text(str(index)) for index in range(10_000)]
    entry = TranscriptEntry("assistant", Text(""), spacer_before=spacer)
    entry._render_cache_width = 80
    entry._render_cache_lines = VisibleSlicesOnly(source)
    app = headsup_app_factory()
    app.entries = [entry]
    app._notice = None
    app._follow_latest()
    full = ([Text("")] if spacer else []) + source

    expected, expected_offset = window_transcript(full, 6, scroll_offset)
    actual, offset = app._transcript_window(80, 6, scroll_offset)

    assert actual == expected
    assert offset == expected_offset
    if scroll_offset:
        assert app._scroll_anchor == (0, len(full) - offset - 1)
        # A subsequent paint must preserve the detached reading anchor.
        assert app._transcript_window(80, 6, offset) == (expected, expected_offset)


@pytest.mark.parametrize("discard_armed", [False, True])
@pytest.mark.parametrize("max_lines", [0, 1, 3, 8, 20])
def test_multiline_settings_only_style_the_visible_draft(
    monkeypatch, discard_armed, max_lines,
):
    edit = {
        "buffer": "界 é line\n" * 10_000,
        "cursor": 90_000,
        "discard_armed": discard_armed,
    }
    def unexpected_full_render(*args, **kwargs):
        raise AssertionError("a bounded editor must not style the entire draft")

    monkeypatch.setattr(
        settings_command, "_settings_multiline_visual_lines", unexpected_full_render,
    )
    actual = settings_command._settings_multiline_editor_lines(
        edit, 80, max_lines=max_lines,
    )
    assert len(actual) == max_lines
    if max_lines >= 8:
        assert "Enter newline" in actual[-1].plain
        assert any("界 é line" in line.plain for line in actual)


def _reference_logo(chars, colors, top, t, width, reveal):
    """Pre-optimization per-cell renderer, used as a frame/color oracle."""
    intro = intro_animation
    col_start = (width - intro._LOGO_W) // 2
    wipe_pos = intro._ease_out(reveal) * (intro._LOGO_W + intro._WIPE_EDGE)
    sheen = (t * 6.0) % (intro._LOGO_W + 26.0) - 13.0
    for gi, letter in enumerate(intro._LOGO_ORDER):
        glyph = intro._LOGO[letter]
        gx = gi * (intro._GLYPH_W + intro._GLYPH_GAP)
        for ry, row in enumerate(glyph):
            for cx, cell in enumerate(row):
                if cell == " ":
                    continue
                abs_cx = gx + cx
                dist = wipe_pos - abs_cx
                if dist <= 0:
                    continue
                hue = (abs_cx / intro._LOGO_W) * 0.74 + 0.55 + t * 0.045
                rgb = intro._hsv_rgb(hue, 0.8, 0.92)
                lead = max(0.0, 1.0 - dist / intro._WIPE_EDGE)
                if lead > 0:
                    rgb = intro._mix(rgb, intro._WHITE, lead * 0.9)
                elif reveal >= 1.0:
                    s = 1.0 - abs(abs_cx - sheen) / 3.5
                    if s > 0:
                        rgb = intro._mix(rgb, intro._WHITE, s * 0.45)
                intro._place(chars, colors, top + ry, col_start + abs_cx, "█", intro._hex(*rgb))


@pytest.mark.parametrize("width,height", [(32, 11), (81, 14), (140, 40)])
@pytest.mark.parametrize("elapsed", [0, 0.2, 0.55, 0.9, 1.1, 2.5, 10, 123])
@pytest.mark.parametrize("exit", [0.0, 0.3, 0.8])
def test_intro_preserves_every_character_and_color(monkeypatch, width, height, elapsed, exit):
    actual = intro_animation.render_intro(width, height, elapsed, exit)
    monkeypatch.setattr(intro_animation, "_draw_logo", _reference_logo)
    expected = intro_animation.render_intro(width, height, elapsed, exit)

    assert actual == expected


def test_intro_computes_logo_gradient_once_per_column(monkeypatch):
    calls = 0
    original = intro_animation._hsv_rgb

    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(intro_animation, "_hsv_rgb", counted)
    chars = [[" "] * 80 for _ in range(14)]
    colors = [[None] * 80 for _ in range(14)]
    intro_animation._draw_logo(chars, colors, 2, 10, 80, 1.0)

    assert calls == len(intro_animation._LOGO_ORDER) * intro_animation._GLYPH_W
    assert sum(cell != " " for row in chars for cell in row) > calls
