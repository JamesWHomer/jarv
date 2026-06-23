from jarv.intro_animation import render_intro


def _hint_position_and_color(width: int, height: int, elapsed: float):
    rows = render_intro(width, height, elapsed)
    assert rows is not None
    for row in rows:
        pos = row.plain.find("type a message")
        if pos < 0:
            continue
        for span in row.spans:
            if span.start <= pos < span.end:
                triplet = span.style.color.triplet
                return pos, (triplet.red, triplet.green, triplet.blue)
    raise AssertionError("hint line not rendered")


def test_hint_completion_does_not_shift_or_dim():
    typing_pos, typing_color = _hint_position_and_color(81, 14, 2.19)
    complete_pos, complete_color = _hint_position_and_color(81, 14, 2.21)

    assert complete_pos == typing_pos
    assert max(abs(a - b) for a, b in zip(typing_color, complete_color)) <= 2
