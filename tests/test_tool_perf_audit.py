"""Output-equivalence checks for bounded tool processing optimizations."""

import random

import pytest

from jarv.pdf_extract import _normalize_page_text
from jarv.shell import _BoundedOutput
from jarv.tool_outputs import flatten_content_text, parse_image_data_url


@pytest.mark.parametrize("limit", [0, 1, 2, 7, 20, 1000])
def test_capture_preserves_head_tail_across_arbitrary_chunks_and_snapshots(limit):
    rng = random.Random(312)
    capture = _BoundedOutput(limit)
    source = ""
    for index in range(150):
        chunk = "".join(rng.choices("ab\n\r界🙂", k=rng.randrange(40)))
        source += chunk
        capture.append(chunk)
        # Leave several chunks unobserved to cover deferred prefix assembly.
        if index % 7:
            continue
        prefix = source[:limit // 2]
        suffix_size = min(len(source) - len(prefix), limit - limit // 2)
        suffix = source[-suffix_size:] if suffix_size else ""
        omitted = len(source) - len(prefix) - suffix_size
        marker = (
            f"\n[command capture limit reached; {omitted} characters "
            "omitted from the middle and unavailable for read]\n"
        ) if omitted else ""
        assert capture.head == prefix
        assert capture.tail_size == suffix_size
        assert capture.total == len(source)
        assert capture.text() == prefix + marker + suffix
        assert capture.text() is capture.text()


def test_capture_head_stays_current_when_observed_before_later_appends():
    capture = _BoundedOutput(100)
    capture.append("first")
    assert capture.head == "first"
    capture.append(" second")
    assert capture.text() == "first second"
    capture.append(" third")
    assert capture.head == capture.text() == "first second third"


def test_pdf_normalization_matches_original_with_blank_and_unicode_lines():
    def original(value):
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in normalized.split("\n")]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    rng = random.Random(613)
    samples = ["", " \n\t\r\n", "\n" * 10000 + "  content \n\n"]
    samples.extend(
        "".join(rng.choices("ab 界🙂\r\n\t\v\f\u0085\u00a0\u2028", k=300))
        for _ in range(100)
    )
    for value in samples:
        assert _normalize_page_text(value) == original(value)


@pytest.mark.parametrize("url", [
    "https://example.test/image.png", "data:;base64,abc", "",
    "data:image/png;base64,", "DATA:IMAGE/PNG;BASE64,a\nb==",
    "data:image/svg+xml;base64,界🙂\r\n", "data:image/png;name=x;base64,abc",
    pytest.param("data:image/png;base64," + "a" * 2_000_000, id="large-image"),
])
def test_image_summaries_match_data_url_parser_without_changing_size_rules(url):
    parsed = parse_image_data_url(url)
    expected = (
        f"[image output 2: {parsed[0]}, {(len(parsed[1]) * 3) // 4} bytes]"
        if parsed is not None
        else "[image output 2: external or invalid image URL]"
    )
    blocks = [
        {"type": "input_text", "text": "  result  "},
        {"type": "input_image", "image_url": "invalid"},
        {"type": "input_image", "image_url": url},
    ]
    assert flatten_content_text(blocks) == (
        "result\n[image output 1: external or invalid image URL]\n" + expected
    )
