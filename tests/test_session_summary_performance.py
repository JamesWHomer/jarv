"""Read-card summaries preserve header semantics without scanning body text."""

import pytest

from jarv import session_render


@pytest.mark.parametrize("separator", ["\n\n", "\r\n\r\n", "\n\r\n", "\r\n\n"])
def test_read_summary_sanitizes_only_header(monkeypatch, separator):
    header = "[READ RESULT]\nReturned size: 120\nTotal size: 240\nEOF: false"
    body = "\x1b[31mprivate read content\n" * 8_000
    sanitized = []
    original = session_render.safe_terminal_text

    def observe(text):
        sanitized.append(text)
        return original(text)

    monkeypatch.setattr(session_render, "safe_terminal_text", observe)

    assert session_render._read_result_summary(header + separator + body) == (
        "120 of 240 chars  •  more available"
    )
    assert sanitized == [header]


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Returned size: 120\nEOF: true", "120 chars  •  EOF"),
        ("Returned size: 120\nEOF: false", "120 chars  •  EOF"),
        ("Returned size: invalid\nTotal size: 240", ""),
        ("Image media type: image/png\nImage bytes: 2048", "image image/png  •  2 KB"),
        ("Image media type: \x1b[31mimage/png", "image \\x1b[31mimage/png"),
        ("Returned size: 120\n\t \nReturned size: 999", "120 chars  •  EOF"),
        ("Returned size: 120\u2028\u2028Returned size: 999", "120 chars  •  EOF"),
        ("Returned size: 120\n\x0b\nTotal size: 240", "120 of 240 chars  •  more available"),
    ],
)
def test_read_summary_preserves_missing_delimiters_and_unusual_headers(header, expected):
    output = "[READ RESULT]\n" + header
    assert session_render._read_result_summary(output) == expected
    # Content cannot replace header values after the ordinary blank line.
    assert session_render._read_result_summary(output + "\n\nReturned size: 999") == expected
