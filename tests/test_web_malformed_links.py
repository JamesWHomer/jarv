"""Malformed links do not discard otherwise readable remote content."""

import httpx
import pytest

from jarv.web import (
    WebToolError, _DuckDuckGoHTMLParser, _decode_search_url,
    _request_bytes, web_content_from_bytes,
)


@pytest.mark.parametrize("href", [
    "http://[broken", "//[broken", "http://example.test:invalid/",
    "//duckduckgo.com/l/?uddg=http%3A%2F%2F%5Bbroken",
])
def test_search_skips_malformed_result_urls(href):
    assert _decode_search_url(href) is None
    parser = _DuckDuckGoHTMLParser()
    parser.feed(
        f'<a class="result__a" href="{href}">Malformed result</a>'
        '<a class="result__a" href="https://example.test/good">Good result</a>'
    )
    assert parser.results == [{
        "title": "Good result", "url": "https://example.test/good", "snippet": "",
    }]


@pytest.mark.parametrize("href", ["http://[broken", "//[broken"])
def test_readable_page_preserves_text_around_malformed_links(href):
    content = web_content_from_bytes(
        "https://example.test/", "https://example.test/", "text/html",
        f'<p>Before</p><a href="{href}">Link text</a><p>After</p>'.encode(),
    )
    assert content.text == "Before\nLink text\nAfter"


def test_malformed_redirect_is_reported_as_web_error(monkeypatch):
    requests = []

    def redirect(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "http://[broken"})

    monkeypatch.setattr("jarv.web._create_client", lambda timeout: httpx.Client(
        transport=httpx.MockTransport(redirect), follow_redirects=False,
    ))
    with pytest.raises(WebToolError, match="invalid URL"):
        _request_bytes("https://example.test/", timeout=5)
    assert len(requests) == 1
