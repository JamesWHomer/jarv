import time
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest

from jarv.artifacts import ArtifactStore
from jarv.agent import TOOLS
from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.config import DEFAULT_CONFIG
from jarv.orchestrator import (
    AgentNode,
    build_subagent_tools,
    dispatch_parallel_safe_tool_batch,
    dispatch_tool,
)
from jarv.read_tool import READ_TOOL, dispatch_read_tool
from jarv.retained_outputs import RetainedOutputStore
from jarv.web import (
    MAX_RESPONSE_BYTES,
    WEB_SEARCH_TOOL,
    WebToolError,
    _DuckDuckGoHTMLParser,
    _ReadableHTMLParser,
    _decode_search_url,
    dispatch_web_tool,
    fetch_web,
    search_web,
)


def test_default_prompt_and_tool_descriptions_guide_parallel_search_reads():
    assert (
        "When several tool calls are independent, issue them in the same response "
        "instead of one tool call per turn."
    ) in DEFAULT_CONFIG["system_prompt"]
    assert "does not read page contents" in WEB_SEARCH_TOOL["description"]
    assert "Read or fetch a known" in READ_TOOL["description"]


def _mock_client(handler):
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )


def test_duckduckgo_parser_decodes_results_snippets_and_ignores_ads():
    html = """
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs&amp;rut=x">
      Example <b>Docs</b>
    </a>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs">
      Useful &amp; current.
    </a>
    <a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=example.test">Ad</a>
    """
    parser = _DuckDuckGoHTMLParser()
    parser.feed(html)

    assert parser.results == [{
        "title": "Example Docs",
        "url": "https://example.com/docs",
        "snippet": "Useful & current.",
    }]


def test_decode_search_url_rejects_non_http_destinations():
    assert _decode_search_url("javascript:alert(1)") is None
    assert _decode_search_url(
        "//duckduckgo.com/l/?uddg=file%3A%2F%2F%2Fsecret"
    ) is None


def test_readable_html_prefers_main_and_omits_scripts():
    parser = _ReadableHTMLParser()
    parser.feed("""
        <html><head><title>Page Title</title><script>bad()</script></head>
        <body><nav>Navigation</nav><main><h1>Hello</h1><p>Useful text.</p></main></body>
        </html>
    """)

    assert parser.title() == "Page Title"
    assert parser.readable_text() == "Hello\n\nUseful text."


def test_readable_html_preserves_absolute_and_relative_link_urls():
    parser = _ReadableHTMLParser("https://example.test/base/page")
    parser.feed("""
        <main><p>
          Read <a href="/docs?q=1#part">the docs</a>
          and <a href="javascript:bad()">ignore this scheme</a>.
        </p></main>
    """)

    assert parser.readable_text() == (
        "Read the docs <https://example.test/docs?q=1#part> "
        "and ignore this scheme."
    )


def test_search_web_formats_unique_results(monkeypatch):
    body = b"""
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">One</a>
    <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">First</a>
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">Duplicate</a>
    <a class="result__a" href="https://two.example/path">Two</a>
    """

    def handler(request):
        assert request.url.params["q"] == "query"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=body,
        )

    monkeypatch.setattr(
        "jarv.web._create_client",
        lambda _timeout: _mock_client(handler),
    )

    output = search_web("query", 2, timeout=5)

    assert "UNTRUSTED WEB SEARCH RESULTS" in output
    assert "1. One" in output
    assert "Snippet: First" in output
    assert "2. Two" in output
    assert "Duplicate" not in output


def test_search_web_follows_next_form_and_applies_offset(monkeypatch):
    first_page = b"""
    <a class="result__a" href="https://one.example">One</a>
    <a class="result__a" href="https://two.example">Two</a>
    <form class="nav-link" action="/html/" method="post">
      <input type="submit" value="Next">
      <input type="hidden" name="q" value="query">
      <input type="hidden" name="s" value="10">
      <input type="hidden" name="vqd" value="token">
    </form>
    """
    second_page = b"""
    <a class="result__a" href="https://three.example">Three</a>
    <a class="result__a" href="https://four.example">Four</a>
    """
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=first_page,
            )
        assert parse_qs(request.content.decode()) == {
            "q": ["query"],
            "s": ["10"],
            "vqd": ["token"],
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=second_page,
        )

    monkeypatch.setattr(
        "jarv.web._create_client",
        lambda _timeout: _mock_client(handler),
    )

    output = search_web("query", 2, offset=1, timeout=5)

    assert [request.method for request in requests] == ["GET", "POST"]
    assert "Offset: 1" in output
    assert "2. Two" in output
    assert "3. Three" in output
    assert "1. One" not in output
    assert "4. Four" not in output


def test_fetch_web_follows_redirect_and_extracts_html(monkeypatch):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/final"})
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<title>Test</title><main><h1>Heading</h1><p>Body</p></main>",
        )

    monkeypatch.setattr(
        "jarv.web._create_client",
        lambda _timeout: _mock_client(handler),
    )

    output = fetch_web("http://localhost:8123/start", timeout=5)

    assert seen == [
        "http://localhost:8123/start",
        "http://localhost:8123/final",
    ]
    assert "Requested URL: http://localhost:8123/start" in output
    assert "Final URL: http://localhost:8123/final" in output
    assert "Title: Test" in output
    assert output.endswith("Heading\n\nBody")


def test_fetch_web_formats_json(monkeypatch):
    def handler(_request):
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"ok": True},
        )

    monkeypatch.setattr(
        "jarv.web._create_client",
        lambda _timeout: _mock_client(handler),
    )

    output = fetch_web("https://example.test/data", timeout=5)

    assert '"ok": true' in output


def test_fetch_web_rejects_credentials_and_non_http():
    assert "embedded URL credentials" in dispatch_read_tool(
        {"input": "https://user:pass@example.test/"},
        visible_labels=set(),
        artifact_store=ArtifactStore(),
        retained_store=RetainedOutputStore(),
        config=DEFAULT_CONFIG,
    )
    assert "unsupported URL scheme" in dispatch_read_tool(
        {"input": "file:///tmp/secret"},
        visible_labels=set(),
        artifact_store=ArtifactStore(),
        retained_store=RetainedOutputStore(),
        config=DEFAULT_CONFIG,
    )


def test_fetch_web_rejects_binary_and_oversized_responses(monkeypatch):
    responses = [
        httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF",
        ),
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-length": str(MAX_RESPONSE_BYTES + 1),
            },
            content=b"x",
        ),
    ]

    def handler(_request):
        return responses.pop(0)

    monkeypatch.setattr(
        "jarv.web._create_client",
        lambda _timeout: _mock_client(handler),
    )

    with pytest.raises(WebToolError, match="unsupported non-text"):
        fetch_web("https://example.test/file.pdf", timeout=5)
    assert "exceeds" in dispatch_read_tool(
        {"input": "https://example.test/large"},
        visible_labels=set(),
        artifact_store=ArtifactStore(),
        retained_store=RetainedOutputStore(),
        config=DEFAULT_CONFIG,
    )


def test_web_dispatch_keeps_requested_count_and_validates_offset(monkeypatch):
    captured = {}

    def fake_search(query, max_results, **_kwargs):
        captured.update(
            query=query,
            max_results=max_results,
            offset=_kwargs["offset"],
        )
        return "ok"

    monkeypatch.setattr("jarv.web.search_web", fake_search)

    assert dispatch_web_tool(
        "web_search",
        {"query": "test", "max_results": 100, "offset": 25},
        DEFAULT_CONFIG,
    ) == "ok"
    assert captured == {"query": "test", "max_results": 100, "offset": 25}
    assert "max_results must be an integer" in dispatch_web_tool(
        "web_search",
        {"query": "test", "max_results": "5"},
        DEFAULT_CONFIG,
    )
    assert "positive integer" in dispatch_web_tool(
        "web_search",
        {"query": "test", "max_results": 0},
        DEFAULT_CONFIG,
    )
    assert "non-negative integer" in dispatch_web_tool(
        "web_search",
        {"query": "test", "offset": -1},
        DEFAULT_CONFIG,
    )


def test_pre_cancelled_web_request_propagates_cancellation(monkeypatch):
    token = CancellationToken()
    token.cancel()
    monkeypatch.setattr(
        "jarv.web._create_client",
        lambda _timeout: _mock_client(lambda _request: httpx.Response(200)),
    )

    with pytest.raises(TurnCancelled):
        search_web("query", timeout=5, cancellation_token=token)


def test_transport_error_after_cancellation_propagates_cancellation(monkeypatch):
    token = CancellationToken()

    def handler(request):
        token.cancel()
        raise httpx.ReadError("closed", request=request)

    monkeypatch.setattr(
        "jarv.web._create_client",
        lambda _timeout: _mock_client(handler),
    )

    with pytest.raises(TurnCancelled):
        search_web("query", timeout=5, cancellation_token=token)


def test_subagents_receive_and_dispatch_web_tools(monkeypatch):
    root_names = {tool["name"] for tool in TOOLS}
    assert {"web_search", "read"} <= root_names
    assert {"web_fetch", "read_artifact"}.isdisjoint(root_names)

    names = {tool["name"] for tool in build_subagent_tools(sterile=True)}
    assert {"web_search", "read"} <= names
    assert {"web_fetch", "read_artifact"}.isdisjoint(names)

    monkeypatch.setattr(
        "jarv.orchestrator.dispatch_web_tool",
        lambda *args, **kwargs: "web",
    )
    output = dispatch_tool(
        "web_search",
        {"query": "test"},
        AgentNode("child", 1, "root", "task", True),
        ArtifactStore(),
        client=None,
        config=DEFAULT_CONFIG,
    )
    assert output == "web"


def test_parallel_safe_batch_runs_mixed_tools_concurrently_and_preserves_order(
    monkeypatch,
):
    def fake_web(_name, args, *_pos, **_kwargs):
        time.sleep(0.1)
        return "web:" + args["query"]

    def fake_read(args, **_kwargs):
        time.sleep(0.1)
        return "read:" + args["input"]

    monkeypatch.setattr("jarv.orchestrator.dispatch_web_tool", fake_web)
    monkeypatch.setattr("jarv.orchestrator.dispatch_read_tool", fake_read)

    calls = [
        SimpleNamespace(
            name="web_search",
            arguments='{"query": "first"}',
        ),
        SimpleNamespace(
            name="read",
            arguments='{"input": "second"}',
        ),
    ]
    started = time.perf_counter()
    results = dispatch_parallel_safe_tool_batch(
        calls,
        node=AgentNode("root", 0, None, "task", False),
        store=ArtifactStore(),
        client=None,
        config=DEFAULT_CONFIG,
        retained_store=RetainedOutputStore(),
    )
    elapsed = time.perf_counter() - started

    assert [result.output for result in results] == ["web:first", "read:second"]
    assert elapsed < 0.18
