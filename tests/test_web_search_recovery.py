"""Search recovery must distinguish empty results from provider failures."""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import threading
from time import monotonic

import httpx
import pytest

from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.search_control import SearchCoordinator
from jarv.web import WebToolError, search_web
from jarv import web
from jarv.config import DEFAULT_CONFIG
from jarv.tool_outputs import tool_outcome


def _page(title="One", *, next_offset=None):
    body = f'<a class="result__a" href="https://example.test/{title}">{title}</a>'
    if next_offset is not None:
        body += f"""
        <form class="nav-link" action="/html/" method="post">
          <input type="submit" value="Next">
          <input type="hidden" name="q" value="query">
          <input type="hidden" name="s" value="{next_offset}">
        </form>
        """
    return body


def _response(body, status=200, **headers):
    return httpx.Response(
        status,
        headers={"content-type": "text/html; charset=utf-8", **headers},
        text=body,
    )


@pytest.fixture(autouse=True)
def isolated_search_control(monkeypatch):
    monkeypatch.setattr("jarv.web._SEARCH_COORDINATOR", SearchCoordinator())
    monkeypatch.setattr("jarv.web._retry_delay", lambda _attempt: 0)


@pytest.fixture
def serve(monkeypatch):
    def install(*outcomes):
        pending = list(outcomes)
        requests = []

        def handler(request):
            requests.append(request)
            assert pending, "search made an unexpected extra request"
            outcome = pending.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(
            "jarv.web._create_client",
            lambda _timeout: httpx.Client(
                transport=httpx.MockTransport(handler), follow_redirects=False,
            ),
        )
        return requests

    return install


def _search(**kwargs):
    return search_web("query", timeout=kwargs.pop("timeout", 2), min_interval=0, **kwargs)


@pytest.mark.parametrize("status", [200, 202, 403])
@pytest.mark.parametrize("body", [
    '<form id="challenge-form"><p>Verify you are human</p></form>',
    '<div class="anomaly-modal"><p>Select all squares with a duck</p></div>',
    '<form action="//duckduckgo.com/anomaly.js?sv=html"><p>Verification</p></form>',
])
def test_challenge_is_identified_without_immediate_retry(serve, status, body):
    requests = serve(_response(body, status))

    with pytest.raises(WebToolError) as caught:
        _search()

    assert caught.value.kind == "blocked"
    assert caught.value.status_code == status
    assert len(requests) == 1


@pytest.mark.parametrize("css_class", ["no-results", "no-results__message"])
def test_confirmed_empty_search_is_a_success(serve, css_class):
    requests = serve(_response(f'<div class="{css_class}">No results found</div>'))

    output = _search()

    assert "Query: query" in output
    assert "No search results found" in output
    assert "challenge" not in output.lower()
    assert len(requests) == 1


def test_unrecognized_page_is_not_reported_as_empty_or_blocked(serve):
    requests = serve(_response("<main>Search interface temporarily unavailable</main>"))

    with pytest.raises(WebToolError) as caught:
        _search()

    assert caught.value.kind == "unexpected_response"
    assert caught.value.status_code == 200
    assert len(requests) == 1


def test_result_text_about_challenges_does_not_trigger_block_detection(serve):
    serve(_response(_page("DuckDuckGo challenge-form anomaly-modal explained")))

    assert "1. DuckDuckGo challenge-form anomaly-modal explained" in _search()


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_transient_server_error_retries_then_returns_results(serve, status):
    requests = serve(_response("Unavailable", status), _response(_page()))

    assert "1. One" in _search()
    assert len(requests) == 2


@pytest.mark.parametrize("exception_type", [httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout])
def test_transient_transport_error_retries_then_returns_results(serve, exception_type):
    requests = serve(exception_type("temporary failure"), _response(_page()))

    assert "1. One" in _search()
    assert len(requests) == 2


def test_transient_failures_stop_after_two_retries(serve):
    requests = serve(*[_response("Unavailable", 503) for _ in range(3)])

    with pytest.raises(WebToolError) as caught:
        _search()

    assert caught.value.status_code == 503
    assert len(requests) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_nontransient_http_errors_are_not_retried(serve, status):
    requests = serve(_response("Request rejected", status))

    with pytest.raises(WebToolError) as caught:
        _search()

    assert caught.value.status_code == status
    assert caught.value.kind != "blocked"
    assert len(requests) == 1


def test_retry_after_zero_allows_rate_limit_recovery(serve):
    requests = serve(
        _response("Too many requests", 429, **{"retry-after": "0"}),
        _response(_page()),
    )

    assert "1. One" in _search()
    assert len(requests) == 2


def test_retry_after_exceeding_budget_does_not_retry_early(serve):
    requests = serve(_response("Too many requests", 429, **{"retry-after": "60"}))

    with pytest.raises(WebToolError) as caught:
        _search(timeout=0.05)

    assert caught.value.kind == "rate_limited"
    assert caught.value.status_code == 429
    assert caught.value.retry_after == "60"
    assert len(requests) == 1


def test_retry_after_past_http_date_allows_recovery(serve):
    retry_after = format_datetime(datetime.now(timezone.utc) - timedelta(minutes=1), usegmt=True)
    requests = serve(
        _response("Too many requests", 429, **{"retry-after": retry_after}),
        _response(_page()),
    )

    assert "1. One" in _search()
    assert len(requests) == 2


def test_retry_after_future_http_date_respects_search_budget(serve):
    retry_after = format_datetime(datetime.now(timezone.utc) + timedelta(minutes=2), usegmt=True)
    requests = serve(_response("Too many requests", 429, **{"retry-after": retry_after}))

    with pytest.raises(WebToolError) as caught:
        _search(timeout=0.05)

    assert caught.value.kind == "rate_limited"
    assert caught.value.status_code == 429
    assert caught.value.retry_after == retry_after
    assert len(requests) == 1


@pytest.mark.parametrize("blocked_response", [
    _response('<form id="challenge-form"></form>', 202),
    _response("Too many requests", 429, **{"retry-after": "60"}),
])
def test_provider_cooldown_prevents_other_searches_from_requesting(serve, blocked_response):
    requests = serve(blocked_response)
    with pytest.raises(WebToolError):
        _search(timeout=0.05)

    with pytest.raises(WebToolError) as caught:
        search_web("a different query", timeout=2, min_interval=0)

    assert caught.value.kind == "cooldown"
    assert len(requests) == 1


def test_repeating_a_query_fetches_fresh_results_without_caching(serve):
    requests = serve(_response(_page()), _response(_page("Two")))

    assert "1. One" in _search()
    assert "1. Two" in _search()
    assert len(requests) == 2


def test_later_page_challenge_keeps_results_and_discloses_failure(serve):
    requests = serve(
        _response(_page(next_offset=10)),
        _response('<form id="challenge-form"></form>', 202),
    )

    output = _search(max_results=2)

    assert "1. One" in output
    assert "Partial results:" in output
    assert "challenge" in output.lower() or "verification" in output.lower()
    assert len(requests) == 2


def test_failure_before_requested_offset_preserves_actual_cause(serve):
    requests = serve(
        _response(_page(next_offset=10)),
        _response('<form id="challenge-form"></form>', 202),
    )

    with pytest.raises(WebToolError) as caught:
        _search(offset=1)

    assert caught.value.kind == "blocked"
    assert caught.value.status_code == 202
    assert len(requests) == 2


def test_offset_past_natural_exhaustion_is_a_successful_empty_result(serve):
    requests = serve(_response(_page()))

    output = _search(offset=5)

    assert "No search results found" in output
    assert "Offset: 5" in output
    assert "Partial results:" not in output
    assert len(requests) == 1


def test_confirmed_empty_later_page_is_natural_exhaustion(serve):
    requests = serve(
        _response(_page(next_offset=10)),
        _response('<div class="no-results">No more results</div>'),
    )

    output = _search(max_results=2)

    assert "1. One" in output
    assert "Partial results:" not in output
    assert len(requests) == 2


def test_page_cap_keeps_results_and_discloses_incomplete_search(serve):
    requests = serve(_response(_page(next_offset=10)))

    output = _search(max_results=2, max_pages=1)

    assert "1. One" in output
    assert "Partial results:" in output
    assert "page" in output.lower()
    assert len(requests) == 1


def test_page_cap_before_requested_offset_is_not_an_empty_success(serve):
    requests = serve(_response(_page(next_offset=10)))

    with pytest.raises(WebToolError) as caught:
        _search(offset=1, max_pages=1)

    assert caught.value.kind == "incomplete"
    assert len(requests) == 1


def test_repeated_continuation_stops_and_discloses_partial_results(serve):
    requests = serve(
        _response(_page(next_offset=10)),
        _response(_page("Two", next_offset=10)),
    )

    output = _search(max_results=3)

    assert "1. One" in output
    assert "2. Two" in output
    assert "Partial results:" in output
    assert len(requests) == 2


def test_retries_are_bounded_across_the_entire_search(serve):
    requests = serve(
        _response("Unavailable", 503),
        _response(_page(next_offset=10)),
        _response("Unavailable", 503),
        _response(_page("Two", next_offset=20)),
        _response("Unavailable", 503),
    )

    output = _search(max_results=3)

    assert "1. One" in output
    assert "2. Two" in output
    assert "Partial results:" in output
    assert "503" in output
    assert len(requests) == 5


@pytest.mark.parametrize("has_previous_results", [False, True])
def test_cancellation_does_not_turn_into_a_retry_or_partial_success(monkeypatch, has_previous_results):
    token = CancellationToken()
    calls = []

    def handler(request):
        calls.append(request)
        if has_previous_results and len(calls) == 1:
            return _response(_page(next_offset=10))
        token.cancel()
        raise httpx.ReadError("request cancelled")

    monkeypatch.setattr(
        "jarv.web._create_client",
        lambda _timeout: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(TurnCancelled):
        _search(cancellation_token=token)

    assert len(calls) == (2 if has_previous_results else 1)


def test_retry_after_positive_delay_is_observed(serve):
    requests = serve(
        _response("Too many requests", 429, **{"retry-after": "1"}),
        _response(_page()),
    )
    started = monotonic()

    assert "1. One" in _search(timeout=3)

    assert monotonic() - started >= 1
    assert len(requests) == 2


def test_challenge_honours_longer_retry_after(serve):
    requests = serve(_response('<form id="challenge-form"></form>', 202, **{"retry-after": "3600"}))
    with pytest.raises(WebToolError, match="3600 seconds"):
        _search()

    with pytest.raises(WebToolError, match="3600 seconds") as caught:
        _search()

    assert caught.value.kind == "cooldown"
    assert len(requests) == 1


def test_deadline_expiring_during_parsing_is_reported(serve, monkeypatch):
    serve(_response(_page()))
    original = web._parse_search_response

    def slow_parse(response):
        threading.Event().wait(0.04)
        return original(response)

    monkeypatch.setattr(web, "_parse_search_response", slow_parse)
    with pytest.raises(WebToolError) as caught:
        _search(timeout=0.02)
    assert caught.value.kind == "timeout"


@pytest.mark.parametrize("has_previous_results", [False, True])
def test_total_deadline_closes_active_request_and_preserves_partial_results(monkeypatch, has_previous_results):
    closed = threading.Event()
    calls = []

    def handler(request):
        calls.append(request)
        if has_previous_results and len(calls) == 1:
            return _response(_page(next_offset=10))
        assert closed.wait(1), "deadline did not close the active client"
        raise httpx.ReadError("closed at deadline")

    def create(_timeout):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        original_close = client.close

        def close():
            if not has_previous_results or len(calls) > 1:
                closed.set()
            original_close()

        client.close = close
        return client

    monkeypatch.setattr(web, "_create_client", create)
    if has_previous_results:
        output = _search(timeout=0.04, max_results=2)
        assert "1. One" in output
        assert "Partial results: timeout:" in output
    else:
        with pytest.raises(WebToolError) as caught:
            _search(timeout=0.04)
        assert caught.value.kind == "timeout"
    assert len(calls) == (2 if has_previous_results else 1)


def test_dispatch_reports_empty_search_as_success_and_challenge_as_failure(serve):
    serve(
        _response('<div class="no-results">No results found</div>'),
        _response('<form id="challenge-form"></form>', 202),
    )
    empty = web.dispatch_web_tool("web_search", {"query": "query"}, DEFAULT_CONFIG)
    blocked = web.dispatch_web_tool("web_search", {"query": "query"}, DEFAULT_CONFIG)

    assert tool_outcome(empty).status == "success"
    assert "No search results found" in empty
    assert tool_outcome(blocked).status == "failed"
    assert blocked.startswith("[web error: blocked:")


def test_dispatch_applies_search_settings(monkeypatch):
    options = {}

    def search(_query, _max_results, **kwargs):
        options.update(kwargs)
        return "ok"

    monkeypatch.setattr(web, "search_web", search)
    config = {**DEFAULT_CONFIG, "web_timeout": 20, "web_search_interval": 3, "web_search_max_pages": 2}
    web.dispatch_web_tool("web_search", {"query": "query"}, config)

    assert options["timeout"] == 20
    assert options["min_interval"] == 3
    assert options["max_pages"] == 2
