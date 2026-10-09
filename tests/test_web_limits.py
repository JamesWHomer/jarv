"""URL deadlines and JSON formatting stay bounded before read pagination."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import httpcore
import httpx
import pytest

from jarv import search_control, web
from jarv.artifacts import ArtifactStore
from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.config import DEFAULT_CONFIG
from jarv.read_tool import dispatch_read_tool
from jarv.retained_outputs import RetainedOutputStore
from jarv.tool_outputs import tool_outcome


@pytest.fixture
def slow_server():
    received = threading.Event()
    disconnected = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(8192 * 100))
            self.end_headers()
            received.set()
            try:
                for _ in range(100):
                    self.wfile.write(b"x" * 8192)
                    self.wfile.flush()
                    time.sleep(0.03)
            except OSError:
                disconnected.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/", received, disconnected
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)


def test_url_deadline_interrupts_continuously_streaming_response(slow_server):
    url, received, disconnected = slow_server
    started = time.monotonic()
    with pytest.raises(web.WebToolError) as caught:
        web.fetch_web_bytes(url, timeout=0.25)
    assert caught.value.kind == "timeout"
    assert "0.25 seconds" in str(caught.value)
    assert received.is_set()
    assert time.monotonic() - started < 1
    assert disconnected.wait(1)


def test_url_cancellation_is_not_reported_as_timeout(slow_server):
    url, received, disconnected = slow_server
    token = CancellationToken()
    outcomes = []

    def request():
        try:
            web.fetch_web_bytes(url, timeout=5, cancellation_token=token)
        except BaseException as exc:
            outcomes.append(exc)

    worker = threading.Thread(target=request, daemon=True)
    worker.start()
    try:
        assert received.wait(2)
        token.cancel()
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert len(outcomes) == 1 and isinstance(outcomes[0], TurnCancelled)
        assert disconnected.wait(1)
    finally:
        token.cancel()
        worker.join(timeout=1)


@pytest.mark.parametrize(
    "setup_time, expected_paths, expected_timeouts",
    [
        (0, ["/", "/redirect/1"], [0.07, 0.03]),
        (0.04, ["/"], [0.03]),
    ],
)
def test_url_redirects_share_one_deadline(
    monkeypatch, setup_time, expected_paths, expected_timeouts,
):
    now = [0.0]
    requests = []
    # Advance the budget clock explicitly so scheduler delays cannot change
    # how many redirects fit. Real timer cancellation is covered separately.
    monkeypatch.setattr(search_control, "monotonic", lambda: now[0])
    monkeypatch.setattr(search_control.threading, "Timer", Mock())

    def handler(request):
        requests.append(request)
        now[0] += 0.04
        return httpx.Response(302, headers={"Location": f"/redirect/{len(requests)}"})

    def create_client(timeout):
        now[0] += setup_time
        return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(web, "_create_client", create_client)
    with pytest.raises(web.WebToolError) as caught:
        web.fetch_web_bytes("https://example.test/", timeout=0.07)
    assert caught.value.kind == "timeout"
    assert [request.url.path for request in requests] == expected_paths
    assert [request.extensions["timeout"]["read"] for request in requests] == pytest.approx(expected_timeouts)


def test_url_timeout_does_not_cancel_parent_or_later_requests(monkeypatch):
    parent = CancellationToken()
    clients = []

    def create(timeout):
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"ok")))
        clients.append(client)
        return client

    monkeypatch.setattr(web, "_create_client", create)
    with pytest.raises(web.WebToolError) as caught:
        web.fetch_web_bytes("https://example.test/", timeout=0, cancellation_token=parent)
    assert caught.value.kind == "timeout"
    assert not parent.cancelled
    assert clients == []
    assert web.fetch_web_bytes("https://example.test/", timeout=1, cancellation_token=parent).body == b"ok"
    parent.cancel()
    assert all(client.is_closed for client in clients)


def test_url_deadline_returns_during_dns_and_never_sends_late_request(monkeypatch):
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    stream = Mock()
    stream.close.side_effect = closed.set

    def connect(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return stream

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", connect)
    try:
        begin = time.monotonic()
        with pytest.raises(web.WebToolError) as caught:
            web.fetch_web_bytes("http://example.invalid/", timeout=0.2)
        assert caught.value.kind == "timeout"
        assert started.is_set()
        assert time.monotonic() - begin < 1
        stream.write.assert_not_called()
    finally:
        release.set()
    assert closed.wait(1)
    stream.write.assert_not_called()
    stream.close.assert_called_once()


@pytest.mark.parametrize("configured, expected", [
    (2.5, 2.5), ("3.5", 3.5), (True, 1.0),
    (None, None), ("invalid", None), ([], None),
    (0, None), (-1, None), (float("inf"), None), (float("nan"), None),
])
def test_web_timeout_outcome_survives_url_read_and_search(monkeypatch, configured, expected):
    config = {**DEFAULT_CONFIG, "web_timeout": configured}
    if expected is None:
        expected = float(DEFAULT_CONFIG["web_timeout"])

    def timed_out(*args, **kwargs):
        assert kwargs["timeout"] == expected
        raise web.WebToolError("request timed out", kind="timeout")

    monkeypatch.setattr("jarv.read_tool.fetch_web_bytes", timed_out)
    output = dispatch_read_tool(
        {"input": "https://example.test/"}, config=config,
        visible_labels=set(), artifact_store=ArtifactStore(), retained_store=RetainedOutputStore(),
    )
    assert output.startswith("[read error:")
    assert tool_outcome(output).status == "timed_out"
    monkeypatch.setattr(web, "search_web", timed_out)
    output = web.dispatch_web_tool("web_search", {"query": "test"}, config)
    assert tool_outcome(output).status == "timed_out"


def _json_content(text):
    return web.web_content_from_bytes("url", "url", "application/json", text.encode()).text


@pytest.mark.parametrize("depth", [web.MAX_JSON_FORMAT_DEPTH + 1, 1100, 5000])
def test_deep_json_falls_back_without_parsing_or_pretty_printing(monkeypatch, depth):
    text = "[" * depth + "0" + "]" * depth
    monkeypatch.setattr(web.json, "loads", lambda *a, **k: pytest.fail("deep JSON must not be parsed"))
    assert _json_content(text) == text


def test_large_json_preserves_original_text_without_parsing(monkeypatch):
    text = '["' + "x" * web.MAX_JSON_FORMAT_INPUT_BYTES + '"]'
    monkeypatch.setattr(web.json, "loads", lambda *a, **k: pytest.fail("large JSON must not be parsed"))
    assert _json_content(text) == text


def test_json_at_input_and_depth_limits_is_still_formatted():
    text = '["' + "x" * (web.MAX_JSON_FORMAT_INPUT_BYTES - 4) + '"]'
    assert _json_content(text) == json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    text = "[" * web.MAX_JSON_FORMAT_DEPTH + "0" + "]" * web.MAX_JSON_FORMAT_DEPTH
    assert _json_content(text) == json.dumps(json.loads(text), indent=2)


def test_json_depth_scan_ignores_escaped_quotes_and_brackets_in_strings():
    value = {"string": '["\\' * 100, "nested": [["é"]]}
    text = json.dumps(value, ensure_ascii=False)
    assert _json_content(text) == json.dumps(value, indent=2, ensure_ascii=False)


def test_wide_nested_json_stops_formatting_at_output_budget():
    text = "[" * 63 + ",".join("0" for _ in range(20_000)) + "]" * 63
    # Indenting every leaf would expand this 40 KiB body past 2 MiB.
    assert _json_content(text) == text


def test_json_formatter_stops_consuming_chunks_when_budget_is_exceeded(monkeypatch):
    monkeypatch.setattr(web, "MAX_JSON_FORMAT_OUTPUT_BYTES", 4)

    def encode(self, value):
        yield "1234"
        yield "5"
        pytest.fail("formatter consumed output beyond its budget")

    monkeypatch.setattr(web.json.JSONEncoder, "iterencode", encode)
    assert _json_content("[0]") == "[0]"


def test_multibyte_json_input_is_bounded_by_utf8_bytes(monkeypatch):
    text = '["' + "é" * (web.MAX_JSON_FORMAT_INPUT_BYTES // 2) + '"]'
    assert len(text) < web.MAX_JSON_FORMAT_INPUT_BYTES < len(text.encode("utf-8"))
    monkeypatch.setattr(web.json, "loads", lambda *a, **k: pytest.fail("oversized UTF-8 input was parsed"))
    assert _json_content(text) == text


def test_multibyte_json_output_is_bounded_by_utf8_bytes(monkeypatch):
    text = '["é"]'
    pretty = json.dumps(["é"], indent=2, ensure_ascii=False)
    monkeypatch.setattr(web, "MAX_JSON_FORMAT_OUTPUT_BYTES", len(pretty))
    assert len(pretty.encode("utf-8")) > len(pretty)
    assert _json_content(text) == text
    monkeypatch.setattr(web, "MAX_JSON_FORMAT_OUTPUT_BYTES", len(pretty.encode("utf-8")))
    assert _json_content(text) == pretty


@pytest.mark.parametrize("error", [ValueError("number too large"), RecursionError("too deep")])
def test_json_parser_resource_errors_preserve_original(monkeypatch, error):
    text = " { invalid input } "

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(web.json, "loads", fail)
    assert _json_content(text) == text
