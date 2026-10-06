"""Cancellation must stop socket I/O, including before a response exists."""

import socket
import socketserver
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import urlsplit

import httpx
import pytest

from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.http_cancellation import CancellableClient
from jarv.http_transport import (
    create_client, open_stream_response, request_json, send_with_retries,
)


@pytest.mark.parametrize("stream", [False, True])
def test_cancelled_request_has_no_side_effects(stream):
    token = CancellationToken()
    token.cancel()
    client = Mock()

    with pytest.raises(TurnCancelled):
        send_with_retries(client, "POST", "/", stream=stream, cancellation_token=token)

    client.build_request.assert_not_called()
    client.send.assert_not_called()


def test_cancellation_during_request_build_prevents_send():
    token = CancellationToken()
    client = Mock()
    client.build_request.side_effect = lambda *a, **k: token.cancel()

    with pytest.raises(TurnCancelled):
        send_with_retries(client, "POST", "/", cancellation_token=token)

    client.send.assert_not_called()


@pytest.mark.parametrize("transport_error", [False, True])
def test_cancellation_at_send_completion_does_not_return_or_retry(transport_error):
    token = CancellationToken()
    response = httpx.Response(200, stream=httpx.ByteStream(b"{}"))
    client = Mock()

    def send(*args, **kwargs):
        token.cancel()
        if transport_error:
            raise httpx.ReadError("cancelled socket")
        return response

    client.send.side_effect = send
    with pytest.raises(TurnCancelled):
        send_with_retries(
            client, "POST", "/", stream=True,
            cancellation_token=token, max_retries=0,
        )

    client.send.assert_called_once()
    if not transport_error:
        assert response.is_closed


@pytest.fixture
def stalled_server():
    received = threading.Event()
    disconnected = threading.Event()
    paths = []
    connections = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            self.path = urlsplit(self.path).path
            paths.append(self.path)
            connections.append(self.connection)
            if self.path == "/ok":
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
                return
            if self.path in ("/partial_headers", "/delayed"):
                self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Len")
                self.wfile.flush()
            if self.path == "/delayed":
                # Exceed several cancellation polling intervals without
                # exceeding the request's configured read timeout.
                threading.Event().wait(0.2)
                self.wfile.write(b"gth: 2\r\n\r\n{}")
                return
            if self.path in ("/body", "/error_body", "/stream_body"):
                self.send_response(400 if self.path == "/error_body" else 200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.flush()
            received.set()
            self.connection.settimeout(3)
            try:
                if self.connection.recv(1) == b"":
                    disconnected.set()
            except ConnectionResetError:
                # Closing a socket with unread response bytes can send TCP RST
                # instead of EOF. Both prove the cancelled connection is closed.
                disconnected.set()
            except (OSError, socket.timeout):
                pass
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True,
    )
    server_thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", received, disconnected, paths, connections
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)


@pytest.mark.parametrize("connection_mode", ["fresh", "reused", "proxy"])
@pytest.mark.parametrize("path", [
    "/headers", "/partial_headers", "/body", "/error_body", "/stream_body",
])
def test_cancel_interrupts_stalled_request_and_preserves_client(
    stalled_server, monkeypatch, connection_mode, path,
):
    url, received, disconnected, paths, connections = stalled_server
    token = CancellationToken()
    outcomes = []
    if connection_mode == "proxy":
        for name in ("HTTP_PROXY", "http_proxy"):
            monkeypatch.setenv(name, url)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.setenv(name, "")
        url = "http://provider.invalid"

    with create_client(url, {}, timeout=10) as client:
        if connection_mode == "reused":
            assert request_json("test", client, "GET", "/ok") == {}

        def request():
            try:
                if path == "/body":
                    request_json("test", client, "GET", path, cancellation_token=token)
                else:
                    response, unregister = open_stream_response(
                        client, "GET", path, provider="test", cancellation_token=token,
                    )
                    try:
                        response.read()
                    finally:
                        unregister()
                        response.close()
            except BaseException as exc:
                outcomes.append(exc)

        worker = threading.Thread(target=request, daemon=True)
        worker.start()
        try:
            assert received.wait(2), "server did not receive the request"
            token.cancel()
            worker.join(timeout=1)
            assert not worker.is_alive(), "cancelled request remained blocked on socket I/O"
            assert len(outcomes) == 1
            assert isinstance(outcomes[0], TurnCancelled)
            assert disconnected.wait(1), "cancelled request left its socket open"
            assert paths.count(path) == 1, "cancelled request was retried"
            if connection_mode == "reused":
                assert connections[0] is connections[1]
            assert not client.is_closed
            assert request_json(
                "test", client, "GET", "/ok", cancellation_token=CancellationToken(),
            ) == {}
        finally:
            token.cancel()
            worker.join(timeout=4)


def test_read_polling_preserves_partial_headers_without_resending(stalled_server):
    url, _, _, paths, _ = stalled_server
    with create_client(url, {}, timeout=2) as client:
        assert request_json(
            "test", client, "GET", "/delayed", cancellation_token=CancellationToken(),
        ) == {}
    assert paths == ["/delayed"]


def test_read_polling_preserves_configured_timeout(stalled_server):
    url, _, disconnected, paths, _ = stalled_server
    with create_client(url, {}, timeout=0.15) as client:
        with pytest.raises(httpx.ReadTimeout):
            request_json(
                "test", client, "GET", "/headers",
                cancellation_token=CancellationToken(), max_retries=0,
            )
    assert disconnected.wait(1)
    assert paths == ["/headers"]


@pytest.fixture
def stalled_tls_server():
    received = threading.Event()
    disconnected = threading.Event()
    handshakes = []

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(3)
            try:
                data = self.request.recv(4096)
                if data.startswith(b"CONNECT "):
                    while b"\r\n\r\n" not in data:
                        data += self.request.recv(4096)
                    self.request.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                    data = self.request.recv(4096)
                if data:
                    handshakes.append(data)
                    received.set()
                # Receive the ClientHello but never send a ServerHello.
                while self.request.recv(4096):
                    pass
                disconnected.set()
            except ConnectionResetError:
                disconnected.set()
            except (OSError, socket.timeout):
                pass

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler) as server:
        server.daemon_threads = True
        worker = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True,
        )
        worker.start()
        try:
            yield server.server_address[1], received, disconnected, handshakes
        finally:
            server.shutdown()
            worker.join(timeout=1)


@pytest.mark.parametrize("proxy", [False, True])
def test_cancel_interrupts_tls_handshake(stalled_tls_server, stalled_server, monkeypatch, proxy):
    port, received, disconnected, handshakes = stalled_tls_server
    normal_url, _, _, _, _ = stalled_server
    url = f"https://127.0.0.1:{port}"
    if proxy:
        monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        monkeypatch.setenv("no_proxy", "127.0.0.1")
        url = "https://provider.invalid"

    token = CancellationToken()
    outcomes = []
    with create_client(url, {}, connect_timeout=3) as client:
        def request():
            try:
                request_json("test", client, "GET", "/", cancellation_token=token)
            except BaseException as exc:
                outcomes.append(exc)

        worker = threading.Thread(target=request, daemon=True)
        worker.start()
        try:
            assert received.wait(2), "server did not receive TLS ClientHello"
            token.cancel()
            worker.join(timeout=1)
            assert not worker.is_alive(), "cancelled request remained in TLS handshake"
            assert len(outcomes) == 1
            assert isinstance(outcomes[0], TurnCancelled)
            assert disconnected.wait(1), "cancelled TLS handshake left its socket open"
            assert len(handshakes) == 1, "cancelled handshake was retried"
            assert not client.is_closed
            assert request_json("test", client, "GET", normal_url + "/ok") == {}
        finally:
            token.cancel()
            worker.join(timeout=4)


def test_tls_handshake_preserves_connect_timeout(stalled_tls_server):
    port, _, disconnected, handshakes = stalled_tls_server
    with create_client(f"https://127.0.0.1:{port}", {}, connect_timeout=0.15) as client:
        with pytest.raises(httpx.ConnectTimeout):
            request_json(
                "test", client, "GET", "/",
                cancellation_token=CancellationToken(), max_retries=0,
            )
    assert disconnected.wait(1)
    assert len(handshakes) == 1


@pytest.fixture
def local_tls_server():
    # A self-signed certificate and public test key, never production credentials.
    certificate = Path(__file__).parent / "fixtures" / "http-cancellation-test.pem"
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(certificate)
    connections = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            connections.append(self.connection)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.socket = server_context.wrap_socket(server.socket, server_side=True)
        worker = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.01), daemon=True,
        )
        worker.start()
        try:
            yield f"https://127.0.0.1:{server.server_port}", certificate, connections
        finally:
            server.shutdown()
            worker.join(timeout=1)


def test_completed_tls_handshake_releases_previous_cancellation_token(local_tls_server):
    url, certificate, connections = local_tls_server
    context = ssl.create_default_context(cafile=str(certificate))
    first_token = CancellationToken()
    with CancellableClient(base_url=url, verify=context, trust_env=False, timeout=2) as client:
        assert request_json("test", client, "GET", "/", cancellation_token=first_token) == {}
        first_token.cancel()
        assert request_json(
            "test", client, "GET", "/", cancellation_token=CancellationToken(),
        ) == {}
    assert len(connections) == 2
    assert connections[0] is connections[1], "completed turn closed a reusable TLS connection"


def test_cancellable_tls_keeps_certificate_verification(local_tls_server):
    url, _, connections = local_tls_server
    with CancellableClient(base_url=url, trust_env=False, timeout=2) as client:
        with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
            request_json(
                "test", client, "GET", "/",
                cancellation_token=CancellationToken(), max_retries=0,
            )
    assert connections == []


@pytest.mark.parametrize("method", ["connect_tcp", "connect_unix_socket"])
def test_cancel_interrupts_connect_and_closes_late_stream(method):
    from jarv.http_cancellation import _NetworkBackend, cancellation_scope

    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    stream = Mock()
    stream.close.side_effect = closed.set
    backend = Mock()

    def connect(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return stream

    getattr(backend, method).side_effect = connect
    token = CancellationToken()
    outcomes = []

    def request():
        try:
            with cancellation_scope(token):
                getattr(_NetworkBackend(backend), method)("example.invalid", timeout=10)
        except BaseException as exc:
            outcomes.append(exc)

    worker = threading.Thread(target=request, daemon=True)
    worker.start()
    try:
        assert started.wait(1)
        token.cancel()
        worker.join(timeout=1)
        assert not worker.is_alive(), "cancelled DNS/connect kept its caller blocked"
        assert len(outcomes) == 1 and isinstance(outcomes[0], TurnCancelled)
        stream.write.assert_not_called()
    finally:
        token.cancel()
        release.set()
        worker.join(timeout=1)
    assert closed.wait(1)
    stream.close.assert_called_once()


def test_cancelled_connect_waits_for_no_helper_slot(monkeypatch):
    from jarv import http_cancellation

    slots = threading.BoundedSemaphore(1)
    waiting = threading.Event()

    class ObservedSlots:
        def acquire(self, timeout):
            waiting.set()
            return slots.acquire(timeout=timeout)

        def release(self):
            slots.release()

    monkeypatch.setattr(http_cancellation, "_CONNECT_SLOTS", ObservedSlots())
    token = CancellationToken()
    backend = Mock()
    outcomes = []

    def request():
        try:
            with http_cancellation.cancellation_scope(token):
                http_cancellation._NetworkBackend(backend).connect_tcp("host", 80)
        except BaseException as exc:
            outcomes.append(exc)

    with slots:
        worker = threading.Thread(target=request, daemon=True)
        worker.start()
        try:
            assert waiting.wait(1)
            token.cancel()
            worker.join(timeout=1)
            assert not worker.is_alive()
            assert len(outcomes) == 1 and isinstance(outcomes[0], TurnCancelled)
            backend.connect_tcp.assert_not_called()
        finally:
            token.cancel()
    worker.join(timeout=1)


def test_cancellation_during_connect_completion_closes_result():
    from jarv.http_cancellation import _NetworkBackend, cancellation_scope

    token = CancellationToken()
    stream = Mock()
    closed = threading.Event()
    stream.close.side_effect = closed.set
    backend = Mock()

    def connect(*args, **kwargs):
        token.cancel()
        return stream

    backend.connect_tcp.side_effect = connect
    with cancellation_scope(token), pytest.raises(TurnCancelled):
        _NetworkBackend(backend).connect_tcp("host", 80)
    assert closed.wait(1)
    stream.close.assert_called_once()


def test_completed_connect_releases_previous_token():
    from jarv.http_cancellation import _NetworkBackend, cancellation_scope

    token = CancellationToken()
    stream = Mock()
    backend = Mock()
    backend.connect_tcp.return_value = stream
    with cancellation_scope(token):
        connected = _NetworkBackend(backend).connect_tcp("host", 80)
    token.cancel()
    stream.close.assert_not_called()
    connected.close()
    stream.close.assert_called_once()


def test_cancel_interrupts_blocked_write():
    from jarv.http_cancellation import _NetworkStream, cancellation_scope

    writer, reader = socket.socketpair()
    writer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    writer.settimeout(5)
    started = threading.Event()
    token = CancellationToken()
    outcomes = []
    stream = Mock()
    stream.get_extra_info.return_value = writer

    def write(buffer, timeout):
        started.set()
        writer.sendall(buffer)

    stream.write.side_effect = write

    def send():
        try:
            with cancellation_scope(token):
                _NetworkStream(stream).write(b"x" * 2_000_000, timeout=5)
        except BaseException as exc:
            outcomes.append(exc)

    worker = threading.Thread(target=send, daemon=True)
    worker.start()
    try:
        assert started.wait(1)
        token.cancel()
        worker.join(timeout=1)
        assert not worker.is_alive(), "cancelled write remained blocked"
        assert len(outcomes) == 1 and isinstance(outcomes[0], TurnCancelled)
        stream.write.assert_called_once()
    finally:
        token.cancel()
        writer.close()
        reader.close()
        worker.join(timeout=1)


def test_completed_write_does_not_leave_abort_callback():
    from jarv.http_cancellation import _NetworkStream, cancellation_scope

    token = CancellationToken()
    stream = Mock()
    stream.get_extra_info.return_value = None
    with cancellation_scope(token):
        _NetworkStream(stream).write(b"request", timeout=1)
    token.cancel()
    stream.close.assert_not_called()
