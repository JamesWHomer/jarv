"""Cancellation-aware reads for the synchronous HTTPX connection pool.

Polling at the network read boundary preserves partially received headers and
bodies. Retrying an entire request after a short read timeout would not.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar

import httpcore
import httpx

from .cancellation import CancellationToken


_active_token: ContextVar[CancellationToken | None] = ContextVar(
    "http_cancellation_token", default=None,
)
_POLL_INTERVAL = 0.05


@contextmanager
def cancellation_scope(token: CancellationToken | None):
    marker = _active_token.set(token)
    try:
        if token is not None:
            token.throw_if_cancelled()
        yield
    finally:
        _active_token.reset(marker)


class _NetworkStream(httpcore.NetworkStream):
    def __init__(self, stream):
        self._stream = stream

    def read(self, max_bytes, timeout=None):
        token = _active_token.get()
        if token is None:
            return self._stream.read(max_bytes, timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            token.throw_if_cancelled()
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise httpcore.ReadTimeout("timed out")
            read_timeout = (
                _POLL_INTERVAL if remaining is None
                else min(_POLL_INTERVAL, max(0.0, remaining))
            )
            try:
                data = self._stream.read(max_bytes, read_timeout)
            except httpcore.ReadTimeout:
                token.throw_if_cancelled()
                if deadline is not None and time.monotonic() >= deadline:
                    raise
            except Exception:
                token.throw_if_cancelled()
                raise
            else:
                token.throw_if_cancelled()
                return data

    def write(self, buffer, timeout=None):
        token = _active_token.get()
        if token is not None:
            token.throw_if_cancelled()
        self._stream.write(buffer, timeout)

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        token = _active_token.get()
        if token is not None:
            token.throw_if_cancelled()
        return _NetworkStream(
            self._stream.start_tls(ssl_context, server_hostname, timeout),
        )

    def get_extra_info(self, info):
        return self._stream.get_extra_info(info)

    def close(self):
        self._stream.close()


class _NetworkBackend(httpcore.NetworkBackend):
    def __init__(self, backend):
        self._backend = backend

    def connect_tcp(self, *args, **kwargs):
        token = _active_token.get()
        if token is not None:
            token.throw_if_cancelled()
        return _NetworkStream(self._backend.connect_tcp(*args, **kwargs))

    def connect_unix_socket(self, *args, **kwargs):
        token = _active_token.get()
        if token is not None:
            token.throw_if_cancelled()
        return _NetworkStream(self._backend.connect_unix_socket(*args, **kwargs))

    def sleep(self, seconds):
        self._backend.sleep(seconds)


class CancellableClient(httpx.Client):
    # HTTPX does not expose HTTPCore's network_backend option. Keep this adapter
    # at transport construction so normal pools, proxy settings and TLS options
    # are retained, and pooled connections never capture a previous turn's token.
    @staticmethod
    def _with_cancellation(transport):
        if isinstance(transport, httpx.HTTPTransport):
            pool = transport._pool
            pool._network_backend = _NetworkBackend(pool._network_backend)
        return transport

    def _init_transport(self, *args, **kwargs):
        return self._with_cancellation(super()._init_transport(*args, **kwargs))

    def _init_proxy_transport(self, *args, **kwargs):
        return self._with_cancellation(super()._init_proxy_transport(*args, **kwargs))


class CancellableResponseStream(httpx.SyncByteStream):
    def __init__(self, stream, token):
        self._stream = stream
        self._token = token

    def __iter__(self):
        iterator = iter(self._stream)
        while True:
            with cancellation_scope(self._token):
                try:
                    chunk = next(iterator)
                except StopIteration:
                    return
            yield chunk

    def close(self):
        self._stream.close()
