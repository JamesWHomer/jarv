"""Shared HTTP helpers for direct provider integrations."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from typing import Any

from .cancellation import CancellationToken


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}


def normalized_token_count(value: Any, *, default: int = 0) -> int:
    """Keep malformed usage metadata from discarding a valid provider reply."""
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if parsed >= 0 else default


class ProviderHTTPError(Exception):
    def __init__(
        self,
        provider: str,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.error_type = error_type
        self.request_id = request_id
        details = [str(value) for value in (status_code, error_type) if value]
        prefix = f"{provider} API error"
        if details:
            prefix += f" ({', '.join(details)})"
        if request_id:
            message = f"{message} [request_id={request_id}]"
        super().__init__(f"{prefix}: {message}")


def create_client(
    base_url: str,
    headers: dict[str, str],
    *,
    timeout: float = 600,
    connect_timeout: float = 10,
):
    import httpx
    from .http_cancellation import CancellableClient

    return CancellableClient(
        base_url=base_url.rstrip("/"),
        headers=headers,
        timeout=httpx.Timeout(timeout, connect=connect_timeout),
    )


def response_error(provider: str, response, data: dict | None = None) -> ProviderHTTPError:
    import httpx

    if data is None:
        try:
            data = response.json()
        except (ValueError, httpx.HTTPError, httpx.StreamError):
            data = {}
    error = data.get("error") if isinstance(data, dict) else None
    error = error if isinstance(error, dict) else {}
    try:
        response_text = response.text
    except (ValueError, httpx.HTTPError, httpx.StreamError):
        response_text = ""
    message = str(
        error.get("message")
        or (data.get("message") if isinstance(data, dict) else "")
        or response_text
        or "request failed"
    )
    return ProviderHTTPError(
        provider,
        message,
        status_code=getattr(response, "status_code", None),
        error_type=str(error.get("type") or error.get("status") or "") or None,
        request_id=(
            response.headers.get("x-request-id")
            or response.headers.get("request-id")
            or None
        ),
    )


def _sleep(delay: float, cancellation_token: CancellationToken | None) -> None:
    deadline = time.monotonic() + delay
    while True:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def _retry_delay(response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            delay = float(retry_after)
            if math.isfinite(delay):
                return min(max(delay, 0.0), 30.0)
        except ValueError:
            pass
    return min(0.5 * (2 ** attempt), 4.0)


def send_with_retries(
    client,
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    params: dict | None = None,
    stream: bool = False,
    cancellation_token: CancellationToken | None = None,
    max_retries: int = 2,
):
    import httpx
    from .http_cancellation import CancellableResponseStream, cancellation_scope

    for attempt in range(max_retries + 1):
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        request = client.build_request(method, path, json=json_body, params=params)
        response = None
        try:
            with cancellation_scope(cancellation_token):
                response = client.send(request, stream=stream)
                if cancellation_token is not None:
                    cancellation_token.throw_if_cancelled()
        except httpx.TransportError:
            if cancellation_token is not None:
                cancellation_token.throw_if_cancelled()
            if attempt >= max_retries:
                raise
            _sleep(min(0.5 * (2 ** attempt), 4.0), cancellation_token)
            continue
        except BaseException:
            if response is not None:
                response.close()
            raise
        if response.status_code not in RETRYABLE_STATUS_CODES or attempt >= max_retries:
            if stream and cancellation_token is not None:
                response.stream = CancellableResponseStream(
                    response.stream, cancellation_token,
                )
            return response
        delay = _retry_delay(response, attempt)
        response.close()
        _sleep(delay, cancellation_token)
    raise RuntimeError("unreachable")


def request_json(
    provider: str,
    client,
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    params: dict | None = None,
    cancellation_token: CancellationToken | None = None,
    max_retries: int = 2,
) -> dict:
    response = send_with_retries(
        client,
        method,
        path,
        json_body=json_body,
        params=params,
        cancellation_token=cancellation_token,
        max_retries=max_retries,
    )
    try:
        if response.status_code >= 400:
            raise response_error(provider, response)
        data = response.json()
        if not isinstance(data, dict):
            raise ProviderHTTPError(provider, "response was not a JSON object")
        return data
    finally:
        response.close()


def iter_sse(response) -> Iterator[tuple[str, str]]:
    event_name = ""
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if data_lines:
                yield event_name, "\n".join(data_lines)
            event_name = ""
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield event_name, "\n".join(data_lines)


def iter_sse_json(provider: str, response) -> Iterator[tuple[str, dict[str, Any]]]:
    for event_name, raw in iter_sse(response):
        if raw == "[DONE]":
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderHTTPError(provider, f"invalid SSE JSON: {exc}") from exc
        if isinstance(data, dict):
            yield event_name or str(data.get("type") or ""), data


def open_stream_response(
    client,
    method: str,
    path: str,
    *,
    provider: str,
    json_body: dict | None = None,
    params: dict | None = None,
    cancellation_token: CancellationToken | None = None,
    max_retries: int = 2,
):
    """POST (or other method) a streaming request and validate the HTTP status."""
    response = send_with_retries(
        client,
        method,
        path,
        json_body=json_body,
        params=params,
        stream=True,
        cancellation_token=cancellation_token,
        max_retries=max_retries,
    )
    unregister = (
        cancellation_token.register(response.close)
        if cancellation_token is not None else lambda: None
    )
    try:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        if response.status_code >= 400:
            import httpx

            try:
                response.read()
                data = response.json()
            except (ValueError, httpx.HTTPError, httpx.StreamError):
                data = None
            if cancellation_token is not None:
                cancellation_token.throw_if_cancelled()
            raise response_error(provider, response, data)
        return response, unregister
    except BaseException:
        unregister()
        response.close()
        raise


# Retain the older transport entry point without duplicating retry/cleanup logic.
request_json_response = request_json
