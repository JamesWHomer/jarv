"""Keyless web search and bounded web page fetching."""

from __future__ import annotations

import json
import math
import random
import re
import time
import zlib
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, unquote, urljoin, urlsplit, urlunsplit

if TYPE_CHECKING:
    import httpx

from . import __version__
from .cancellation import CancellationToken, TurnCancelled
from .config import DEFAULT_CONFIG, get_setting
from .search_control import (
    SearchBudget,
    SearchCoordinator,
    SearchCooldownError,
    SearchDeadlineExceeded,
)
from .tool_outputs import with_tool_outcome


DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
SEARCH_ENGINE_LABEL = "DuckDuckGo"
from .pdf_extract import PDF_MAGIC, is_pdf_bytes, is_pdf_media_type
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5
FRAGMENT_MIN_CHARS = 200
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 20
_DECODE_CHUNK_BYTES = 64 * 1024
MAX_JSON_FORMAT_INPUT_BYTES = 256 * 1024
MAX_JSON_FORMAT_DEPTH = 64
MAX_JSON_FORMAT_OUTPUT_BYTES = MAX_RESPONSE_BYTES
DEFAULT_SEARCH_INTERVAL = 1.0
DEFAULT_SEARCH_MAX_PAGES = 5
MAX_SEARCH_RETRIES = 2
SEARCH_CHALLENGE_COOLDOWN = 60.0
_SEARCH_COORDINATOR = SearchCoordinator()

WEB_SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": (
        "Search the public web. Returns search result titles, URLs, and snippets "
        "from DuckDuckGo; it does not read page contents."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_SEARCH_RESULTS,
                "description": "Maximum results to return. Defaults to 5.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "Number of unique results to skip. Defaults to 0.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

class WebToolError(Exception):
    """A user-visible web tool failure."""

    def __init__(
        self, message: str, *, kind: str = "request_failed",
        status_code: int | None = None, retry_after: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = retryable


@dataclass(frozen=True)
class WebResponse:
    final_url: str
    content_type: str
    body: bytes
    status_code: int
    retry_after: str | None = None


@dataclass(frozen=True)
class FetchedWebContent:
    requested_url: str
    final_url: str
    media_type: str
    title: str
    text: str
    fragment: str = ""
    fragment_applied: bool = False


@dataclass(frozen=True)
class FetchedWebBytes:
    requested_url: str
    final_url: str
    content_type: str
    media_type: str
    body: bytes


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


def _normalize_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_normalize_space(line) for line in normalized.split("\n")]
    output: list[str] = []
    blank = False
    for line in lines:
        if line:
            output.append(line)
            blank = False
        elif output and not blank:
            output.append("")
            blank = True
    return "\n".join(output).strip()


def _class_tokens(attrs: list[tuple[str, str | None]]) -> set[str]:
    for key, value in attrs:
        if key == "class" and value:
            return set(value.split())
    return set()


def _attr(attrs: list[tuple[str, str | None]], name: str) -> str:
    for key, value in attrs:
        if key == name and value is not None:
            return value
    return ""


def _decode_search_url(href: str) -> str | None:
    absolute = urljoin("https://duckduckgo.com/", href)
    parsed = urlsplit(absolute)
    duckduckgo_hosts = {"duckduckgo.com", "www.duckduckgo.com"}
    if parsed.hostname in duckduckgo_hosts and parsed.path == "/l/":
        destination = parse_qs(parsed.query).get("uddg", [""])[0]
        if not destination:
            return None
        absolute = destination
        parsed = urlsplit(absolute)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.hostname in duckduckgo_hosts:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._anchor_kind: str | None = None
        self._anchor_href = ""
        self._anchor_text: list[str] = []
        self.next_params: dict[str, str] | None = None
        self._form_params: dict[str, str] | None = None
        self._form_is_next = False
        self.challenge = False
        self.no_results = False

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        classes = _class_tokens(attrs)
        if (
            any(name == "anomaly-modal" or name.startswith("anomaly-modal__") for name in classes)
            or (tag == "form" and (
                _attr(attrs, "id") == "challenge-form"
                or _attr(attrs, "action").partition("?")[0].endswith("/anomaly.js")
            ))
        ):
            self.challenge = True
        if classes & {"no-results", "no-results__message", "result--no-result"}:
            self.no_results = True
        if tag == "form" and "nav-link" in classes:
            self._form_params = {}
            self._form_is_next = False
            return
        if tag == "input" and self._form_params is not None:
            name = _attr(attrs, "name")
            value = _attr(attrs, "value")
            if name:
                self._form_params[name] = value
            if _attr(attrs, "type").lower() == "submit" and value.lower() == "next":
                self._form_is_next = True
            return
        if tag != "a" or self._anchor_kind is not None:
            return
        if "result__a" in classes:
            self._anchor_kind = "title"
        elif "result__snippet" in classes:
            self._anchor_kind = "snippet"
        else:
            return
        self._anchor_href = _attr(attrs, "href")
        self._anchor_text = []

    def handle_data(self, data: str) -> None:
        if self._anchor_kind is not None:
            self._anchor_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._form_params is not None:
            if self._form_is_next and self._form_params.get("q"):
                self.next_params = dict(self._form_params)
            self._form_params = None
            self._form_is_next = False
            return
        if tag != "a" or self._anchor_kind is None:
            return
        kind = self._anchor_kind
        href = self._anchor_href
        text = _normalize_space("".join(self._anchor_text))
        self._anchor_kind = None
        self._anchor_href = ""
        self._anchor_text = []

        url = _decode_search_url(href)
        if kind == "title":
            if text and url:
                self.results.append({"title": text, "url": url, "snippet": ""})
            return
        if not text:
            return
        for result in reversed(self.results):
            if result["snippet"]:
                continue
            if url is None or result["url"] == url:
                result["snippet"] = text
                return


_IGNORED_HTML_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
}
_VOID_HTML_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_BLOCK_HTML_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "dl",
    "dt",
    "dd",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tr",
    "ul",
}


class _ReadableHTMLParser(HTMLParser):
    def __init__(self, base_url: str = "", fragment: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.fragment = fragment
        self.fragment_found = False
        self.title_parts: list[str] = []
        self.all_parts: list[str] = []
        self.preferred_parts: list[str] = []
        self.fragment_parts: list[str] = []
        self.fragment_tail_parts: list[str] = []
        self._ignored_depth = 0
        self._head_depth = 0
        self._title_depth = 0
        self._preferred_depth = 0
        self._fragment_depth = 0
        self._link_url: str | None = None
        self._link_text: list[str] = []

    def _append(self, value: str) -> None:
        if self._ignored_depth:
            return
        self.all_parts.append(value)
        if self._preferred_depth:
            self.preferred_parts.append(value)
        if self._fragment_depth:
            self.fragment_parts.append(value)
        if self.fragment_found:
            self.fragment_tail_parts.append(value)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        if tag == "head":
            self._head_depth += 1
            return
        if tag in _IGNORED_HTML_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._title_depth += 1
            return
        if self._head_depth:
            return
        if tag == "a":
            self._link_url = _readable_link_url(
                self.base_url,
                _attr(attrs, "href"),
            )
            self._link_text = []
        if tag in {"main", "article"}:
            self._preferred_depth += 1
        if self._fragment_depth:
            if tag not in _VOID_HTML_TAGS:
                self._fragment_depth += 1
        elif (
            self.fragment
            and not self.fragment_found
            and (
                _attr(attrs, "id") == self.fragment
                or (tag == "a" and _attr(attrs, "name") == self.fragment)
            )
        ):
            self.fragment_found = True
            if tag not in _VOID_HTML_TAGS:
                self._fragment_depth = 1
        if tag in _BLOCK_HTML_TAGS:
            self._append("\n")

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() in _BLOCK_HTML_TAGS:
            self._append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._title_depth:
            self.title_parts.append(data)
            return
        if self._head_depth:
            return
        if self._link_url is not None:
            self._link_text.append(data)
        self._append(re.sub(r"\s+", " ", data))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _IGNORED_HTML_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._title_depth = max(0, self._title_depth - 1)
            return
        if tag == "head":
            self._head_depth = max(0, self._head_depth - 1)
            return
        if self._head_depth:
            return
        if tag == "a":
            link_text = _normalize_space("".join(self._link_text))
            if self._link_url and self._link_url not in link_text:
                self._append(f" <{self._link_url}>")
            self._link_url = None
            self._link_text = []
        if tag in _BLOCK_HTML_TAGS:
            self._append("\n")
        if tag in {"main", "article"}:
            self._preferred_depth = max(0, self._preferred_depth - 1)
        if self._fragment_depth and tag not in _VOID_HTML_TAGS:
            self._fragment_depth -= 1

    def readable_text(self) -> str:
        preferred = _normalize_text("".join(self.preferred_parts))
        full = _normalize_text("".join(self.all_parts))
        # Some pages use <article> for sidebar widgets (link lists, teasers)
        # while the real content lives in plain <div>s, so only trust the
        # main/article extraction when it holds a meaningful share of the page.
        if preferred and len(preferred) >= 0.2 * len(full):
            return preferred
        return full

    def title(self) -> str:
        return _normalize_space("".join(self.title_parts))

    def fragment_text(self) -> str:
        return _normalize_text("".join(self.fragment_parts))

    def fragment_tail_text(self) -> str:
        return _normalize_text("".join(self.fragment_tail_parts))


def _readable_link_url(base_url: str, href: str) -> str | None:
    if not href:
        return None
    absolute = urljoin(base_url, href)
    try:
        parsed = urlsplit(absolute)
        parsed.port
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    return absolute


def _validated_url(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WebToolError("url must be a non-empty string")
    url = value.strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise WebToolError("invalid URL: control characters are not allowed")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise WebToolError(f"invalid URL: {exc}") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise WebToolError("url scheme must be http or https")
    if not parsed.hostname:
        raise WebToolError("url must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise WebToolError("embedded URL credentials are not allowed")
    host = parsed.hostname
    try:
        host.encode("idna")
    except UnicodeError as exc:
        raise WebToolError(f"invalid URL hostname: {exc}") from exc
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))


def _create_client(timeout: float) -> httpx.Client:
    import httpx
    from .http_cancellation import CancellableClient

    return CancellableClient(
        timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
        follow_redirects=False,
        headers={
            "user-agent": f"jarv/{__version__}",
            # Decode these ourselves with an output budget; httpx's automatic
            # decoders allocate their entire expanded chunk before yielding.
            "accept-encoding": "gzip, deflate",
            "accept": (
                "text/html,application/xhtml+xml,application/json,text/plain,"
                "application/xml;q=0.9,*/*;q=0.1"
            ),
        },
    )


def _limited_response_bytes(response, max_bytes: int, cancellation_token):
    """Yield bounded decoded chunks without allocating a decompression bomb."""
    encoding = response.headers.get("content-encoding", "").strip().lower()
    if encoding not in {"", "identity", "gzip", "deflate"}:
        raise WebToolError(f"unsupported Content-Encoding: {encoding}")
    if response.is_stream_consumed:
        # A transport may supply an already-buffered response. Its content is
        # already decoded; iter_bytes slices it without invoking a decoder.
        chunks = response.iter_bytes(chunk_size=_DECODE_CHUNK_BYTES)
        encoding = ""
    else:
        chunks = response.iter_raw(chunk_size=_DECODE_CHUNK_BYTES)
    compressed = encoding in {"gzip", "deflate"}
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else None
    prefix = b""
    wire_size = decoded_size = 0
    try:
        for raw in chunks:
            if cancellation_token is not None:
                cancellation_token.throw_if_cancelled()
            wire_size += len(raw)
            if wire_size > max_bytes:
                raise WebToolError(f"response exceeds {max_bytes} byte limit")
            if not compressed:
                yield raw
                continue
            data = prefix + raw
            prefix = b""
            if decoder is None:
                # HTTP deflate is normally zlib-wrapped; some servers send
                # raw DEFLATE. Wait for two bytes to recognize the wrapper.
                if len(data) < 2:
                    prefix = data
                    continue
                wrapped = data[0] & 15 == 8 and int.from_bytes(data[:2], "big") % 31 == 0
                decoder = zlib.decompressobj(zlib.MAX_WBITS if wrapped else -zlib.MAX_WBITS)
            drain = False
            while data or drain:
                if cancellation_token is not None:
                    cancellation_token.throw_if_cancelled()
                if decoder.eof:
                    if encoding != "gzip":
                        raise WebToolError("invalid compressed response: trailing data")
                    # Gzip permits concatenated members. The cumulative
                    # decoded budget applies across every member.
                    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
                limit = min(_DECODE_CHUNK_BYTES, max_bytes - decoded_size + 1)
                decoded = decoder.decompress(data, limit)
                decoded_size += len(decoded)
                if decoded_size > max_bytes:
                    raise WebToolError(f"response exceeds {max_bytes} byte limit")
                data = decoder.unused_data if decoder.eof else decoder.unconsumed_tail
                drain = len(decoded) == limit and not decoder.eof
                if decoded:
                    yield decoded
        if compressed and (decoder is None or not decoder.eof):
            raise WebToolError("invalid compressed response: incomplete stream")
    except zlib.error as exc:
        raise WebToolError(f"invalid compressed response: {exc}") from exc


def _request_bytes(
    url: str,
    *,
    timeout: float,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    params: dict[str, str] | None = None,
    form_data: dict[str, str] | None = None,
    cancellation_token: CancellationToken | None = None,
) -> tuple[str, str, bytes]:
    response = _request_response(
        url, timeout=timeout, max_response_bytes=max_response_bytes,
        params=params, form_data=form_data, cancellation_token=cancellation_token,
    )
    return response.final_url, response.content_type, response.body


def _request_response(
    url: str,
    *,
    timeout: float,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    params: dict[str, str] | None = None,
    form_data: dict[str, str] | None = None,
    cancellation_token: CancellationToken | None = None,
    allow_error_response: bool = False,
    _budget: SearchBudget | None = None,
) -> WebResponse:
    from .http_cancellation import cancellation_scope

    def request(budget: SearchBudget) -> WebResponse:
        try:
            budget.check()
            with cancellation_scope(budget.token):
                response = _request_response_with_budget(
                    url, timeout=timeout, budget=budget,
                    max_response_bytes=max_response_bytes,
                    params=params, form_data=form_data,
                    allow_error_response=allow_error_response,
                )
            budget.check()
            return response
        except BaseException:
            # The same child token stops HTTP I/O for either reason. Preserve
            # user cancellation, but translate an elapsed deadline correctly.
            budget.check()
            raise

    if _budget is not None:
        return request(_budget)
    try:
        # URL reads share the search budget's cancellation and cleanup
        # machinery. One budget covers every redirect and response chunk.
        with SearchBudget(timeout, cancellation_token) as budget:
            return request(budget)
    except SearchDeadlineExceeded as exc:
        raise WebToolError(
            f"request timed out after {timeout:g} seconds", kind="timeout",
        ) from exc


def _request_response_with_budget(
    url: str,
    *,
    timeout: float,
    budget: SearchBudget,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    params: dict[str, str] | None = None,
    form_data: dict[str, str] | None = None,
    allow_error_response: bool = False,
) -> WebResponse:
    import httpx

    cancellation_token = budget.token
    client = _create_client(budget.remaining())
    unregister = cancellation_token.register(client.close)
    try:
        current_url = _validated_url(url)
        current_params = params
        current_form_data = form_data
        method = "POST" if form_data is not None else "GET"
        for redirect_count in range(MAX_REDIRECTS + 1):
            remaining = budget.remaining()
            client.cookies.clear()
            with client.stream(
                method,
                current_url,
                params=current_params,
                data=current_form_data,
                timeout=httpx.Timeout(remaining, connect=min(remaining, 10.0)),
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise WebToolError(
                            f"redirect response {response.status_code} had no Location header"
                        )
                    if redirect_count >= MAX_REDIRECTS:
                        raise WebToolError(f"too many redirects (maximum {MAX_REDIRECTS})")
                    current_url = _validated_url(urljoin(str(response.url), location))
                    if response.status_code == 303 or (
                        response.status_code in {301, 302} and method == "POST"
                    ):
                        method = "GET"
                        current_form_data = None
                    current_params = None
                    continue
                if response.status_code >= 400 and not allow_error_response:
                    message = f"HTTP {response.status_code} {response.reason_phrase}".strip()
                    raise WebToolError(
                        message, status_code=response.status_code,
                        retry_after=response.headers.get("retry-after"),
                    )

                # Rate limits and transient server errors can be classified
                # without downloading an arbitrary error body.
                if allow_error_response and response.status_code in {429, 500, 502, 503, 504}:
                    return WebResponse(
                        str(response.url), response.headers.get("content-type", ""),
                        b"", response.status_code, response.headers.get("retry-after"),
                    )

                content_type = response.headers.get("content-type", "")
                media_type = _media_type(content_type)
                effective_max_bytes = max_response_bytes
                if (
                    max_response_bytes > MAX_RESPONSE_BYTES
                    and (
                        _is_textual_media_type(media_type)
                        or is_pdf_media_type(media_type)
                    )
                ):
                    effective_max_bytes = MAX_RESPONSE_BYTES

                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError:
                        declared_size = 0
                    if declared_size > effective_max_bytes:
                        raise WebToolError(
                            f"response exceeds {effective_max_bytes} byte limit"
                        )

                chunks: list[bytes] = []
                size = 0
                prefix = b""
                for chunk in _limited_response_bytes(response, effective_max_bytes, cancellation_token):
                    if cancellation_token is not None:
                        cancellation_token.throw_if_cancelled()
                    size += len(chunk)
                    if size > effective_max_bytes:
                        raise WebToolError(
                            f"response exceeds {effective_max_bytes} byte limit"
                        )
                    chunks.append(chunk)
                    if (
                        effective_max_bytes > MAX_RESPONSE_BYTES
                        and len(prefix) < len(PDF_MAGIC)
                    ):
                        prefix = (prefix + chunk)[: len(PDF_MAGIC)]
                        if len(prefix) == len(PDF_MAGIC) and prefix == PDF_MAGIC:
                            effective_max_bytes = MAX_RESPONSE_BYTES
                            if size > effective_max_bytes:
                                raise WebToolError(
                                    f"response exceeds {effective_max_bytes} byte limit"
                                )
                return WebResponse(
                    str(response.url), content_type, b"".join(chunks),
                    response.status_code, response.headers.get("retry-after"),
                )
        raise WebToolError(f"too many redirects (maximum {MAX_REDIRECTS})")
    except httpx.TimeoutException as exc:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        raise WebToolError(
            f"request timed out after {timeout:g} seconds", kind="timeout", retryable=True,
        ) from exc
    except httpx.HTTPError as exc:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        raise WebToolError(
            f"request failed: {exc}",
            retryable=isinstance(exc, (httpx.NetworkError, httpx.RemoteProtocolError)),
        ) from exc
    except (httpx.InvalidURL, UnicodeError) as exc:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        raise WebToolError(f"invalid URL: {exc}") from exc
    except RuntimeError:
        # Closing a client at cancellation/deadline can race with stream setup.
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        raise
    finally:
        unregister()
        client.close()


def _decode_body(body: bytes, content_type: str) -> str:
    match = re.search(r"charset\s*=\s*[\"']?([^;\"'\s]+)", content_type, re.I)
    encoding = match.group(1) if match else "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _media_type(content_type: str) -> str:
    return content_type.partition(";")[0].strip().lower()


def _is_textual_media_type(media_type: str) -> bool:
    return (
        media_type.startswith("text/")
        or media_type in {
            "application/json",
            "application/javascript",
            "application/xml",
            "application/rss+xml",
            "application/atom+xml",
            "application/xhtml+xml",
        }
        or media_type.endswith("+json")
        or media_type.endswith("+xml")
    )


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        if value.strip().isdigit():
            seconds = float(value)
        else:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _retry_delay(attempt: int) -> float:
    return (2.0, 5.0)[min(attempt, 1)] + random.uniform(0.0, 0.5)


def _parse_search_response(response: WebResponse) -> _DuckDuckGoHTMLParser:
    status = response.status_code
    if status == 429 or status in {500, 502, 503, 504}:
        raise WebToolError(
            f"DuckDuckGo {'rate limited the search' if status == 429 else 'is temporarily unavailable'} "
            f"(HTTP {status})",
            kind="rate_limited" if status == 429 else "server_error",
            status_code=status, retry_after=response.retry_after, retryable=True,
        )
    parser = _DuckDuckGoHTMLParser()
    media_type = _media_type(response.content_type)
    if media_type in {"", "text/html", "application/xhtml+xml"}:
        parser.feed(_decode_body(response.body, response.content_type))
    if parser.challenge:
        cooldown = max(SEARCH_CHALLENGE_COOLDOWN, _retry_after_seconds(response.retry_after) or 0.0)
        raise WebToolError(
            f"DuckDuckGo requires human verification (HTTP {status}); "
            f"searches paused for {math.ceil(cooldown)} seconds. Try again later.",
            kind="blocked", status_code=status, retry_after=response.retry_after,
        )
    if status >= 400:
        raise WebToolError(
            f"DuckDuckGo returned HTTP {status}", kind="http_error",
            status_code=status, retry_after=response.retry_after,
        )
    if 200 <= status < 300 and parser.results:
        return parser
    if status == 200 and parser.no_results:
        return parser
    raise WebToolError(
        f"DuckDuckGo returned an unexpected search page (HTTP {status}); "
        "could not determine whether results exist",
        kind="unexpected_response", status_code=status,
    )


def _search_page(
    budget: SearchBudget,
    *,
    params: dict[str, str] | None,
    form_data: dict[str, str] | None,
    min_interval: float,
    retries_remaining: int,
) -> tuple[_DuckDuckGoHTMLParser, int]:
    while True:
        retry_delay: float | None = None
        try:
            with _SEARCH_COORDINATOR.request(budget, min_interval):
                try:
                    response = _request_response(
                        DUCKDUCKGO_HTML_URL, timeout=budget.remaining(),
                        params=params, form_data=form_data,
                        cancellation_token=budget.token, allow_error_response=True,
                        _budget=budget,
                    )
                    budget.check()
                    parser = _parse_search_response(response)
                    budget.check()
                    return parser, retries_remaining
                except WebToolError as exc:
                    budget.check()
                    if exc.retryable:
                        retry_delay = _retry_after_seconds(exc.retry_after)
                        if retry_delay is None:
                            retry_delay = _retry_delay(MAX_SEARCH_RETRIES - retries_remaining)
                    if exc.kind == "blocked":
                        _SEARCH_COORDINATOR.cooldown(
                            max(SEARCH_CHALLENGE_COOLDOWN, _retry_after_seconds(exc.retry_after) or 0.0),
                            "human verification",
                        )
                    elif exc.kind == "rate_limited" or (exc.retryable and exc.retry_after is not None):
                        _SEARCH_COORDINATOR.cooldown(
                            retry_delay or 0.0,
                            "rate limiting" if exc.kind == "rate_limited" else "temporary server failure",
                        )
                    raise
        except WebToolError as exc:
            if not exc.retryable or retries_remaining <= 0:
                raise
            delay = retry_delay or 0.0
            if delay >= budget.remaining():
                raise WebToolError(
                    f"{exc}; retry delay exceeds the remaining search time budget. Try again later.",
                    kind=exc.kind, status_code=exc.status_code, retry_after=exc.retry_after,
                ) from exc
            retries_remaining -= 1
            budget.wait(delay)
        except TurnCancelled:
            # The child token also cancels HTTP I/O when the overall budget
            # expires; user cancellation always takes precedence.
            budget.check()
            raise


def search_web(
    query: str,
    max_results: int = DEFAULT_SEARCH_RESULTS,
    *,
    offset: int = 0,
    timeout: float,
    cancellation_token: CancellationToken | None = None,
    min_interval: float = DEFAULT_SEARCH_INTERVAL,
    max_pages: int = DEFAULT_SEARCH_MAX_PAGES,
) -> str:
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    skipped = 0
    request_params: dict[str, str] | None = {"q": query}
    form_data: dict[str, str] | None = None
    seen_pages: set[tuple[tuple[str, str], ...]] = set()
    page_count = 0
    retries_remaining = MAX_SEARCH_RETRIES
    failure: WebToolError | None = None

    try:
        with SearchBudget(timeout, cancellation_token) as budget:
            while len(unique) < max_results:
                if page_count >= max_pages:
                    raise WebToolError(
                        f"search stopped at the {max_pages}-page limit", kind="incomplete",
                    )
                parser, retries_remaining = _search_page(
                    budget, params=request_params, form_data=form_data,
                    min_interval=min_interval, retries_remaining=retries_remaining,
                )
                page_count += 1
                if not parser.results:
                    break

                for result in parser.results:
                    if result["url"] in seen:
                        continue
                    seen.add(result["url"])
                    if skipped < offset:
                        skipped += 1
                        continue
                    unique.append(result)
                    if len(unique) >= max_results:
                        break

                if len(unique) >= max_results or parser.next_params is None:
                    break
                page_key = tuple(sorted(parser.next_params.items()))
                if page_key in seen_pages:
                    raise WebToolError(
                        "DuckDuckGo repeated a pagination cursor; search stopped", kind="incomplete",
                    )
                seen_pages.add(page_key)
                request_params = None
                form_data = parser.next_params
            budget.check()
    except SearchDeadlineExceeded:
        failure = WebToolError(
            f"search exceeded its {timeout:g}-second total time budget", kind="timeout",
        )
    except SearchCooldownError as exc:
        failure = WebToolError(
            f"DuckDuckGo searches paused after {exc.reason}; try again in {math.ceil(exc.remaining)} seconds",
            kind="cooldown",
        )
    except WebToolError as exc:
        failure = exc

    if cancellation_token is not None:
        cancellation_token.throw_if_cancelled()
    if failure is not None and not unique:
        raise failure

    lines = [
        f"Query: {query}",
        f"Offset: {offset}",
        f"Source pages: {page_count}",
        "",
    ]
    if failure is not None:
        lines.extend([f"Partial results: {failure.kind}: {failure}", ""])
    if not unique:
        lines.append(f"No search results found{' at offset ' + str(offset) if offset else ''}.")
    for index, result in enumerate(unique, offset + 1):
        lines.extend([
            f"{index}. {result['title']}",
            f"URL: {result['url']}",
        ])
        if result["snippet"]:
            lines.append(f"Snippet: {result['snippet']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def url_fragment(url: str) -> str:
    """The decoded #fragment of a URL, or "" when absent or unparseable."""
    if not isinstance(url, str):
        return ""
    try:
        fragment = urlsplit(url.strip()).fragment
    except ValueError:
        return ""
    return unquote(fragment)


def fetch_web_content(
    url: str,
    *,
    timeout: float,
    cancellation_token: CancellationToken | None = None,
) -> FetchedWebContent:
    requested_url = _validated_url(url)
    final_url, content_type, body = _request_bytes(
        requested_url,
        timeout=timeout,
        cancellation_token=cancellation_token,
    )
    return web_content_from_bytes(
        requested_url,
        final_url,
        content_type,
        body,
        fragment=url_fragment(url),
    )


def fetch_web_bytes(
    url: str,
    *,
    timeout: float,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    cancellation_token: CancellationToken | None = None,
) -> FetchedWebBytes:
    requested_url = _validated_url(url)
    final_url, content_type, body = _request_bytes(
        requested_url,
        timeout=timeout,
        max_response_bytes=max_response_bytes,
        cancellation_token=cancellation_token,
    )
    return FetchedWebBytes(
        requested_url=requested_url,
        final_url=final_url,
        content_type=content_type,
        media_type=_media_type(content_type),
        body=body,
    )


def web_content_from_bytes(
    requested_url: str,
    final_url: str,
    content_type: str,
    body: bytes,
    *,
    fragment: str = "",
) -> FetchedWebContent:
    media_type = _media_type(content_type)
    if not _is_textual_media_type(media_type):
        label = media_type or "unknown content type"
        raise WebToolError(f"unsupported non-text content type: {label}")

    decoded = _decode_body(body, content_type)
    title = ""
    fragment_applied = False
    if media_type in {"text/html", "application/xhtml+xml"}:
        parser = _ReadableHTMLParser(final_url, fragment=fragment)
        parser.feed(decoded)
        text = parser.readable_text()
        title = parser.title()
        if fragment and parser.fragment_found:
            section = parser.fragment_text()
            if len(section) < FRAGMENT_MIN_CHARS:
                # Heading-style anchors (<h2 id=...>) hold only their own
                # label; the section content follows as siblings, so take
                # everything from the anchor onward instead.
                tail = parser.fragment_tail_text()
                if len(tail) > len(section):
                    section = tail
            if section:
                text = section
                fragment_applied = True
    elif media_type == "application/json" or media_type.endswith("+json"):
        text = _bounded_json_format(decoded)
    else:
        text = _normalize_text(decoded)
    if not text:
        raise WebToolError("response contained no readable text")

    return FetchedWebContent(
        requested_url=requested_url,
        final_url=final_url,
        media_type=media_type,
        title=title,
        text=text,
        fragment=fragment,
        fragment_applied=fragment_applied,
    )


def _bounded_json_format(decoded: str) -> str:
    """Pretty-print ordinary JSON without expanding arbitrary input in memory.

    Expensive or invalid inputs retain their original text, so pagination never
    loses data merely because indentation would exceed the formatting budget.
    """
    # A character cannot occupy fewer than one UTF-8 byte. Check that cheap
    # bound before allocating a bounded encoding for the exact byte count.
    if len(decoded) > MAX_JSON_FORMAT_INPUT_BYTES:
        return decoded
    try:
        input_bytes = len(decoded.encode("utf-8"))
    except UnicodeError:
        return decoded
    if input_bytes > MAX_JSON_FORMAT_INPUT_BYTES:
        return decoded
    depth = 0
    quoted = escaped = False
    for char in decoded:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_FORMAT_DEPTH:
                return decoded
        elif char in "]}":
            depth -= 1
    try:
        value = json.loads(decoded)
        parts = []
        size = 0
        for part in json.JSONEncoder(indent=2, ensure_ascii=False).iterencode(value):
            if len(part) > MAX_JSON_FORMAT_OUTPUT_BYTES - size:
                return decoded
            size += len(part.encode("utf-8"))
            if size > MAX_JSON_FORMAT_OUTPUT_BYTES:
                return decoded
            parts.append(part)
        return "".join(parts)
    except (ValueError, RecursionError):
        return decoded


def fragment_note(content: FetchedWebContent) -> str | None:
    """A user-visible line describing how the URL fragment was handled."""
    if not content.fragment:
        return None
    if content.fragment_applied:
        return f"Fragment: #{content.fragment} (section extracted)"
    return f"Fragment: #{content.fragment} not found; returning full page"


def fetch_web(
    url: str,
    *,
    timeout: float,
    cancellation_token: CancellationToken | None = None,
) -> str:
    content = fetch_web_content(
        url,
        timeout=timeout,
        cancellation_token=cancellation_token,
    )
    lines = [
        f"Requested URL: {content.requested_url}",
        f"Final URL: {content.final_url}",
        f"Content-Type: {content.media_type or 'unknown'}",
    ]
    if content.title:
        lines.append(f"Title: {content.title}")
    note = fragment_note(content)
    if note:
        lines.append(note)
    lines.extend(["", content.text])
    return "\n".join(lines)


def web_timeout(config: dict) -> float:
    """Normalize the shared timeout for web searches and URL reads."""
    try:
        timeout = float(get_setting(config, "web_timeout"))
    except (TypeError, ValueError):
        timeout = float(DEFAULT_CONFIG["web_timeout"])
    if not math.isfinite(timeout) or timeout <= 0:
        timeout = float(DEFAULT_CONFIG["web_timeout"])
    return timeout


def dispatch_web_tool(
    name: str,
    args: dict[str, Any],
    config: dict,
    *,
    cancellation_token: CancellationToken | None = None,
) -> str:
    timeout = web_timeout(config)

    try:
        if name == "web_search":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return with_tool_outcome("[tool argument error: query must be a non-empty string]", "failed")
            max_results = args.get("max_results", DEFAULT_SEARCH_RESULTS)
            if max_results is None:
                max_results = DEFAULT_SEARCH_RESULTS
            if isinstance(max_results, bool) or not isinstance(max_results, int):
                return with_tool_outcome("[tool argument error: max_results must be an integer]", "failed")
            if max_results <= 0:
                return with_tool_outcome("[tool argument error: max_results must be a positive integer]", "failed")
            if max_results > MAX_SEARCH_RESULTS:
                return with_tool_outcome(
                    "[tool argument error: max_results must be at most "
                    f"{MAX_SEARCH_RESULTS}]", "failed",
                )
            offset = args.get("offset", 0)
            if offset is None:
                offset = 0
            if isinstance(offset, bool) or not isinstance(offset, int):
                return with_tool_outcome("[tool argument error: offset must be an integer]", "failed")
            if offset < 0:
                return with_tool_outcome("[tool argument error: offset must be a non-negative integer]", "failed")
            interval = get_setting(config, "web_search_interval")
            if type(interval) is not int or interval <= 0:
                interval = DEFAULT_CONFIG["web_search_interval"]
            max_pages = get_setting(config, "web_search_max_pages")
            if type(max_pages) is not int or max_pages <= 0:
                max_pages = DEFAULT_CONFIG["web_search_max_pages"]
            output = search_web(
                query.strip(),
                max_results,
                offset=offset,
                timeout=timeout,
                cancellation_token=cancellation_token,
                min_interval=interval,
                max_pages=max_pages,
            )
            return with_tool_outcome(output, "success")
    except WebToolError as exc:
        return with_tool_outcome(
            f"[web error: {exc.kind}: {exc}]",
            "timed_out" if exc.kind == "timeout" else "failed",
        )
    return with_tool_outcome(f"[unknown tool: {name}]", "failed")
