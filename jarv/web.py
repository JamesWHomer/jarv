"""Keyless web search and bounded web page fetching."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urljoin, urlsplit, urlunsplit

import httpx

from . import __version__
from .cancellation import CancellationToken
from .config import DEFAULT_CONFIG, get_setting


DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
SEARCH_ENGINE_LABEL = "DuckDuckGo"
from .pdf_extract import PDF_MAGIC, is_pdf_bytes, is_pdf_media_type
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 20

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


@dataclass(frozen=True)
class FetchedWebContent:
    requested_url: str
    final_url: str
    media_type: str
    title: str
    text: str


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

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag == "form" and "nav-link" in _class_tokens(attrs):
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
        classes = _class_tokens(attrs)
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
    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.all_parts: list[str] = []
        self.preferred_parts: list[str] = []
        self._ignored_depth = 0
        self._head_depth = 0
        self._title_depth = 0
        self._preferred_depth = 0
        self._link_url: str | None = None
        self._link_text: list[str] = []

    def _append(self, value: str) -> None:
        if self._ignored_depth:
            return
        self.all_parts.append(value)
        if self._preferred_depth:
            self.preferred_parts.append(value)

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

    def readable_text(self) -> str:
        preferred = _normalize_text("".join(self.preferred_parts))
        if preferred:
            return preferred
        return _normalize_text("".join(self.all_parts))

    def title(self) -> str:
        return _normalize_space("".join(self.title_parts))


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
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))


def _create_client(timeout: float) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0)),
        follow_redirects=False,
        headers={
            "user-agent": f"jarv/{__version__}",
            "accept": (
                "text/html,application/xhtml+xml,application/json,text/plain,"
                "application/xml;q=0.9,*/*;q=0.1"
            ),
        },
    )


def _request_bytes(
    url: str,
    *,
    timeout: float,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    params: dict[str, str] | None = None,
    form_data: dict[str, str] | None = None,
    cancellation_token: CancellationToken | None = None,
) -> tuple[str, str, bytes]:
    client = _create_client(timeout)
    unregister = (
        cancellation_token.register(client.close)
        if cancellation_token is not None
        else lambda: None
    )
    try:
        current_url = _validated_url(url)
        current_params = params
        current_form_data = form_data
        method = "POST" if form_data is not None else "GET"
        for redirect_count in range(MAX_REDIRECTS + 1):
            if cancellation_token is not None:
                cancellation_token.throw_if_cancelled()
            client.cookies.clear()
            with client.stream(
                method,
                current_url,
                params=current_params,
                data=current_form_data,
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
                if response.status_code >= 400:
                    message = f"HTTP {response.status_code} {response.reason_phrase}".strip()
                    raise WebToolError(message)

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
                for chunk in response.iter_bytes():
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
                return str(response.url), content_type, b"".join(chunks)
        raise WebToolError(f"too many redirects (maximum {MAX_REDIRECTS})")
    except httpx.TimeoutException as exc:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        raise WebToolError(f"request timed out after {timeout:g} seconds") from exc
    except httpx.HTTPError as exc:
        if cancellation_token is not None:
            cancellation_token.throw_if_cancelled()
        raise WebToolError(f"request failed: {exc}") from exc
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


def search_web(
    query: str,
    max_results: int = DEFAULT_SEARCH_RESULTS,
    *,
    offset: int = 0,
    timeout: float,
    cancellation_token: CancellationToken | None = None,
) -> str:
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    skipped = 0
    request_params: dict[str, str] | None = {"q": query}
    form_data: dict[str, str] | None = None
    seen_pages: set[tuple[tuple[str, str], ...]] = set()
    source_urls: list[str] = []

    while len(unique) < max_results:
        final_url, content_type, body = _request_bytes(
            DUCKDUCKGO_HTML_URL,
            timeout=timeout,
            params=request_params,
            form_data=form_data,
            cancellation_token=cancellation_token,
        )
        source_urls.append(final_url)
        parser = _DuckDuckGoHTMLParser()
        parser.feed(_decode_body(body, content_type))
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
            break
        seen_pages.add(page_key)
        request_params = None
        form_data = parser.next_params

    if not unique:
        if seen:
            raise WebToolError(f"no search results found at offset {offset}")
        raise WebToolError("no search results found; DuckDuckGo may have returned a challenge page")

    lines = [
        f"Query: {query}",
        f"Offset: {offset}",
        f"Source pages: {len(source_urls)}",
        "",
    ]
    for index, result in enumerate(unique, offset + 1):
        lines.extend([
            f"{index}. {result['title']}",
            f"URL: {result['url']}",
        ])
        if result["snippet"]:
            lines.append(f"Snippet: {result['snippet']}")
        lines.append("")
    return "\n".join(lines).rstrip()


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
    return web_content_from_bytes(requested_url, final_url, content_type, body)


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
) -> FetchedWebContent:
    media_type = _media_type(content_type)
    if not _is_textual_media_type(media_type):
        label = media_type or "unknown content type"
        raise WebToolError(f"unsupported non-text content type: {label}")

    decoded = _decode_body(body, content_type)
    title = ""
    if media_type in {"text/html", "application/xhtml+xml"}:
        parser = _ReadableHTMLParser(final_url)
        parser.feed(decoded)
        text = parser.readable_text()
        title = parser.title()
    elif media_type == "application/json" or media_type.endswith("+json"):
        try:
            text = json.dumps(json.loads(decoded), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            text = _normalize_text(decoded)
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
    )


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
    lines.extend(["", content.text])
    return "\n".join(lines)


def dispatch_web_tool(
    name: str,
    args: dict[str, Any],
    config: dict,
    *,
    cancellation_token: CancellationToken | None = None,
) -> str:
    try:
        timeout = float(get_setting(config, "web_timeout"))
    except (TypeError, ValueError):
        timeout = float(DEFAULT_CONFIG["web_timeout"])
    if timeout <= 0:
        timeout = float(DEFAULT_CONFIG["web_timeout"])

    try:
        if name == "web_search":
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return "[tool argument error: query must be a non-empty string]"
            max_results = args.get("max_results", DEFAULT_SEARCH_RESULTS)
            if max_results is None:
                max_results = DEFAULT_SEARCH_RESULTS
            if isinstance(max_results, bool) or not isinstance(max_results, int):
                return "[tool argument error: max_results must be an integer]"
            if max_results <= 0:
                return "[tool argument error: max_results must be a positive integer]"
            if max_results > MAX_SEARCH_RESULTS:
                return (
                    "[tool argument error: max_results must be at most "
                    f"{MAX_SEARCH_RESULTS}]"
                )
            offset = args.get("offset", 0)
            if offset is None:
                offset = 0
            if isinstance(offset, bool) or not isinstance(offset, int):
                return "[tool argument error: offset must be an integer]"
            if offset < 0:
                return "[tool argument error: offset must be a non-negative integer]"
            return search_web(
                query.strip(),
                max_results,
                offset=offset,
                timeout=timeout,
                cancellation_token=cancellation_token,
            )
    except WebToolError as exc:
        return f"[web error: {exc}]"
    return f"[unknown tool: {name}]"
