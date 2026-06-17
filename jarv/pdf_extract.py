from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from threading import Lock
from typing import Any


PDF_MAGIC = b"%PDF-"

_PDF_READER_CLASS: Any | None = None
_PDF_READER_LOCK = Lock()


class PdfExtractionError(Exception):
    """A user-visible PDF extraction failure."""


@dataclass(frozen=True)
class ExtractedPdfText:
    text: str
    page_count: int
    metadata: tuple[str, ...]


def is_pdf_bytes(data: bytes) -> bool:
    return data.startswith(PDF_MAGIC)


def _load_pdf_reader_class() -> Any:
    global _PDF_READER_CLASS
    if _PDF_READER_CLASS is not None:
        return _PDF_READER_CLASS
    with _PDF_READER_LOCK:
        if _PDF_READER_CLASS is None:
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise PdfExtractionError(
                    "pypdf is required to read PDF files"
                ) from exc
            _PDF_READER_CLASS = PdfReader
    return _PDF_READER_CLASS


def _normalize_page_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _metadata_value(value: object) -> str:
    return " ".join(str(value).split())


def _pdf_metadata(reader: Any, page_count: int) -> tuple[str, ...]:
    lines = [
        f"PDF pages: {page_count}",
        "PDF extraction: embedded text",
    ]
    try:
        metadata = reader.metadata
    except Exception:
        metadata = None
    if metadata is None:
        return tuple(lines)

    for label, attr in (
        ("Title", "title"),
        ("Author", "author"),
        ("Subject", "subject"),
        ("Creator", "creator"),
        ("Producer", "producer"),
    ):
        try:
            value = getattr(metadata, attr, None)
        except Exception:
            value = None
        if value:
            lines.append(f"PDF {label}: {_metadata_value(value)}")
    return tuple(lines)


def extract_pdf_text(data: bytes) -> ExtractedPdfText:
    PdfReader = _load_pdf_reader_class()
    try:
        reader = PdfReader(BytesIO(data), strict=False)
    except Exception as exc:
        raise PdfExtractionError(f"could not parse PDF: {exc}") from exc

    try:
        if reader.is_encrypted:
            try:
                decrypt_result = reader.decrypt("")
            except Exception as exc:
                raise PdfExtractionError(
                    "PDF is encrypted and could not be read"
                ) from exc
            if not decrypt_result:
                raise PdfExtractionError("PDF is encrypted and could not be read")
        page_count = len(reader.pages)
    except PdfExtractionError:
        raise
    except Exception as exc:
        raise PdfExtractionError(f"could not read PDF pages: {exc}") from exc

    parts: list[str] = []
    has_text = False
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            page_text = _normalize_page_text(page.extract_text() or "")
        except Exception as exc:
            raise PdfExtractionError(
                f"could not extract text from PDF page {page_index}: {exc}"
            ) from exc
        parts.append(f"--- Page {page_index} of {page_count} ---")
        if page_text:
            parts.append(page_text)
            has_text = True

    if not has_text:
        raise PdfExtractionError(
            "PDF contained no extractable text; scanned/image-only PDFs are not supported"
        )

    return ExtractedPdfText(
        text="\n\n".join(parts),
        page_count=page_count,
        metadata=_pdf_metadata(reader, page_count),
    )
