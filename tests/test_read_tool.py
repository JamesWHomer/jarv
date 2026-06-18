import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest

from jarv.artifacts import ArtifactStore
from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.config import DEFAULT_CONFIG
from jarv.model_catalog import ModelImageCapability
from jarv.read_tool import (
    READ_TOOL,
    dispatch_read_batch,
    dispatch_read_tool,
    read_tool_for_config,
    retain_command_output,
)
from jarv.retained_outputs import (
    RetainedOutputStore,
    load_retained_output_store,
    save_retained_output_store,
)
from jarv.web import FetchedWebBytes


PNG_BYTES = b"\x89PNG\r\n\x1a\npngdata"
WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBPwebpdata"


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _minimal_pdf(pages: list[str]) -> bytes:
    kids = " ".join(f"{4 + index * 2} 0 R" for index in range(len(pages)))
    objects: list[tuple[int, bytes]] = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (
            2,
            f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(
                "ascii"
            ),
        ),
        (3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    for index, text in enumerate(pages):
        page_id = 4 + index * 2
        content_id = page_id + 1
        stream = (
            f"BT /F1 12 Tf 72 720 Td ({_escape_pdf_text(text)}) Tj ET"
        ).encode("ascii")
        objects.append(
            (
                page_id,
                (
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
                ).encode("ascii"),
            )
        )
        objects.append(
            (
                content_id,
                b"<< /Length "
                + str(len(stream)).encode("ascii")
                + b" >>\nstream\n"
                + stream
                + b"\nendstream",
            )
        )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, content in objects:
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("ascii"))
        pdf.extend(content)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


def _read(args, *, artifacts=None, retained=None, visible=None, config=None):
    return dispatch_read_tool(
        args,
        visible_labels=visible or set(),
        artifact_store=artifacts or ArtifactStore(),
        retained_store=retained or RetainedOutputStore(),
        config=config or DEFAULT_CONFIG,
    )


def test_read_schema_exposes_input_offset_and_size():
    assert "PDFs with embedded text" in READ_TOOL["description"]
    assert "image-capable models" in READ_TOOL["description"]
    assert "image reads" in READ_TOOL["description"]
    parameters = READ_TOOL["parameters"]
    assert parameters["required"] == ["input"]
    assert parameters["properties"]["offset"]["minimum"] == 0
    assert parameters["properties"]["size"]["minimum"] == 1
    assert parameters["properties"]["size"]["maximum"] == 200000
    assert parameters["additionalProperties"] is False


def test_read_tool_for_image_capable_model_includes_image_description(monkeypatch):
    monkeypatch.setattr(
        "jarv.read_tool.get_image_output_capability",
        lambda _config: ModelImageCapability(True, "responses"),
    )

    tool = read_tool_for_config(DEFAULT_CONFIG)

    assert tool is not READ_TOOL
    assert tool["parameters"] is not READ_TOOL["parameters"]
    assert "image-capable models" in tool["description"]
    assert "image reads" in tool["description"]


def test_read_tool_for_text_only_model_omits_image_description(monkeypatch):
    original_description = READ_TOOL["description"]
    monkeypatch.setattr(
        "jarv.read_tool.get_image_output_capability",
        lambda _config: ModelImageCapability(False, reason="text-only model"),
    )

    tool = read_tool_for_config(DEFAULT_CONFIG)

    assert "PDFs with embedded text" in tool["description"]
    assert "image-capable models" not in tool["description"]
    assert "image reads" not in tool["description"]
    assert READ_TOOL["description"] == original_description


def test_read_rejects_non_object_arguments():
    assert "must be an object" in dispatch_read_tool(
        [],
        visible_labels=set(),
        artifact_store=ArtifactStore(),
        retained_store=RetainedOutputStore(),
        config=DEFAULT_CONFIG,
    )


def test_read_retained_output_pages_exact_characters_and_reports_metadata():
    retained = RetainedOutputStore()
    output_id = retained.put("0123456789")

    output = _read(
        {"input": output_id, "offset": 2, "size": 4},
        retained=retained,
        config={**DEFAULT_CONFIG, "max_tool_output_chars": 2},
    )

    assert "Offset: 2" in output
    assert "Requested size: 4" in output
    assert "Returned size: 4" in output
    assert "Total size: 10" in output
    assert "EOF: false" in output
    assert "Next offset: 6" in output
    assert output.endswith("2345")


def test_read_default_size_uses_config_and_eof_allows_empty_final_page():
    retained = RetainedOutputStore()
    output_id = retained.put("abcdef")
    config = {**DEFAULT_CONFIG, "max_tool_output_chars": 3}

    first = _read({"input": output_id}, retained=retained, config=config)
    final = _read(
        {"input": output_id, "offset": 6},
        retained=retained,
        config=config,
    )

    assert first.endswith("abc")
    assert "Next offset: 3" in first
    assert "Returned size: 0" in final
    assert "EOF: true" in final
    assert "Next offset: none" in final


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"input": "x", "offset": -1}, "non-negative integer"),
        ({"input": "x", "offset": "1"}, "offset must be an integer"),
        ({"input": "x", "size": 0}, "positive integer"),
        ({"input": "x", "size": True}, "size must be an integer"),
        ({"input": "x", "size": 200001}, "at most 200000"),
    ],
)
def test_read_validates_paging_arguments(args, message):
    assert message in _read(args)


def test_read_treats_null_optional_values_as_defaults():
    retained = RetainedOutputStore()
    output_id = retained.put("abcdef")

    output = _read(
        {"input": output_id, "offset": None, "size": None},
        retained=retained,
        config={**DEFAULT_CONFIG, "max_tool_output_chars": 3},
    )

    assert "Offset: 0" in output
    assert "Requested size: 3" in output
    assert output.endswith("abc")


def test_read_rejects_offset_beyond_end():
    retained = RetainedOutputStore()
    output_id = retained.put("abc")

    assert "beyond end" in _read(
        {"input": output_id, "offset": 4},
        retained=retained,
    )


def test_read_resolver_prefers_visible_artifact_over_matching_file(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("file content", encoding="utf-8")
    artifacts = ArtifactStore()
    artifacts.put(str(path), "artifact content", "summary", "child")

    output = _read(
        {"input": str(path), "size": 100},
        artifacts=artifacts,
        visible={str(path)},
    )

    assert "Source: artifact" in output
    assert output.endswith("artifact content")


def test_read_preserves_artifact_visibility():
    artifacts = ArtifactStore()
    artifacts.put("secret", "content", "summary", "parent")

    assert "not visible" in _read(
        {"input": "secret"},
        artifacts=artifacts,
    )


def test_read_local_relative_file_and_replaces_invalid_utf8(tmp_path, monkeypatch):
    path = tmp_path / "input.txt"
    path.write_bytes(b"abc\xffdef")
    monkeypatch.chdir(tmp_path)

    output = _read({"input": "input.txt", "offset": 2, "size": 3})

    assert "Source: local file" in output
    assert output.endswith("c\ufffdd")


def test_read_tool_import_and_text_read_do_not_import_pypdf():
    script = textwrap.dedent(
        """
        import sys
        import tempfile
        from pathlib import Path

        import jarv.read_tool as read_tool
        from jarv.artifacts import ArtifactStore
        from jarv.config import DEFAULT_CONFIG
        from jarv.retained_outputs import RetainedOutputStore

        print("after_import=" + str("pypdf" in sys.modules))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plain.txt"
            path.write_text("plain text", encoding="utf-8")
            read_tool.dispatch_read_tool(
                {"input": str(path)},
                visible_labels=set(),
                artifact_store=ArtifactStore(),
                retained_store=RetainedOutputStore(),
                config=DEFAULT_CONFIG,
            )
        print("after_text_read=" + str("pypdf" in sys.modules))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert result.stdout.splitlines() == [
        "after_import=False",
        "after_text_read=False",
    ]


def test_read_local_pdf_extracts_embedded_text_with_page_markers(tmp_path):
    path = tmp_path / "document.pdf"
    path.write_bytes(_minimal_pdf(["First PDF page", "Second PDF page"]))

    output = _read({"input": str(path), "size": 1000})

    assert "Source: local PDF" in output
    assert "PDF pages: 2" in output
    assert "PDF extraction: embedded text" in output
    assert "--- Page 1 of 2 ---" in output
    assert "First PDF page" in output
    assert "--- Page 2 of 2 ---" in output
    assert "Second PDF page" in output


def test_read_pdf_paging_uses_extracted_text_offsets(tmp_path):
    path = tmp_path / "document.pdf"
    path.write_bytes(_minimal_pdf(["First PDF page", "Second PDF page"]))
    full_output = _read({"input": str(path), "size": 1000})
    full_text = full_output.split("\n\n", 1)[1]

    paged_output = _read({"input": str(path), "offset": 10, "size": 20})
    paged_text = paged_output.split("\n\n", 1)[1]

    assert paged_text == full_text[10:30]
    assert "Offset: 10" in paged_output
    assert "Returned size: 20" in paged_output


def test_read_pdf_detects_pdf_magic_without_pdf_suffix(tmp_path):
    path = tmp_path / "download.bin"
    path.write_bytes(_minimal_pdf(["Magic PDF text"]))

    output = _read({"input": str(path), "size": 1000})

    assert "Source: local PDF" in output
    assert "Magic PDF text" in output


def test_read_pdf_reports_no_extractable_text(tmp_path):
    path = tmp_path / "scanned.pdf"
    path.write_bytes(_minimal_pdf([""]))

    output = _read({"input": str(path), "size": 1000})

    assert "no extractable text" in output
    assert "scanned/image-only PDFs are not supported" in output


def test_read_pdf_reports_corrupt_pdf(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-not-a-valid-file")

    output = _read({"input": str(path), "size": 1000})

    assert "could not parse PDF" in output or "could not read PDF pages" in output


def test_read_pdf_reports_encrypted_pdf(tmp_path, monkeypatch):
    class FakeEncryptedPdfReader:
        def __init__(self, *_args, **_kwargs):
            self.is_encrypted = True

        def decrypt(self, _password):
            return 0

    path = tmp_path / "encrypted.pdf"
    path.write_bytes(b"%PDF-fake")
    monkeypatch.setattr("jarv.pdf_extract._PDF_READER_CLASS", FakeEncryptedPdfReader)

    output = _read({"input": str(path), "size": 1000})

    assert "encrypted" in output
    assert "could not be read" in output


def test_read_web_pdf_content_type_extracts_text_and_marks_untrusted(monkeypatch):
    monkeypatch.setattr(
        "jarv.read_tool.fetch_web_bytes",
        lambda *args, **kwargs: FetchedWebBytes(
            requested_url="https://example.test/start",
            final_url="https://example.test/document.pdf",
            content_type="application/pdf",
            media_type="application/pdf",
            body=_minimal_pdf(["Remote PDF text"]),
        ),
    )

    output = _read({"input": "https://example.test/start", "size": 1000})

    assert "Source: web PDF" in output
    assert "UNTRUSTED WEB CONTENT" in output
    assert "Requested URL: https://example.test/start" in output
    assert "Final URL: https://example.test/document.pdf" in output
    assert "Content-Type: application/pdf" in output
    assert "PDF pages: 1" in output
    assert "Remote PDF text" in output


def test_read_web_pdf_magic_extracts_text(monkeypatch):
    monkeypatch.setattr(
        "jarv.read_tool.fetch_web_bytes",
        lambda *args, **kwargs: FetchedWebBytes(
            requested_url="https://example.test/download",
            final_url="https://example.test/download",
            content_type="application/octet-stream",
            media_type="application/octet-stream",
            body=_minimal_pdf(["Magic remote PDF text"]),
        ),
    )

    output = _read({"input": "https://example.test/download", "size": 1000})

    assert "Source: web PDF" in output
    assert "Content-Type: application/octet-stream" in output
    assert "Magic remote PDF text" in output


def test_read_batch_handles_concurrent_pdf_reads_in_order(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(_minimal_pdf(["First batch PDF"]))
    second.write_bytes(_minimal_pdf(["Second batch PDF"]))

    outputs = dispatch_read_batch(
        [{"input": str(first), "size": 1000}, {"input": str(second), "size": 1000}],
        visible_labels=set(),
        artifact_store=ArtifactStore(),
        retained_store=RetainedOutputStore(),
        config=DEFAULT_CONFIG,
    )

    assert "First batch PDF" in outputs[0]
    assert "Second batch PDF" in outputs[1]


def test_read_local_image_returns_structured_image_output(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    path.write_bytes(PNG_BYTES)
    monkeypatch.setattr(
        "jarv.read_tool.get_image_output_capability",
        lambda _config: ModelImageCapability(True, "responses"),
    )

    output = _read({"input": str(path), "offset": 5, "size": 10})

    assert isinstance(output, list)
    assert output[0]["type"] == "input_text"
    assert "Source: local image" in output[0]["text"]
    assert "Offset: not applicable for image reads" in output[0]["text"]
    assert "Image media type: image/png" in output[0]["text"]
    assert output[1]["type"] == "input_image"
    assert output[1]["image_url"].startswith("data:image/png;base64,")


def test_read_image_without_model_capability_returns_text_fallback(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    path.write_bytes(PNG_BYTES)
    monkeypatch.setattr(
        "jarv.read_tool.get_image_output_capability",
        lambda _config: ModelImageCapability(False, reason="text-only model"),
    )

    output = _read({"input": str(path)})

    assert isinstance(output, str)
    assert "does not have image capability" in output
    assert "text-only model" in output
    assert "base64" not in output


def test_read_rejects_unsupported_image_format(tmp_path):
    path = tmp_path / "image.bmp"
    path.write_bytes(b"BMnot-supported")

    output = _read({"input": str(path)})

    assert "unsupported image media type" in output
    assert "image/bmp" in output


def test_read_rejects_oversized_image(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    path.write_bytes(PNG_BYTES + b"x" * 10)
    monkeypatch.setattr("jarv.read_tool.MAX_IMAGE_READ_BYTES", len(PNG_BYTES) - 1)

    output = _read({"input": str(path)})

    assert "exceeding" in output
    assert "byte limit" in output


def test_every_web_page_is_marked_untrusted(monkeypatch):
    monkeypatch.setattr(
        "jarv.read_tool.fetch_web_bytes",
        lambda *args, **kwargs: FetchedWebBytes(
            requested_url="https://example.test/start",
            final_url="https://example.test/final",
            content_type="text/html",
            media_type="text/html",
            body=b"<title>Example</title><main>abcdefghij</main>",
        ),
    )

    output = _read(
        {"input": "https://example.test/start", "offset": 4, "size": 3}
    )

    assert "UNTRUSTED WEB CONTENT" in output
    assert "Requested URL: https://example.test/start" in output
    assert "Final URL: https://example.test/final" in output
    assert output.endswith("efg")


def test_read_direct_image_url_returns_untrusted_structured_output(monkeypatch):
    def handler(_request):
        return httpx.Response(
            200,
            headers={"content-type": "image/webp"},
            content=WEBP_BYTES,
        )

    monkeypatch.setattr(
        "jarv.web._create_client",
        lambda _timeout: httpx.Client(
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
        ),
    )
    monkeypatch.setattr(
        "jarv.read_tool.get_image_output_capability",
        lambda _config: ModelImageCapability(True, "openai_chat"),
    )

    output = _read({"input": "https://example.test/image"})

    assert isinstance(output, list)
    assert "Source: web image" in output[0]["text"]
    assert "UNTRUSTED WEB IMAGE" in output[0]["text"]
    assert "Content-Type: image/webp" in output[0]["text"]
    assert output[1]["image_url"].startswith("data:image/webp;base64,")


def test_truncated_command_output_can_be_reconstructed_from_retained_id():
    retained = RetainedOutputStore()
    content = "HEAD-" + ("middle" * 10) + "-TAIL"

    rendered, output_id = retain_command_output(
        content,
        5,
        5,
        retained,
        default_read_size=7,
    )

    assert output_id is not None
    assert f"id={output_id}" in rendered
    assert "omitted offset=5 size=60" in rendered
    assert f'read(input="{output_id}", offset=5, size=7)' in rendered
    middle = _read(
        {"input": output_id, "offset": 5, "size": 60},
        retained=retained,
    ).split("\n\n", 1)[1]
    assert content[:5] + middle + content[-5:] == content


def test_retained_output_store_round_trips(tmp_path):
    path = tmp_path / "reads.json"
    store = RetainedOutputStore()
    output_id = store.put("persisted")

    save_retained_output_store(store, path)
    loaded = load_retained_output_store(path)

    assert loaded.get(output_id).content == "persisted"


def test_read_batch_runs_concurrently_and_preserves_order(monkeypatch):
    def fake_dispatch(args, **kwargs):
        time.sleep(0.1)
        return args["input"]

    monkeypatch.setattr("jarv.read_tool.dispatch_read_tool", fake_dispatch)
    started = time.perf_counter()
    outputs = dispatch_read_batch(
        [{"input": "first"}, {"input": "second"}],
        visible_labels=set(),
        artifact_store=ArtifactStore(),
        retained_store=RetainedOutputStore(),
        config=DEFAULT_CONFIG,
    )
    elapsed = time.perf_counter() - started

    assert outputs == ["first", "second"]
    assert elapsed < 0.18


def test_read_batch_propagates_cancellation():
    token = CancellationToken()
    token.cancel()

    with pytest.raises(TurnCancelled):
        dispatch_read_batch(
            [{"input": "missing"}, {"input": "missing"}],
            visible_labels=set(),
            artifact_store=ArtifactStore(),
            retained_store=RetainedOutputStore(),
            config=DEFAULT_CONFIG,
            cancellation_token=token,
        )
