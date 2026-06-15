import time
from pathlib import Path

import pytest

from jarv.artifacts import ArtifactStore
from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.config import DEFAULT_CONFIG
from jarv.read_tool import (
    READ_TOOL,
    dispatch_read_batch,
    dispatch_read_tool,
    retain_command_output,
)
from jarv.retained_outputs import (
    RetainedOutputStore,
    load_retained_output_store,
    save_retained_output_store,
)
from jarv.web import FetchedWebContent


def _read(args, *, artifacts=None, retained=None, visible=None, config=None):
    return dispatch_read_tool(
        args,
        visible_labels=visible or set(),
        artifact_store=artifacts or ArtifactStore(),
        retained_store=retained or RetainedOutputStore(),
        config=config or DEFAULT_CONFIG,
    )


def test_read_schema_exposes_input_offset_and_size():
    parameters = READ_TOOL["parameters"]
    assert parameters["required"] == ["input"]
    assert parameters["properties"]["offset"]["minimum"] == 0
    assert parameters["properties"]["size"]["minimum"] == 1


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
    ],
)
def test_read_validates_paging_arguments(args, message):
    assert message in _read(args)


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


def test_every_web_page_is_marked_untrusted(monkeypatch):
    monkeypatch.setattr(
        "jarv.read_tool.fetch_web_content",
        lambda *args, **kwargs: FetchedWebContent(
            requested_url="https://example.test/start",
            final_url="https://example.test/final",
            media_type="text/html",
            title="Example",
            text="abcdefghij",
        ),
    )

    output = _read(
        {"input": "https://example.test/start", "offset": 4, "size": 3}
    )

    assert "UNTRUSTED WEB CONTENT" in output
    assert "Requested URL: https://example.test/start" in output
    assert "Final URL: https://example.test/final" in output
    assert output.endswith("efg")


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
