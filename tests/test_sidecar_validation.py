import json

import pytest

from jarv.artifacts import load_artifact_store, save_artifact_store
from jarv.retained_outputs import (
    RetainedOutputStore, load_retained_output_store, save_retained_output_store,
)
from jarv.storage import StorageError


@pytest.mark.parametrize("record", [None, "report", [], {"longform": None},
                                    {"tldr": 42}, {"owner_label": ["worker"]}])
def test_malformed_artifact_records_are_explicit_and_preserved(tmp_path, record):
    path = tmp_path / "artifacts.json"
    original = json.dumps({"valid": {"longform": "saved report"}, "broken": record})
    path.write_text(original, encoding="utf-8")

    with pytest.raises(StorageError, match="Invalid artifact record"):
        load_artifact_store(path)

    assert path.read_text(encoding="utf-8") == original


def test_legacy_artifact_optional_fields_remain_supported(tmp_path):
    path = tmp_path / "artifacts.json"
    path.write_text('{"report":{"longform":"saved report"}}', encoding="utf-8")

    store = load_artifact_store(path)
    artifact = store.get("report")
    assert artifact.longform == "saved report"
    assert artifact.tldr == ""
    assert artifact.owner_label == "report"
    save_artifact_store(store, path)
    assert load_artifact_store(path).get("report") == artifact


@pytest.mark.parametrize("output_id, record", [
    ("cmd_broken", None), ("cmd_broken", 42), ("cmd_broken", []),
    ("cmd_broken", {}), ("cmd_broken", {"content": None}),
    ("cmd_broken", {"content": ["output"]}), ("invalid-id", "output"),
])
def test_malformed_retained_records_are_explicit_and_preserved(tmp_path, output_id, record):
    path = tmp_path / "reads.json"
    original = json.dumps({"cmd_valid": {"content": "saved output"}, output_id: record})
    path.write_text(original, encoding="utf-8")

    with pytest.raises(StorageError, match="Invalid retained output record"):
        load_retained_output_store(path)

    assert path.read_text(encoding="utf-8") == original


def test_legacy_retained_strings_remain_supported(tmp_path):
    path = tmp_path / "reads.json"
    path.write_text('{"cmd_old":"saved output"}', encoding="utf-8")

    store = load_retained_output_store(path)
    assert store.get("cmd_old").content == "saved output"
    save_retained_output_store(store, path)
    assert load_retained_output_store(path).get("cmd_old").content == "saved output"


def test_retained_size_limit_applies_after_transaction_recovery(tmp_path, monkeypatch):
    from jarv import retained_outputs

    path = tmp_path / "reads.json"
    path.write_text("{}", encoding="utf-8")
    recovered = {"cmd_large": {"content": "x" * 200}}
    journal = tmp_path / ".jarv-transaction.json"
    journal.write_text(json.dumps({str(path.resolve()): recovered}), encoding="utf-8")
    monkeypatch.setattr(retained_outputs, "MAX_RETAINED_FILE_BYTES", 100)

    with pytest.raises(StorageError, match="byte limit"):
        load_retained_output_store(path)

    assert json.loads(path.read_text(encoding="utf-8")) == recovered
    assert not journal.exists()


def test_retained_truncation_marker_fits_inside_output_limit(monkeypatch):
    from jarv import retained_outputs

    monkeypatch.setattr(retained_outputs, "MAX_RETAINED_OUTPUT_CHARS", 200)
    store = RetainedOutputStore()
    output_id = store.put("start" + "x" * 400 + "end")
    content = store.get(output_id).content

    assert len(content) == 200
    assert content.startswith("start")
    assert content.endswith("end")
    assert "discarded and are unavailable for read" in content


@pytest.mark.parametrize("sidecar_name, record", [
    ("artifacts.json", {"broken": None}), ("reads.json", {"cmd_broken": None}),
])
def test_invalid_sidecar_stops_turn_without_overwriting_it(tmp_path, sidecar_name, record):
    from datetime import datetime, timezone
    from unittest.mock import patch

    from jarv.agent import run_agent
    from jarv.config import DEFAULT_CONFIG
    from jarv.history import SessionContext, load_history

    history_path = tmp_path / "history.json"
    transcript = [{"role": "user", "id": "first", "content": "Existing prompt"}]
    history_path.write_text(json.dumps(transcript), encoding="utf-8")
    path = tmp_path / sidecar_name
    original = json.dumps(record)
    path.write_text(original, encoding="utf-8")
    context = SessionContext("test", "test", history_path, datetime.now(timezone.utc))

    with (
        patch("jarv.agent.prepare_session_context", return_value=context),
        patch("jarv.agent.stream_response") as provider,
    ):
        result = run_agent("New prompt", DEFAULT_CONFIG, client=object())

    assert result.error is not None
    assert "Invalid " in result.error
    provider.assert_not_called()
    assert path.read_text(encoding="utf-8") == original
    assert load_history(history_path) == transcript
