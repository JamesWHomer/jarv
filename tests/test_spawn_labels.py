import json
import threading
from unittest.mock import Mock

import pytest

from jarv.artifacts import ArtifactStore, load_artifact_store, save_artifact_store
from jarv.cancellation import CancellationToken, TurnCancelled
from jarv.config import DEFAULT_CONFIG
from jarv.orchestrator import AgentNode, dispatch_tool, spawn_batch, spawn_tool_output


@pytest.fixture
def parent():
    return AgentNode("root", 0, None, "root task", sterile=False)


def test_successive_batches_cannot_overwrite_an_artifact(parent, monkeypatch):
    store = ArtifactStore()
    worker = Mock(return_value=("original report", "original summary"))
    monkeypatch.setattr("jarv.orchestrator.run_subagent_loop", worker)
    children = [{"label": "report", "task": "work"}]

    first = spawn_batch(parent, children, store, None, DEFAULT_CONFIG)
    second = spawn_tool_output(parent, children, store, None, DEFAULT_CONFIG)

    assert first == [{"label": "report", "status": "done", "tldr": "original summary"}]
    assert second == (
        "[tool argument error: child label 'report' is already used in this session; "
        "choose a new label]"
    )
    worker.assert_called_once()
    assert store.get("report").longform == "original report"
    assert store.get("report").tldr == "original summary"
    assert "original report" in dispatch_tool(
        "read", {"input": "report"}, parent, store, None, DEFAULT_CONFIG,
    )


def test_loaded_artifacts_block_label_reuse(parent, monkeypatch, tmp_path):
    monkeypatch.setattr("jarv.config.CONFIG_DIR", tmp_path)
    path = tmp_path / "artifacts.json"
    store = ArtifactStore()
    store.put("report", "saved report", "saved summary", "report")
    save_artifact_store(store, path)
    restored = load_artifact_store(path)
    worker = Mock(return_value=("replacement", "replacement"))
    monkeypatch.setattr("jarv.orchestrator.run_subagent_loop", worker)

    with pytest.raises(ValueError, match="already used in this session"):
        spawn_batch(
            parent, [{"label": "report", "task": "work"}],
            restored, None, DEFAULT_CONFIG,
        )

    worker.assert_not_called()
    assert restored.get("report") == store.get("report")


def test_conflicting_batch_does_not_start_or_reserve_other_children(parent, monkeypatch):
    store = ArtifactStore()
    store.put("report", "original", "summary", "report")
    worker = Mock(return_value=("new report", "new summary"))
    observer = Mock()
    monkeypatch.setattr("jarv.orchestrator.run_subagent_loop", worker)
    fresh = {"label": "fresh", "task": "work"}

    with pytest.raises(ValueError, match="already used in this session"):
        spawn_batch(
            parent, [fresh, {"label": "report", "task": "work"}],
            store, None, DEFAULT_CONFIG, observer=observer,
        )

    worker.assert_not_called()
    observer.on_spawn_start.assert_not_called()
    results = spawn_batch(parent, [fresh], store, None, DEFAULT_CONFIG)
    assert results[0]["status"] == "done"
    assert store.get("report").longform == "original"


@pytest.mark.parametrize("invalid", [
    {"label": "fresh", "task": "duplicate"},
    {"label": "invalid", "task": ""},
])
def test_invalid_batch_does_not_consume_labels(parent, monkeypatch, invalid):
    store = ArtifactStore()
    worker = Mock(return_value=("report", "summary"))
    monkeypatch.setattr("jarv.orchestrator.run_subagent_loop", worker)
    fresh = {"label": "fresh", "task": "work"}

    with pytest.raises(ValueError):
        spawn_batch(parent, [fresh, invalid], store, None, DEFAULT_CONFIG)

    worker.assert_not_called()
    assert spawn_batch(parent, [fresh], store, None, DEFAULT_CONFIG)[0]["status"] == "done"


def test_concurrent_nested_batches_reserve_labels_before_work_starts(parent, monkeypatch):
    store = ArtifactStore()
    parents_ready = threading.Barrier(2)
    conflict_reported = threading.Event()
    nested_outputs = {}
    report_readers = set()

    def worker(node, *_args, **_kwargs):
        if node.depth == 1:
            parents_ready.wait(timeout=5)
            output = spawn_tool_output(
                node, [{"label": "report", "task": "work"}],
                store, None, DEFAULT_CONFIG,
            )
            nested_outputs[node.label] = output
            if output.startswith("[tool argument error:"):
                conflict_reported.set()
            if "report" in node.visible_labels:
                report_readers.add(node.label)
            return f"parent {node.label}", "parent done"

        # The other batch must reject this label while its artifact is still
        # absent. Waiting for that rejection makes the race deterministic.
        assert conflict_reported.wait(timeout=5)
        assert not store.exists("report")
        assert "report" not in store.all_labels()
        return f"report from {node.parent_label}", node.parent_label

    monkeypatch.setattr("jarv.orchestrator.run_subagent_loop", worker)
    results = spawn_batch(
        parent,
        [
            {"label": "left", "task": "work", "sterile": False},
            {"label": "right", "task": "work", "sterile": False},
        ],
        store, None, DEFAULT_CONFIG,
    )

    assert all(result["status"] == "done" for result in results)
    errors = [
        output for output in nested_outputs.values()
        if output.startswith("[tool argument error:")
    ]
    assert len(errors) == 1
    assert "child label 'report' is already used in this session" in errors[0]
    winner = store.get("report").tldr
    assert store.get("report").longform == f"report from {winner}"
    assert json.loads(nested_outputs[winner])[0]["status"] == "done"
    assert report_readers == {winner}
    assert store.all_labels() == {"left", "right", "report"}


@pytest.mark.parametrize("label", ["branch", "sibling"])
def test_nested_batch_cannot_claim_an_active_ancestor_or_queued_sibling(
    parent, monkeypatch, label,
):
    store = ArtifactStore()
    nested_outputs = []

    def worker(node, *_args, **_kwargs):
        if node.label == "branch" and node.depth == 1:
            nested_outputs.append(spawn_tool_output(
                node, [{"label": label, "task": "nested work"}],
                store, None, DEFAULT_CONFIG,
            ))
        return node.task, "done"

    monkeypatch.setattr("jarv.orchestrator.run_subagent_loop", worker)
    results = spawn_batch(
        parent,
        [
            {"label": "branch", "task": "parent work", "sterile": False},
            {"label": "sibling", "task": "sibling work"},
        ],
        store, None, {**DEFAULT_CONFIG, "subagent_thread_pool_max_workers": 1},
    )

    assert all(result["status"] == "done" for result in results)
    assert "already used in this session" in nested_outputs[0]
    assert store.get("branch").longform == "parent work"
    assert store.get("sibling").longform == "sibling work"


@pytest.mark.parametrize("failure", [None, RuntimeError("failed"), TurnCancelled()])
def test_failed_or_cancelled_workers_keep_their_label_reserved(parent, monkeypatch, failure):
    store = ArtifactStore()
    worker = Mock(return_value=(None, "failed"), side_effect=failure)
    monkeypatch.setattr("jarv.orchestrator.run_subagent_loop", worker)
    children = [{"label": "report", "task": "work"}]

    if isinstance(failure, TurnCancelled):
        with pytest.raises(TurnCancelled):
            spawn_batch(
                parent, children, store, None, DEFAULT_CONFIG,
                cancellation_token=CancellationToken(),
            )
    else:
        assert spawn_batch(parent, children, store, None, DEFAULT_CONFIG)[0]["status"] == "failed"

    with pytest.raises(ValueError, match="already used in this session"):
        spawn_batch(parent, children, store, None, DEFAULT_CONFIG)
    worker.assert_called_once()
    assert store.all_labels() == set()


def test_artifact_store_rejects_overwrites():
    store = ArtifactStore()
    store.put("report", "original", "summary", "owner")
    original = store.get("report")

    with pytest.raises(ValueError, match="already exists"):
        store.put("report", "replacement", "new summary", "different owner")

    assert store.get("report") == original


@pytest.mark.parametrize("saved,reserved", [("alpha", "zeta"), ("zeta", "alpha")])
def test_reservations_report_first_conflict_across_saved_and_active_labels(saved, reserved):
    store = ArtifactStore()
    store.put(saved, "saved report", "summary", "owner")
    store.reserve_labels({reserved})

    with pytest.raises(ValueError) as error:
        store.reserve_labels({saved, reserved, "fresh"})

    assert str(error.value) == (
        "child label 'alpha' is already used in this session; choose a new label"
    )
    # Failed batches must leave every new label available, even when their
    # conflicts span both persisted artifacts and still-running children.
    store.reserve_labels({"fresh"})
    store.reserve_labels(set())
    assert store.get(saved).longform == "saved report"
