import json
from pathlib import Path
import subprocess
import sys

import pytest

from jarv import storage
from jarv.history import load_history, save_history, load_sessions, save_sessions


def test_failure_before_commit_preserves_original(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    save_history(["old"], path)
    original = storage.os.replace

    def fail(source, destination):
        if Path(destination).name == ".jarv-transaction.json":
            raise OSError("disk full")
        original(source, destination)

    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(storage.StorageError, match="disk full"):
        save_history(["new"], path)
    assert load_history(path) == ["old"]
    assert not list(tmp_path.glob(".jarv-*"))


def test_killed_writer_recovers_whole_transaction(tmp_path):
    history = tmp_path / "history.json"
    reads = tmp_path / "reads.json"
    save_history(["old"], history)
    storage.write_json(reads, {"old": True})
    script = '''
import os, sys
from pathlib import Path
from jarv import storage
root = Path(sys.argv[1])
original = storage._atomic
def crash(path, data):
    original(path, data)
    if path.name == "history.json":
        os._exit(17)
storage._atomic = crash
with storage.transaction(root / "history.json"):
    storage.write_json(root / "history.json", ["new"])
    storage.write_json(root / "reads.json", {"new": True})
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], timeout=20)
    assert result.returncode == 17
    assert load_history(history) == ["new"]
    assert storage.read_json(reads, {}, dict) == {"new": True}
    assert not (tmp_path / ".jarv-transaction.json").exists()


def test_two_processes_merge_independent_metadata(tmp_path):
    script = '''
import sys
from pathlib import Path
from jarv import history
history.SESSIONS_FILE = Path(sys.argv[1])
data = history.load_sessions()
print("ready", flush=True)
sys.stdin.readline()
data["sessions"][sys.argv[2]] = {"label": sys.argv[2]}
history.save_sessions(data)
'''
    path = tmp_path / "sessions.json"
    workers = [subprocess.Popen(
        [sys.executable, "-c", script, str(path), name],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    ) for name in ("one", "two")]
    try:
        for worker in workers:
            assert worker.stdout.readline().strip() == "ready"
        for worker in workers:
            worker.stdin.write("go\n")
            worker.stdin.flush()
        for worker in workers:
            _, error = worker.communicate(timeout=20)
            assert worker.returncode == 0, error
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
            worker.wait()
    assert set(json.loads(path.read_text())["sessions"]) == {"one", "two"}


def test_stale_history_cannot_overwrite_or_partially_save(tmp_path):
    path = tmp_path / "history.json"
    sidecar = tmp_path / "reads.json"
    save_history(["original"], path)
    stale = load_history(path)
    fresh = load_history(path)
    fresh.append("other terminal")
    save_history(fresh, path)
    stale.append("my terminal")
    with pytest.raises(storage.StorageConflict):
        with storage.transaction(path):
            storage.write_json(sidecar, {"uncommitted": True})
            save_history(stale, path)
    assert load_history(path) == fresh
    assert not sidecar.exists()


def test_metadata_conflict_preserves_winner(tmp_path, monkeypatch):
    from jarv import history
    monkeypatch.setattr(history, "SESSIONS_FILE", tmp_path / "sessions.json")
    save_sessions({"terminals": {}, "sessions": {"s": {"label": "old"}}})
    a, b = load_sessions(), load_sessions()
    a["sessions"]["s"]["label"] = "first"
    b["sessions"]["s"]["label"] = "second"
    save_sessions(a)
    with pytest.raises(storage.StorageConflict):
        save_sessions(b)
    assert load_sessions()["sessions"]["s"]["label"] == "first"


def test_repeated_metadata_save_retains_concurrent_additions(tmp_path, monkeypatch):
    from jarv import history
    monkeypatch.setattr(history, "SESSIONS_FILE", tmp_path / "sessions.json")
    a, b = load_sessions(), load_sessions()
    a["sessions"]["a"] = {"label": "a"}
    b["sessions"]["b"] = {"label": "b"}
    save_sessions(a)
    save_sessions(b)
    b["sessions"]["b"]["label"] = "renamed"
    save_sessions(b)
    assert set(load_sessions()["sessions"]) == {"a", "b"}


def test_sidecar_conflict_does_not_commit_history(tmp_path):
    from types import SimpleNamespace
    from jarv.agent import SessionPersistence
    from jarv.artifacts import load_artifact_store, save_artifact_store
    path = tmp_path / "history.json"
    sidecar = tmp_path / "artifacts.json"
    save_history(["old"], path)
    persistence = SessionPersistence(incognito=False)
    persistence.session_context = SimpleNamespace(history_file=path)
    persistence.history = load_history(path)
    persistence.history.append("new")
    persistence.artifact_file = sidecar
    persistence.artifact_store = load_artifact_store(sidecar)
    other = load_artifact_store(sidecar)
    other.put("other", "content", "summary", "root")
    save_artifact_store(other, sidecar)
    with pytest.raises(storage.StorageConflict):
        persistence.save_turn()
    assert load_history(path) == ["old"]
    assert load_artifact_store(sidecar).get("other") is not None


@pytest.mark.parametrize("content", ["{broken", "null", "{}"])
def test_corruption_is_explicit_and_preserved(tmp_path, content):
    path = tmp_path / "history.json"
    path.write_text(content)
    with pytest.raises(storage.StorageError):
        load_history(path)
    assert path.read_text() == content


def test_transaction_reads_and_updates_staged_history(tmp_path):
    path = tmp_path / "history.json"
    save_history(["original"], path)
    with storage.transaction(path):
        history = load_history(path)
        history.append("first")
        save_history(history, path)
        history = load_history(path)
        history.append("second")
        save_history(history, path)
    assert load_history(path) == ["original", "first", "second"]


def test_transaction_merges_independent_staged_updates(tmp_path):
    path = tmp_path / "metadata.json"
    storage.write_json(path, {"original": True})
    first = storage.read_json(path, {}, dict)
    second = storage.read_json(path, {}, dict)
    first["first"] = True
    second["second"] = True
    with storage.transaction(path):
        storage.write_json(path, first, merge=True, snapshot=first)
        storage.write_json(path, second, merge=True, snapshot=second)
    assert storage.read_json(path, {}, dict) == {
        "original": True, "first": True, "second": True,
    }


def test_transaction_repeatedly_saves_same_snapshot(tmp_path):
    path = tmp_path / "history.json"
    save_history(["original"], path)
    history = load_history(path)
    with storage.transaction(path):
        history.append("first")
        save_history(history, path)
        history.append("second")
        save_history(history, path)
    history.append("third")
    save_history(history, path)
    assert load_history(path) == ["original", "first", "second", "third"]


def test_aborted_transaction_does_not_retain_staged_baseline(tmp_path):
    path = tmp_path / "history.json"
    save_history(["original"], path)
    with pytest.raises(RuntimeError, match="abort"):
        with storage.transaction(path):
            save_history(["staged"], path)
            assert load_history(path) == ["staged"]
            raise RuntimeError("abort")
    save_history(["replacement"], path)
    assert load_history(path) == ["replacement"]


def test_transaction_delete_then_recreate(tmp_path):
    path = tmp_path / "history.json"
    save_history(["original"], path)
    with storage.transaction(path):
        storage.delete_json(path)
        history = load_history(path)
        assert history == []
        history.append("replacement")
        save_history(history, path)
    assert load_history(path) == ["replacement"]


def test_discarded_histories_leave_only_compact_observations(tmp_path):
    import gc
    import pickle
    import weakref

    paths = []
    snapshots = []
    for index in range(20):
        path = tmp_path / f"history-{index}.json"
        path.write_text(json.dumps([{"content": str(index) + "x" * 50_000}]), encoding="utf-8")
        loaded = load_history(path)
        paths.append(str(path.resolve()))
        snapshots.append(weakref.ref(loaded))
    del loaded
    gc.collect()

    assert all(snapshot() is None for snapshot in snapshots)
    observations = {path: storage._state().seen[path] for path in paths}
    # Reading and discarding a megabyte of transcripts must not leave a
    # megabyte in the storage thread's fallback conflict-detection state.
    assert len(pickle.dumps(observations)) < 20_000


def test_detached_history_still_detects_concurrent_changes(tmp_path):
    path = tmp_path / "history.json"
    path.write_text('["original"]', encoding="utf-8")
    detached = list(load_history(path))
    detached.append("my change")
    path.write_text('["original","other process"]', encoding="utf-8")

    with pytest.raises(storage.StorageConflict):
        save_history(detached, path)
    assert json.loads(path.read_text(encoding="utf-8")) == ["original", "other process"]


def test_detached_metadata_merges_independent_nested_changes(tmp_path):
    path = tmp_path / "metadata.json"
    initial = {"settings": {"model": "old", "colour": True}, "remove": "old"}
    path.write_text(json.dumps(initial), encoding="utf-8")
    detached = dict(storage.read_json(path, {}, dict))
    detached["settings"]["colour"] = False
    del detached["remove"]
    path.write_text(json.dumps({"settings": {"model": "new", "colour": True},
                                "remove": "old", "concurrent": "kept"}), encoding="utf-8")

    storage.write_json(path, detached, merge=True)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "settings": {"model": "new", "colour": False}, "concurrent": "kept",
    }


def test_detached_metadata_rejects_conflicting_nested_changes(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text('{"settings":{"model":"old"}}', encoding="utf-8")
    detached = dict(storage.read_json(path, {}, dict))
    detached["settings"]["model"] = "mine"
    path.write_text('{"settings":{"model":"theirs"}}', encoding="utf-8")

    with pytest.raises(storage.StorageConflict):
        storage.write_json(path, detached, merge=True)
    assert json.loads(path.read_text(encoding="utf-8")) == {"settings": {"model": "theirs"}}
