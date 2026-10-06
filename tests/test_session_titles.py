import json
from pathlib import Path

import pytest

from jarv.session_titles import SessionTitleCache


def history_file(tmp_path, items):
    path = tmp_path / "history.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def test_title_reads_only_the_prefix_and_survives_a_new_cache_instance(tmp_path, monkeypatch):
    path = history_file(tmp_path, [{"role": "user", "content": "  Fix\n display sizing  "},
                                   {"role": "assistant", "content": "x" * 1_000_000}])
    cache_path = tmp_path / "titles.json"
    cache = SessionTitleCache(cache_path, [str(path)])
    reads = []
    original_open = Path.open

    class TrackedReader:
        def __enter__(self):
            self.stream = original_open(path, encoding="utf-8")
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size):
            reads.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(Path, "open", lambda self, *args, **kwargs:
                        TrackedReader() if self == path else original_open(self, *args, **kwargs))
    assert cache.read(str(path)) == "Fix display sizing"
    assert sum(reads) == 8192
    cache.save()
    reopened = SessionTitleCache(cache_path, [str(path)])
    reads.clear()
    assert reopened.read(str(path)) == "Fix display sizing"
    assert reads == []


@pytest.mark.parametrize("items, expected", [
    ([], ""),
    ([{"role": "assistant", "content": "No prompt"}], ""),
    ([{"role": "user", "content": " "}, {"role": "user", "content": "Second prompt"}], "Second prompt"),
    ([{"role": "system", "content": "x" * 20000},
      {"role": "user", "content": [{"type": "input_text", "text": "日本語の会話"}]}], "日本語の会話"),
    ([{"role": "user", "content": "x" * 300000}], None),
])
def test_title_prefix_handles_history_shapes_and_bounds_large_messages(tmp_path, items, expected):
    path = history_file(tmp_path, items)
    cache = SessionTitleCache(tmp_path / "titles.json", [str(path)])
    assert cache.read(str(path)) == expected
    assert (cache.get(str(path)) is not None) == (expected is not None)


def test_changed_or_moved_history_cannot_reuse_an_old_title(tmp_path):
    path = history_file(tmp_path, [{"role": "user", "content": "Old prompt"}])
    cache_path = tmp_path / "titles.json"
    cache = SessionTitleCache(cache_path, [str(path)])
    assert cache.read(str(path)) == "Old prompt"
    cache.save()
    path.write_text('[{"role":"user","content":"Replacement prompt"}]', encoding="utf-8")
    reopened = SessionTitleCache(cache_path, [str(path)])
    assert reopened.get(str(path)) is None
    assert reopened.read(str(path)) == "Replacement prompt"
    archived = tmp_path / "archived.json"
    path.rename(archived)
    assert reopened.get(str(path)) is None
    assert reopened.read(str(archived)) == "Replacement prompt"


@pytest.mark.parametrize("bad_cache", ["{bad json", "[]", '{"version":1,"titles":[]}'])
def test_invalid_cache_and_failed_cache_write_do_not_break_titles(tmp_path, monkeypatch, bad_cache):
    path = history_file(tmp_path, [{"role": "user", "content": "Readable prompt"}])
    original = path.read_bytes()
    cache_path = tmp_path / "titles.json"
    cache_path.write_text(bad_cache, encoding="utf-8")
    cache = SessionTitleCache(cache_path, [str(path)])
    assert cache.read(str(path)) == "Readable prompt"

    def denied(*args):
        raise PermissionError("Cache is read-only")

    monkeypatch.setattr("jarv.session_titles.os.replace", denied)
    cache.save()
    assert cache.get(str(path)) == "Readable prompt"
    assert path.read_bytes() == original
    assert cache_path.read_text(encoding="utf-8") == bad_cache
    assert list(tmp_path.glob(".jarv-titles-*")) == []


def test_history_changed_during_prefix_read_is_not_cached(tmp_path, monkeypatch):
    path = history_file(tmp_path, [{"role": "user", "content": "Original prompt"}])
    cache = SessionTitleCache(tmp_path / "titles.json", [str(path)])
    stamps = iter([(str(path), 1, 20), (str(path), 2, 40)])
    monkeypatch.setattr(cache, "_stamp", lambda path: next(stamps))
    assert cache.read(str(path)) is None
    assert not cache.entries


def _metadata(tmp_path, paths):
    from jarv.storage import write_json

    path = tmp_path / "sessions.json"
    write_json(path, {"sessions": {str(index): {"history_file": str(value)}
                                   for index, value in enumerate(paths)}, "terminals": {}})
    return path


def test_deleted_session_title_is_removed_from_disk_and_stale_writer(tmp_path):
    from jarv.session_store import delete_session_files

    path = history_file(tmp_path, [{"role": "user", "content": "Private first prompt"}])
    metadata = _metadata(tmp_path, [path])
    cache_path = tmp_path / "session-titles.json"
    stale = SessionTitleCache(cache_path, [str(path)], metadata_path=metadata)
    assert stale.read(str(path)) == "Private first prompt"
    cache = SessionTitleCache(cache_path, [str(path)], metadata_path=metadata)
    cache.read(str(path))
    cache.save()
    _metadata(tmp_path, [])
    delete_session_files(path)
    assert not path.exists()
    assert "Private first prompt" not in cache_path.read_text(encoding="utf-8")
    stale.save()
    assert json.loads(cache_path.read_text(encoding="utf-8"))["titles"] == {}


def test_pruning_cache_without_new_titles_is_persisted(tmp_path):
    path = history_file(tmp_path, [{"role": "user", "content": "Removed prompt"}])
    cache_path = tmp_path / "session-titles.json"
    cache = SessionTitleCache(cache_path, [str(path)])
    cache.read(str(path))
    cache.save()
    reopened = SessionTitleCache(cache_path, [])
    assert reopened.dirty
    reopened.save()
    assert json.loads(cache_path.read_text(encoding="utf-8"))["titles"] == {}


def test_concurrent_cache_writers_keep_other_live_titles(tmp_path):
    first = history_file(tmp_path, [{"role": "user", "content": "First prompt"}])
    second = tmp_path / "history-second.json"
    second.write_text('[{"role":"user","content":"Second prompt"}]', encoding="utf-8")
    metadata = _metadata(tmp_path, [first, second])
    cache_path = tmp_path / "session-titles.json"
    # Their stale constructor inventories each omit the other live session.
    a = SessionTitleCache(cache_path, [str(first)], metadata_path=metadata)
    b = SessionTitleCache(cache_path, [str(second)], metadata_path=metadata)
    a.read(str(first))
    b.read(str(second))
    a.save()
    b.save()
    titles = json.loads(cache_path.read_text(encoding="utf-8"))["titles"]
    assert {entry["snippet"] for entry in titles.values()} == {"First prompt", "Second prompt"}


def test_cache_update_during_save_is_kept_pending(tmp_path, monkeypatch):
    import threading
    from jarv import session_titles

    first = history_file(tmp_path, [{"role": "user", "content": "First prompt"}])
    second = tmp_path / "history-second.json"
    second.write_text('[{"role":"user","content":"Second prompt"}]', encoding="utf-8")
    metadata = _metadata(tmp_path, [first, second])
    cache_path = tmp_path / "session-titles.json"
    cache = SessionTitleCache(cache_path, [str(first), str(second)], metadata_path=metadata)
    cache.read(str(first))
    entered, resume = threading.Event(), threading.Event()
    write = session_titles.write_json

    def delayed_write(*args, **kwargs):
        entered.set()
        assert resume.wait(5)
        return write(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(session_titles, "write_json", delayed_write)
        worker = threading.Thread(target=cache.save)
        worker.start()
        try:
            assert entered.wait(5)
            assert cache.read(str(second)) == "Second prompt"
        finally:
            resume.set()
            worker.join(5)
        assert not worker.is_alive()
    assert cache.dirty
    cache.save()
    assert not cache.dirty
    assert len(json.loads(cache_path.read_text(encoding="utf-8"))["titles"]) == 2


def test_other_process_cannot_restore_title_after_deletion(tmp_path):
    import subprocess
    import sys
    from jarv.session_store import delete_session_files

    path = history_file(tmp_path, [{"role": "user", "content": "Private first prompt"}])
    metadata = _metadata(tmp_path, [path])
    cache_path = tmp_path / "session-titles.json"
    script = '''
import sys
from pathlib import Path
from jarv.session_titles import SessionTitleCache
path, cache_path, metadata = sys.argv[1:]
cache = SessionTitleCache(Path(cache_path), [path], metadata_path=Path(metadata))
cache.read(path)
print("ready", flush=True)
sys.stdin.readline()
cache.save()
'''
    worker = subprocess.Popen([sys.executable, "-c", script, str(path), str(cache_path), str(metadata)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
    try:
        assert worker.stdout.readline().strip() == "ready"
        cache = SessionTitleCache(cache_path, [str(path)], metadata_path=metadata)
        cache.read(str(path))
        cache.save()
        _metadata(tmp_path, [])
        delete_session_files(path)
        _, error = worker.communicate("save\n", timeout=15)
        assert worker.returncode == 0, error
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.wait()
    assert "Private first prompt" not in cache_path.read_text(encoding="utf-8")


def test_failed_delete_preserves_history_and_cached_title(tmp_path, monkeypatch):
    from jarv import storage
    from jarv.session_store import delete_session_files

    path = history_file(tmp_path, [{"role": "user", "content": "Keep on failure"}])
    cache_path = tmp_path / "session-titles.json"
    cache = SessionTitleCache(cache_path, [str(path)])
    cache.read(str(path))
    cache.save()
    original = (path.read_bytes(), cache_path.read_bytes())

    def fail(*args):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_atomic", fail)
        with pytest.raises(storage.StorageError, match="disk full"):
            delete_session_files(path)
    assert (path.read_bytes(), cache_path.read_bytes()) == original


def test_delete_prunes_previous_archive_location_titles(tmp_path):
    from jarv.session_store import delete_session_files

    path = history_file(tmp_path, [{"role": "user", "content": "Archived secret"}])
    metadata = _metadata(tmp_path, [path])
    cache_path = tmp_path / "session-titles.json"
    cache = SessionTitleCache(cache_path, [str(path)], metadata_path=metadata)
    cache.read(str(path))
    cache.save()
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    archived = archive_dir / "history-archived.json"
    path.rename(archived)
    _metadata(tmp_path, [])
    delete_session_files(archived)
    assert "Archived secret" not in cache_path.read_text(encoding="utf-8")
