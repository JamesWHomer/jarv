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
