"""Disposable titles coordinated with session deletion and concurrent writers."""

import json
import os
from pathlib import Path
from threading import Lock

from .session_browser_render import first_prompt, one_line
from .storage import StorageError, delete_json, read_json, storage_root, transaction, write_json
from .unicode_safety import sanitize_text


def _prefix_title(text: str) -> str:
    decoder = json.JSONDecoder()
    pos = len(text) - len(text.lstrip())
    if text[pos:pos + 1] != "[":
        raise ValueError("History is not an array")
    pos += 1
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if text[pos:pos + 1] == "]":
            return ""
        item, pos = decoder.raw_decode(text, pos)
        title = first_prompt([item])
        if title:
            return title[:240]
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if text[pos:pos + 1] == "]":
            return ""
        if text[pos:pos + 1] != ",":
            raise json.JSONDecodeError("Incomplete history", text, pos)
        pos += 1


def _path_key(path):
    return os.path.normcase(str(Path(path).resolve()))


def _entries(data):
    entries = data.get("titles", {}) if isinstance(data, dict) and data.get("version") == 1 else {}
    if not isinstance(entries, dict):
        return {}
    return {
        name: entry for name, entry in entries.items()
        if isinstance(name, str) and isinstance(entry, dict)
        and isinstance(entry.get("snippet"), str) and len(entry["snippet"]) <= 240
        and isinstance(entry.get("mtime_ns"), int) and isinstance(entry.get("size"), int)
    }


def _cache_document(path):
    """A corrupt disposable cache may be replaced, never session metadata."""
    try:
        return read_json(path, {}, dict)
    except StorageError:
        delete_json(path)
        return read_json(path, {}, dict)


def _referenced_paths(metadata_path):
    data = read_json(metadata_path, {"sessions": {}}, dict)
    sessions = data.get("sessions")
    if not isinstance(sessions, dict):
        raise StorageError(f"Invalid session metadata: {metadata_path}")
    return {
        _path_key(meta["history_file"])
        for meta in sessions.values()
        if isinstance(meta, dict) and isinstance(meta.get("history_file"), str) and meta["history_file"]
    }


def forget_session_title(history_path: Path) -> None:
    """Remove cached prompt text in the same commit as permanent deletion."""
    root = storage_root(history_path)
    path = root / "session-titles.json"
    with transaction(path):
        if not path.exists():
            return
        document = _cache_document(path)
        metadata_path = root / "sessions.json"
        referenced = _referenced_paths(metadata_path) if metadata_path.exists() else None
        forgotten = _path_key(history_path)
        titles = {
            name: entry for name, entry in _entries(document).items()
            if _path_key(name) != forgotten
            and (referenced is None or _path_key(name) in referenced)
        }
        wanted = {"version": 1, "titles": titles}
        if document != wanted:
            write_json(path, wanted, snapshot=document)


class SessionTitleCache:
    def __init__(self, path: Path, history_paths, *, metadata_path: Path | None = None):
        self.path = path
        self.history_paths = set(history_paths)
        self.metadata_path = metadata_path
        self._lock = Lock()
        self._pending = {}
        self.dirty = False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = _entries(data)
            self.entries = {
                name: entry for name, entry in entries.items()
                if name in self.history_paths
            }
            self.dirty = data != {"version": 1, "titles": self.entries}
        except (OSError, ValueError, AttributeError):
            self.entries = {}

    @staticmethod
    def _stamp(path):
        stat = Path(path).stat()
        return path, stat.st_mtime_ns, stat.st_size

    def _lookup(self, stamp):
        with self._lock:
            entry = self.entries.get(stamp[0])
        if entry is not None and (entry["mtime_ns"], entry["size"]) == stamp[1:]:
            return sanitize_text(entry["snippet"])
        return None

    def get(self, path):
        if not path:
            return None
        try:
            return self._lookup(self._stamp(path))
        except OSError:
            return None

    def remember(self, stamp, snippet):
        path, mtime_ns, size = stamp
        if not path or mtime_ns is None or size is None:
            return False
        entry = dict(mtime_ns=mtime_ns, size=size, snippet=sanitize_text(one_line(snippet)[:240]))
        with self._lock:
            if self.entries.get(path) == entry:
                return False
            self.entries[path] = entry
            self._pending[path] = entry
            self.dirty = True
        return True

    def read(self, path):
        """Read only an uncached file's beginning, with a strict size cap.

        None means full background indexing must finish this unusually large
        first message. Empty histories and missing files have an empty title.
        """
        if not path:
            return ""
        try:
            stamp = self._stamp(path)
            cached = self._lookup(stamp)
            if cached is not None:
                return cached
            with Path(path).open(encoding="utf-8-sig") as stream:
                text = ""
                for size in (8192, 32768, 131072, 262144):
                    chunk = stream.read(size - len(text))
                    text += chunk
                    try:
                        snippet = _prefix_title(text)
                    except json.JSONDecodeError:
                        if not chunk:
                            return None
                        continue
                    if self._stamp(path) != stamp:
                        return None
                    self.remember(stamp, snippet)
                    return snippet
        except FileNotFoundError:
            return ""
        except (OSError, ValueError):
            pass
        return None

    def save(self):
        """Merge valid updates without restoring a deleted conversation's title."""
        with self._lock:
            if not self.dirty:
                return
            pending = dict(self._pending)
        try:
            with transaction(self.path):
                # Metadata, not this browser's old path inventory, authorizes
                # a cached title. This lock is also held by session deletion.
                referenced = (
                    _referenced_paths(self.metadata_path) if self.metadata_path is not None
                    else {_path_key(path) for path in self.history_paths if path}
                )
                document = _cache_document(self.path)
                candidates = _entries(document) | pending
                titles = {}
                for name, entry in candidates.items():
                    if _path_key(name) not in referenced:
                        continue
                    try:
                        stamp = self._stamp(name)
                    except OSError:
                        continue
                    if stamp[1:] == (entry["mtime_ns"], entry["size"]):
                        titles[name] = entry
                wanted = {"version": 1, "titles": titles}
                if document != wanted:
                    write_json(self.path, wanted, snapshot=document)
            with self._lock:
                for name, entry in pending.items():
                    if self._pending.get(name) == entry:
                        del self._pending[name]
                self.entries = titles | self._pending
                self.dirty = bool(self._pending)
        except (OSError, StorageError, ValueError):
            pass  # Cache failures must never prevent opening a conversation.
