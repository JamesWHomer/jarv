"""Disposable title cache, independent of transcript indexing and metadata writes."""

import json
import os
import tempfile
from pathlib import Path

from .session_browser_render import first_prompt, one_line
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


class SessionTitleCache:
    def __init__(self, path: Path, history_paths):
        self.path = path
        self.history_paths = set(history_paths)
        self.dirty = False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = data.get("titles", {}) if data.get("version") == 1 else {}
            self.entries = {
                name: entry for name, entry in entries.items()
                if name in self.history_paths and isinstance(entry, dict)
                and isinstance(entry.get("snippet"), str)
                and len(entry["snippet"]) <= 240
                and isinstance(entry.get("mtime_ns"), int) and isinstance(entry.get("size"), int)
            }
        except (OSError, ValueError, AttributeError):
            self.entries = {}

    @staticmethod
    def _stamp(path):
        stat = Path(path).stat()
        return path, stat.st_mtime_ns, stat.st_size

    def _lookup(self, stamp):
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
        if self.entries.get(path) == entry:
            return False
        self.entries[path] = entry
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
        """Best effort: another process may replace this disposable cache too."""
        if not self.dirty:
            return
        name = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=".jarv-titles-", delete=False) as stream:
                name = stream.name
                json.dump({"version": 1, "titles": self.entries}, stream, separators=(",", ":"))
            os.replace(name, self.path)
            self.dirty = False
        except OSError:
            pass  # Cache failures must never prevent opening a conversation.
        finally:
            if name is not None:
                try:
                    Path(name).unlink(missing_ok=True)
                except OSError:
                    pass
