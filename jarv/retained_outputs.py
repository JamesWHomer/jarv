from .storage import read_json, write_json, StorageError
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .display import console


MAX_RETAINED_OUTPUT_CHARS = 4_001_000
MAX_RETAINED_TOTAL_CHARS = 8_000_000
MAX_RETAINED_OUTPUTS = 128
# JSON can escape a non-BMP character as two six-byte surrogate escapes.
MAX_RETAINED_FILE_BYTES = MAX_RETAINED_TOTAL_CHARS * 12 + 128_000


@dataclass(frozen=True)
class RetainedOutput:
    id: str
    content: str


class RetainedOutputStore:
    def __init__(self) -> None:
        self._items: dict[str, RetainedOutput] = {}
        self._lock = Lock()
        self._total_chars = 0

    def _insert(self, output_id: str, content: str) -> None:
        if len(content) > MAX_RETAINED_OUTPUT_CHARS:
            half = MAX_RETAINED_OUTPUT_CHARS // 2
            content = (content[:half] + "\n[retained output limit reached; middle "
                       "characters were discarded and are unavailable for read]\n" + content[-half:])
        while self._items and (len(self._items) >= MAX_RETAINED_OUTPUTS or
                               self._total_chars + len(content) > MAX_RETAINED_TOTAL_CHARS):
            oldest = next(iter(self._items))
            self._total_chars -= len(self._items.pop(oldest).content)
        self._items[output_id] = RetainedOutput(output_id, content)
        self._total_chars += len(content)

    def put(self, content: str) -> str:
        with self._lock:
            while True:
                output_id = f"cmd_{uuid.uuid4().hex[:12]}"
                if output_id not in self._items:
                    break
            self._insert(output_id, content)
            return output_id

    def get(self, output_id: str) -> RetainedOutput | None:
        with self._lock:
            return self._items.get(output_id)

    def exists(self, output_id: str) -> bool:
        with self._lock:
            return output_id in self._items


def load_retained_output_store(path: Path) -> RetainedOutputStore:
    store = RetainedOutputStore()
    try:
        if path.exists() and path.stat().st_size > MAX_RETAINED_FILE_BYTES:
            raise StorageError(f"Retained output file exceeds {MAX_RETAINED_FILE_BYTES} byte limit: {path}")
        data = read_json(path, {}, dict)
        store.baseline = data.baseline
        if isinstance(data, dict):
            for output_id, item in data.items():
                if not isinstance(output_id, str) or not output_id.startswith("cmd_"):
                    continue
                if isinstance(item, str):
                    content = item
                elif isinstance(item, dict):
                    content = item.get("content")
                else:
                    continue
                if isinstance(content, str):
                    with store._lock:
                        store._insert(output_id, content)
    except StorageError:
        raise
    except Exception as e:
        console.print(f"[yellow]Could not load retained outputs:[/yellow] {e}")
    return store


def save_retained_output_store(store: RetainedOutputStore, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with store._lock:
        data = {
            output_id: {"content": item.content}
            for output_id, item in store._items.items()
        }
    write_json(path, data, snapshot=store)
