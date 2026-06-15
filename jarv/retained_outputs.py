import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .display import console


@dataclass(frozen=True)
class RetainedOutput:
    id: str
    content: str


class RetainedOutputStore:
    def __init__(self) -> None:
        self._items: dict[str, RetainedOutput] = {}
        self._lock = Lock()

    def put(self, content: str) -> str:
        with self._lock:
            while True:
                output_id = f"cmd_{uuid.uuid4().hex[:12]}"
                if output_id not in self._items:
                    break
            self._items[output_id] = RetainedOutput(output_id, content)
            return output_id

    def get(self, output_id: str) -> RetainedOutput | None:
        with self._lock:
            return self._items.get(output_id)

    def exists(self, output_id: str) -> bool:
        with self._lock:
            return output_id in self._items


def load_retained_output_store(path: Path) -> RetainedOutputStore:
    store = RetainedOutputStore()
    if not path.exists():
        return store
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
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
                        store._items[output_id] = RetainedOutput(output_id, content)
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
    try:
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    except OSError as e:
        console.print(f"[yellow]Could not save retained outputs:[/yellow] {e}")
