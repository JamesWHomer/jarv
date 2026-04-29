import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock


@dataclass(frozen=True)
class Artifact:
    label: str
    longform: str
    tldr: str
    owner_label: str


class ArtifactStore:
    def __init__(self) -> None:
        self._items: dict[str, Artifact] = {}
        self._lock = Lock()

    def put(self, label: str, longform: str, tldr: str, owner: str) -> None:
        with self._lock:
            self._items[label] = Artifact(label, longform, tldr, owner)

    def get(self, label: str) -> Artifact | None:
        with self._lock:
            return self._items.get(label)

    def exists(self, label: str) -> bool:
        with self._lock:
            return label in self._items

    def all_labels(self) -> set[str]:
        with self._lock:
            return set(self._items.keys())


def load_artifact_store(path: Path) -> ArtifactStore:
    store = ArtifactStore()
    if not path.exists():
        return store
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for label, item in data.items():
                if isinstance(item, dict):
                    store.put(
                        label,
                        item.get("longform", ""),
                        item.get("tldr", ""),
                        item.get("owner_label", label),
                    )
    except Exception:
        pass
    return store


def save_artifact_store(store: ArtifactStore, path: Path) -> None:
    from .config import CONFIG_DIR
    CONFIG_DIR.mkdir(exist_ok=True)
    with store._lock:
        data = {
            label: {
                "longform": art.longform,
                "tldr": art.tldr,
                "owner_label": art.owner_label,
            }
            for label, art in store._items.items()
        }
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass
