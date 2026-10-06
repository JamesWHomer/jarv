from .storage import read_json, write_json, StorageError
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
        self._reserved_labels: set[str] = set()
        self._lock = Lock()

    def reserve_labels(self, labels: set[str]) -> None:
        """Claim a whole spawn batch atomically, or leave the store unchanged.

        Keep claims for this store's lifetime, even after failure or cancellation:
        cancelled workers can still be unwinding while another batch starts.
        Persisted artifacts also block reuse when a session is loaded again.
        """
        with self._lock:
            conflicts = labels & (self._reserved_labels | self._items.keys())
            if conflicts:
                label = min(conflicts)
                raise ValueError(
                    f"child label '{label}' is already used in this session; "
                    "choose a new label"
                )
            self._reserved_labels.update(labels)

    def put(self, label: str, longform: str, tldr: str, owner: str) -> None:
        with self._lock:
            if label in self._items:
                raise ValueError(f"artifact label '{label}' already exists")
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
    data = read_json(path, {}, dict)
    store.baseline = data.baseline
    for label, item in data.items():
        if not isinstance(item, dict) or any(
            not isinstance(item.get(field, ""), str)
            for field in ("longform", "tldr", "owner_label")
        ):
            raise StorageError(f"Invalid artifact record {label!r} in {path}")
        store.put(
            label,
            item.get("longform", ""),
            item.get("tldr", ""),
            item.get("owner_label", label),
        )
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
    write_json(path, data, snapshot=store)
