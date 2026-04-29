from dataclasses import dataclass
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
