"""Atomic JSON storage with OS locks, optimistic conflicts and redo recovery."""
import copy
import hashlib
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


class StorageError(RuntimeError):
    pass


class StorageConflict(StorageError):
    pass


_local = threading.local()
_mutex = threading.RLock()


@dataclass
class _Transaction:
    root: Path
    changes: dict = field(default_factory=dict)
    seen: dict = field(default_factory=dict)
    snapshots: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Observation:
    """Compact fallback baseline for callers that detach a loaded snapshot."""

    digest: bytes
    fields: dict | None
    empty: bool

    def __bool__(self):
        return not self.empty


def _digest(value):
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(ensure_ascii=True, sort_keys=True,
                               separators=(",", ":"))
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("ascii"))
    return digest.digest()


def _observe(value):
    # Lists (including complete transcripts) and strings retain one digest,
    # regardless of their size. Dict field signatures retain three-way merge
    # support for detached metadata without retaining its document payloads.
    fields = {key: _observe(item) for key, item in value.items()} if isinstance(value, dict) else None
    return _Observation(_digest(value), fields, not value)


def _state():
    if not hasattr(_local, "seen"):
        _local.seen = {}
        _local.active = None
    return _local


def storage_root(path):
    parent = Path(path).resolve().parent
    return parent.parent if parent.name in ("sessions", "archive") else parent


def _sync_dir(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _atomic(path, data):
    fd, name = tempfile.mkstemp(prefix=".jarv-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_dir(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _read(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return copy.deepcopy(default)
    except (OSError, ValueError, UnicodeError) as exc:
        raise StorageError(f"Cannot read {path}: {exc}") from exc


def _replay(root, changes):
    if not isinstance(changes, dict):
        raise StorageError(f"Invalid transaction journal in {root}")
    targets = []
    for name, value in changes.items():
        target = Path(name).resolve()
        if not target.is_relative_to(root) or target == root:
            raise StorageError(f"Invalid transaction target: {target}")
        targets.append((target, value))
    # Reject a corrupt journal before changing any of its otherwise valid files.
    for target, value in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if value is None:
            target.unlink(missing_ok=True)
            _sync_dir(target.parent)
        else:
            _atomic(target, value)


@contextmanager
def transaction(path):
    """Batch a directory's writes. A durable journal is the commit point."""
    root = storage_root(path)
    state = _state()
    if state.active is not None:
        if state.active.root != root:
            raise StorageError("A transaction cannot span storage directories")
        yield
        return
    with _mutex:
        lock = None
        locked = False
        try:
            root.mkdir(parents=True, exist_ok=True)
            lock = (root / ".jarv.lock").open("a+b")
            lock.seek(0, 2)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            deadline = time.monotonic() + 10
            while True:
                try:
                    lock.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise StorageError(f"Timed out waiting for storage lock: {root}")
                    time.sleep(.05)
            journal = root / ".jarv-transaction.json"
            if journal.exists():
                _replay(root, _read(journal, {}))
                journal.unlink()
                _sync_dir(root)
            active = state.active = _Transaction(root)
            yield
            observations = {
                key: _observe(value)
                for key, value in (active.seen | active.changes).items()
            }
            if active.changes:
                _atomic(journal, active.changes)
                _replay(root, active.changes)
                journal.unlink()
                _sync_dir(root)
            state.seen.update(observations)
            for snapshot, value in active.snapshots.values():
                snapshot.baseline = copy.deepcopy(value)
        except (OSError, TypeError, ValueError) as exc:
            raise StorageError(f"Storage transaction failed in {root}: {exc}") from exc
        finally:
            state.active = None
            if lock is not None:
                if locked:
                    lock.seek(0)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                lock.close()


class JsonDict(dict):
    pass


class JsonList(list):
    pass


def _current(path, active):
    key = str(path)
    if key in active.changes:
        return active.changes[key]
    value = _read(path, None)
    if value is None and path.exists():
        raise StorageError(f"Invalid JSON structure in {path}")
    return value


def read_json(path, default, expected_type):
    path = Path(path).resolve()
    with transaction(path):
        state = _state()
        raw = _current(path, state.active)
        value = copy.deepcopy(default if raw is None else raw)
        if not isinstance(value, expected_type):
            raise StorageError(f"Invalid JSON structure in {path}")
        # _current returns storage-owned data. Keep that internal value as the
        # observation; only the result and its public baseline need copies.
        state.active.seen[str(path)] = raw
        result = JsonDict(value) if isinstance(value, dict) else JsonList(value)
        result.baseline = copy.deepcopy(raw)
        return result



_MISSING = object()


def _matches(base, value):
    if isinstance(base, _Observation):
        return value is not _MISSING and base.digest == _digest(value)
    return base == value


def _merge(base, wanted, current, path):
    if _matches(base, wanted):
        return current
    if _matches(base, current) or current == wanted:
        return wanted
    if base is _MISSING and isinstance(wanted, dict) and isinstance(current, dict):
        base = {}
    if isinstance(base, _Observation) and base.fields is not None:
        base = base.fields
    if all(isinstance(v, dict) for v in (base, wanted, current)):
        result = {}
        for key in base.keys() | wanted.keys() | current.keys():
            value = _merge(base.get(key, _MISSING), wanted.get(key, _MISSING), current.get(key, _MISSING), path)
            if value is not _MISSING:
                result[key] = value
        return result
    raise StorageConflict(f"Concurrent changes to {path}; reload before saving")


def write_json(path, value, *, merge=False, baseline=_MISSING, snapshot=None):
    path = Path(path).resolve()
    with transaction(path):
        state = _state()
        active = state.active
        key = str(path)
        wanted = copy.deepcopy(value) if snapshot is not None else None
        current = _current(path, active)
        if snapshot is not None and hasattr(snapshot, "baseline"):
            previous = active.snapshots.get((key, id(snapshot)))
            baseline = previous[1] if previous is not None else snapshot.baseline
        base = (
            active.seen.get(key, state.seen.get(key, current))
            if baseline is _MISSING else baseline
        )
        if merge:
            value = _merge(base or {}, value, current or {}, path)
        elif not _matches(base, current):
            raise StorageConflict(f"Concurrent changes to {path}; reload before saving")
        # Without a merge the staged payload and the snapshot's intended value
        # are identical. Neither escapes the transaction until the baseline is
        # copied at commit, so one frozen copy is sufficient.
        active.changes[key] = wanted if snapshot is not None and not merge else copy.deepcopy(value)
        active.seen[key] = active.changes[key]
        if snapshot is not None:
            active.snapshots[(key, id(snapshot))] = (snapshot, wanted)


def delete_json(path):
    path = Path(path).resolve()
    with transaction(path):
        active = _state().active
        active.changes[str(path)] = None
        active.seen[str(path)] = None
