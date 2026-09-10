"""Atomic JSON storage with OS locks, optimistic conflicts and redo recovery."""
import copy
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class StorageError(RuntimeError):
    pass


class StorageConflict(StorageError):
    pass


_local = threading.local()
_mutex = threading.RLock()


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
    for name, value in changes.items():
        target = Path(name).resolve()
        if not target.is_relative_to(root) or target == root:
            raise StorageError(f"Invalid transaction target: {target}")
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
        if state.active[0] != root:
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
            changes = {}
            state.active = (root, changes, [])
            yield
            if changes:
                _atomic(journal, changes)
                _replay(root, changes)
                journal.unlink()
                _sync_dir(root)
                state.seen.update(copy.deepcopy(changes))
                for snapshot, value in state.active[2]:
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


def read_json(path, default, expected_type):
    path = Path(path).resolve()
    with transaction(path):
        state = _state()
        raw = state.active[1][str(path)] if str(path) in state.active[1] else _read(path, None)
        if raw is None and path.exists() and str(path) not in state.active[1]:
            raise StorageError(f"Invalid JSON structure in {path}")
        value = copy.deepcopy(default if raw is None else raw)
        if not isinstance(value, expected_type):
            raise StorageError(f"Invalid JSON structure in {path}")
        state.seen[str(path)] = copy.deepcopy(raw)
        result = JsonDict(value) if isinstance(value, dict) else JsonList(value)
        result.baseline = copy.deepcopy(raw)
        return result



_MISSING = object()


def _merge(base, wanted, current, path):
    if wanted == base:
        return current
    if current == base or current == wanted:
        return wanted
    if base is _MISSING and isinstance(wanted, dict) and isinstance(current, dict):
        base = {}
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
        key = str(path)
        wanted = copy.deepcopy(value)
        current = _read(path, None)
        if current is None and path.exists():
            raise StorageError(f"Invalid JSON structure in {path}")
        if snapshot is not None and hasattr(snapshot, "baseline"):
            baseline = snapshot.baseline
        base = state.seen.get(key, current) if baseline is _MISSING else baseline
        if merge:
            value = _merge(base or {}, value, current or {}, path)
        elif current != base:
            raise StorageConflict(f"Concurrent changes to {path}; reload before saving")
        state.active[1][key] = copy.deepcopy(value)
        if snapshot is not None:
            state.active[2].append((snapshot, wanted))


def delete_json(path):
    path = Path(path).resolve()
    with transaction(path):
        _state().active[1][str(path)] = None
