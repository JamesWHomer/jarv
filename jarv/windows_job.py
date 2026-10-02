"""Own a Windows process tree, including children whose parent has exited."""

import ctypes
from ctypes import wintypes
import threading


class _Limits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
        ("scheduling", wintypes.DWORD),
    ]


class _IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes",
    )]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _Limits), ("io", _IOCounters),
        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
    ]


class WindowsJob:
    """Closing the sole job handle kills the worker and all its descendants.

    Assign before sending any user command. If assignment is unsupported, the
    caller must abandon the worker and use its ordinary fresh-process runner.
    """

    def __init__(self, process):
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(api, name)
            function.argtypes, function.restype = args, result
        self._api = api
        self._lock = threading.Lock()
        self._host_pids = {process.pid}
        self._handle = api.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = _ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not api.SetInformationJobObject(self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not api.AssignProcessToJobObject(self._handle, int(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise

    def _process_ids(self):
        # JOBOBJECT_BASIC_PROCESS_ID_LIST has two DWORDs followed by ULONG_PTRs.
        capacity = 32
        while capacity <= 8192:
            data = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ctypes.c_size_t))
            if self._api.QueryInformationJobObject(self._handle, 3, data, len(data), None):
                count = wintypes.DWORD.from_buffer(data, 4).value
                return set((ctypes.c_size_t * count).from_buffer(data, 8))
            if ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(ctypes.get_last_error())
            capacity *= 2
        raise OSError('Too many processes in PowerShell job')

    def record_host_processes(self):
        # CREATE_NO_WINDOW still creates a hidden conhost.exe on some Windows
        # versions. Capture it before any user code runs, never exempt children
        # by executable name (a command can itself start conhost).
        with self._lock:
            self._host_pids = self._process_ids()

    def has_children(self):
        with self._lock:
            if not self._handle:
                return True
            try:
                return bool(self._process_ids() - self._host_pids)
            except OSError:
                return True  # Do not reuse a worker whose children are unknown.

    def close(self):
        with self._lock:
            if self._handle:
                self._api.CloseHandle(self._handle)
                self._handle = None
