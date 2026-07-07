import codecs
import json
import os
import platform
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass

from .cancellation import CancellationToken, TurnCancelled
from rich.console import Group
from rich.text import Text

from .display import output_renderable, console


COMMAND_OUTPUT_UNSET = object()
MAX_COMMAND_OUTPUT_WINDOW_CHARS = 200_000

_STATE_CWD_FILE_VAR = "JARV_STATE_CWD_FILE"
_STATE_ENV_FILE_VAR = "JARV_STATE_ENV_FILE"

# Fires even if the command calls `exit`; POSIX guarantees an EXIT trap that
# doesn't itself exit leaves the shell's exit status untouched. `env -0`
# survives values containing newlines.
_POSIX_STATE_TRAP = (
    "trap 'pwd > \"$JARV_STATE_CWD_FILE\"; "
    "env -0 > \"$JARV_STATE_ENV_FILE\"' EXIT\n"
)

# Appended after the user command. `$?`/`$LASTEXITCODE` are snapshotted first
# and re-raised at the end because the appended statements would otherwise
# decide the process exit code. Known corner: a native exe mid-command followed
# by a final successful cmdlet re-raises the stale native exit code.
# BOM-less UTF-8 via .NET; `Out-File`/`>` would write UTF-16 in PowerShell 5.1.
_POWERSHELL_STATE_CAPTURE = """
$__jarvOk = $?
$__jarvExit = $LASTEXITCODE
try {
    $__jarvUtf8 = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($env:JARV_STATE_CWD_FILE, (Get-Location).Path, $__jarvUtf8)
    $__jarvVars = @{}
    foreach ($__jarvE in [System.Environment]::GetEnvironmentVariables().GetEnumerator()) {
        $__jarvVars[[string]$__jarvE.Key] = [string]$__jarvE.Value
    }
    [IO.File]::WriteAllText($env:JARV_STATE_ENV_FILE, (ConvertTo-Json -InputObject $__jarvVars -Compress), $__jarvUtf8)
} catch { }
if ($__jarvExit -is [int]) { exit $__jarvExit } elseif ($__jarvOk) { exit 0 } else { exit 1 }
"""


@dataclass
class ShellState:
    """Shell state replayed into each run_command invocation.

    ``env is None`` means inherit jarv's own environment; after the first
    successful capture it holds the full environment of the last shell.
    """

    cwd: str
    env: dict[str, str] | None = None

    @classmethod
    def initial(cls) -> "ShellState":
        return cls(cwd=os.getcwd(), env=None)

    def copy(self) -> "ShellState":
        return ShellState(self.cwd, dict(self.env) if self.env is not None else None)


_session_shell_state: ShellState | None = None
_session_shell_state_lock = threading.Lock()


def get_session_shell_state() -> ShellState:
    """The shell state shared by every root run_command in this process.

    Module-level rather than per ``run_agent`` call because heads-up mode runs
    one ``run_agent`` per user prompt in the same process.
    """
    global _session_shell_state
    with _session_shell_state_lock:
        if _session_shell_state is None:
            _session_shell_state = ShellState.initial()
        return _session_shell_state


@dataclass
class ShellStateCapture:
    """Parses the cwd/env dump a wrapped shell command leaves behind."""

    cwd_path: str
    env_path: str
    windows: bool
    _applied: bool = False

    def apply(self, state: ShellState) -> None:
        """Fold the dumped cwd/env into ``state``; keep old values on any failure."""
        if self._applied:
            return
        self._applied = True
        try:
            cwd = self._read_cwd()
            if cwd is not None:
                state.cwd = cwd
        except Exception:
            pass
        try:
            env = self._read_env()
            if env is not None:
                state.env = env
        except Exception:
            pass

    def cleanup(self) -> None:
        for path in (self.cwd_path, self.env_path):
            try:
                os.unlink(path)
            except OSError:
                pass

    def _read_cwd(self) -> str | None:
        with open(self.cwd_path, encoding="utf-8-sig") as f:
            text = f.read().strip()
        # isdir also rejects PowerShell provider paths like HKCU:\...
        if text and os.path.isdir(text):
            return text
        return None

    def _read_env(self) -> dict[str, str] | None:
        with open(self.env_path, encoding="utf-8-sig") as f:
            text = f.read()
        if self.windows:
            data = json.loads(text)
            if not isinstance(data, dict):
                return None
            env = {
                key: value
                for key, value in data.items()
                if isinstance(key, str) and isinstance(value, str)
            }
        else:
            env = {}
            for entry in text.split("\0"):
                key, sep, value = entry.partition("=")
                if sep and key:
                    env[key] = value
        env = {
            key: value
            for key, value in env.items()
            # Drop our plumbing vars and cmd.exe's hidden =C: drive-cwd
            # entries, which subprocess rejects in an env= mapping.
            if not key.startswith("=")
            and key.upper() not in (_STATE_CWD_FILE_VAR, _STATE_ENV_FILE_VAR)
        }
        return env or None


@dataclass
class ShellInvocation:
    """Popen arguments for one shell command plus state replay/capture plumbing."""

    popen_args: list[str] | str
    cwd: str | None
    env: dict[str, str] | None
    capture: ShellStateCapture | None


def _shell_popen_args(command: str, windows: bool) -> list[str] | str:
    if windows:
        # Match the shell we advertise to the model in get_system_info().
        # subprocess with shell=True uses cmd.exe on Windows, which breaks
        # PowerShell commands like Get-ChildItem.
        return [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ]
    return command


def build_shell_invocation(command: str, state: ShellState | None) -> ShellInvocation:
    """Build the Popen invocation, wrapping the command for state capture.

    With ``state=None`` the command runs exactly as before: unwrapped,
    inheriting jarv's cwd and environment.
    """
    windows = platform.system() == "Windows"
    if state is None:
        return ShellInvocation(_shell_popen_args(command, windows), None, None, None)

    cwd_fd, cwd_path = tempfile.mkstemp(prefix="jarv-shell-state-")
    env_fd, env_path = tempfile.mkstemp(prefix="jarv-shell-state-")
    os.close(cwd_fd)
    os.close(env_fd)

    # File paths travel via the environment, never string interpolation, so
    # they need no quoting inside the wrapper.
    env = dict(state.env) if state.env is not None else dict(os.environ)
    env[_STATE_CWD_FILE_VAR] = cwd_path
    env[_STATE_ENV_FILE_VAR] = env_path

    if not os.path.isdir(state.cwd):
        # The tracked directory disappeared; heal rather than fail every
        # subsequent Popen.
        state.cwd = os.getcwd()

    if windows:
        wrapped = command + "\n" + _POWERSHELL_STATE_CAPTURE
    else:
        wrapped = _POSIX_STATE_TRAP + command

    return ShellInvocation(
        _shell_popen_args(wrapped, windows),
        state.cwd,
        env,
        ShellStateCapture(cwd_path, env_path, windows),
    )


@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False
    timeout: int | float = 60

    def to_model_output(
        self,
        max_chars: int | None = None,
        *,
        head_chars: int | None = None,
        tail_chars: int | None = None,
    ) -> str:
        output = self.full_model_output()
        if head_chars is not None or tail_chars is not None:
            resolved_head, resolved_tail = resolve_command_output_window(
                COMMAND_OUTPUT_UNSET if head_chars is None else head_chars,
                COMMAND_OUTPUT_UNSET if tail_chars is None else tail_chars,
                max_chars,
            )
            return truncate_command_output(output, resolved_head, resolved_tail)
        return truncate_model_output(output, max_chars, label="command output")

    def full_model_output(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.stderr:
            parts.append(f"[stderr] {self.stderr.rstrip()}")
        if self.timed_out:
            parts.append(f"[timed out after {self.timeout:g} seconds]")
        elif self.exit_code not in (None, 0):
            parts.append(f"[exit code {self.exit_code}]")
        return "\n".join(parts) if parts else "(no output)"


@dataclass
class InteractiveCommandSnapshot:
    command: str
    stdout: str
    stderr: str
    exit_code: int | None
    exited: bool = False
    stdin_closed: bool = False
    stdout_start: int = 0
    stderr_start: int = 0
    elapsed_seconds: float = 0.0
    idle_seconds: float = 0.0
    check_in: bool = False
    check_in_after: float | None = None

    def to_command_result(self) -> CommandResult:
        return CommandResult(
            self.command,
            self.stdout,
            self.stderr,
            self.exit_code,
        )

    @property
    def stdout_delta(self) -> str:
        return self.stdout[self.stdout_start:]

    @property
    def stderr_delta(self) -> str:
        return self.stderr[self.stderr_start:]

    def to_delta_command_result(self) -> CommandResult:
        return CommandResult(
            self.command,
            self.stdout_delta,
            self.stderr_delta,
            self.exit_code,
        )


class InteractiveCommandProcess:
    """A command process whose stdin can be driven across model turns."""

    def __init__(
        self,
        command: str,
        proc: subprocess.Popen,
        shell_state: ShellState | None = None,
        capture: ShellStateCapture | None = None,
    ):
        self.command = command
        self.proc = proc
        self._shell_state = shell_state
        self._capture = capture
        self._stdout_parts: list[str] = []
        self._stderr_parts: list[str] = []
        self._stdout_consumed = 0
        self._stderr_consumed = 0
        self._lock = threading.Lock()
        self._last_output_at = time.monotonic()
        self._started_at = self._last_output_at
        self._stdin_closed = False
        self._stdout_thread = threading.Thread(
            target=self._read_stream,
            args=(proc.stdout, self._stdout_parts),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stream,
            args=(proc.stderr, self._stderr_parts),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    @classmethod
    def start(
        cls,
        command: str,
        shell_state: ShellState | None = None,
    ) -> "InteractiveCommandProcess":
        invocation = build_shell_invocation(command, shell_state)
        # Unbuffered binary pipes: the reader threads decode incrementally and
        # a single read returns whatever bytes are available, so output can be
        # drained quickly when the process exits (text mode forced per-char
        # reads to stay responsive).
        try:
            if platform.system() == "Windows":
                proc = subprocess.Popen(
                    invocation.popen_args,
                    shell=False,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    cwd=invocation.cwd,
                    env=invocation.env,
                    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                )
            else:
                proc = subprocess.Popen(
                    invocation.popen_args,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    cwd=invocation.cwd,
                    env=invocation.env,
                    preexec_fn=os.setsid,
                )
        except Exception:
            if invocation.capture is not None:
                invocation.capture.cleanup()
            raise
        return cls(command, proc, shell_state, invocation.capture)

    def _read_stream(self, stream, target: list[str]) -> None:
        if stream is None:
            return
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending_cr = ""

        def emit(text: str) -> None:
            if not text:
                return
            with self._lock:
                target.append(text)
                self._last_output_at = time.monotonic()

        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    tail = pending_cr + decoder.decode(b"", final=True)
                    emit(tail.replace("\r\n", "\n"))
                    return
                text = pending_cr + decoder.decode(chunk)
                pending_cr = ""
                if text.endswith("\r"):
                    # Hold a trailing CR so a \r\n split across chunks still
                    # collapses to \n; a lone CR (progress-bar redraw) flushes
                    # with the next chunk.
                    pending_cr = "\r"
                    text = text[:-1]
                emit(text.replace("\r\n", "\n"))
        except Exception:
            return

    def snapshot(
        self,
        *,
        consume: bool = False,
        check_in: bool = False,
        check_in_after: float | None = None,
    ) -> InteractiveCommandSnapshot:
        now = time.monotonic()
        with self._lock:
            stdout = "".join(self._stdout_parts)
            stderr = "".join(self._stderr_parts)
            stdout_start = self._stdout_consumed
            stderr_start = self._stderr_consumed
            elapsed_seconds = now - self._started_at
            idle_seconds = now - self._last_output_at
            if consume:
                self._stdout_consumed = len(stdout)
                self._stderr_consumed = len(stderr)
        return InteractiveCommandSnapshot(
            self.command,
            stdout,
            stderr,
            self.proc.poll(),
            exited=self.proc.poll() is not None,
            stdin_closed=self._stdin_closed,
            stdout_start=stdout_start,
            stderr_start=stderr_start,
            elapsed_seconds=elapsed_seconds,
            idle_seconds=idle_seconds,
            check_in=check_in,
            check_in_after=check_in_after,
        )

    @property
    def stdout(self) -> str:
        with self._lock:
            return "".join(self._stdout_parts)

    @property
    def stderr(self) -> str:
        with self._lock:
            return "".join(self._stderr_parts)

    def wait_until_idle(
        self,
        *,
        idle_seconds: float = 2.0,
        first_output_grace_seconds: float = 5.0,
        wait_seconds: float | None = None,
        check_in_seconds: float | None = None,
        require_output: bool = False,
        cancellation_token: CancellationToken | None = None,
    ) -> InteractiveCommandSnapshot:
        start = time.monotonic()
        if check_in_seconds is not None and check_in_seconds <= 0:
            check_in_seconds = None
        with self._lock:
            last_output_at_start = self._last_output_at
        saw_output_during_wait = False
        while True:
            if cancellation_token is not None:
                cancellation_token.throw_if_cancelled()
            if self.proc.poll() is not None:
                # Bounded drain window so output the process wrote just before
                # exiting still lands in the final snapshot.
                deadline = time.monotonic() + 2.0
                for thread in (self._stdout_thread, self._stderr_thread):
                    thread.join(timeout=max(0.0, deadline - time.monotonic()))
                self._finalize_shell_state()
                return self.snapshot(consume=True)
            now = time.monotonic()
            with self._lock:
                saw_output_during_wait = (
                    saw_output_during_wait
                    or self._last_output_at > last_output_at_start
                )
                idle_for = now - self._last_output_at
            if wait_seconds is not None and now - start >= wait_seconds:
                return self.snapshot(consume=True)
            if check_in_seconds is not None and now - start >= check_in_seconds:
                return self.snapshot(
                    consume=True,
                    check_in=True,
                    check_in_after=check_in_seconds,
                )
            if wait_seconds is None and not saw_output_during_wait:
                # ``require_output`` (bare <WAIT>) holds past the grace window:
                # only new output, exit, or the check-in interval end the wait.
                if require_output or now - start < first_output_grace_seconds:
                    time.sleep(0.02)
                    continue
            if idle_for >= idle_seconds and (
                wait_seconds is None or saw_output_during_wait
            ):
                return self.snapshot(consume=True)
            time.sleep(0.02)

    def write_stdin(self, text: str) -> None:
        if self._stdin_closed or self.proc.stdin is None:
            return
        data = text
        if platform.system() == "Windows":
            # The pipe is binary now; keep the CRLF translation text mode did.
            data = data.replace("\r\n", "\n").replace("\n", "\r\n")
        try:
            self.proc.stdin.write(data.encode("utf-8"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._stdin_closed = True

    def close_stdin(self) -> None:
        if self._stdin_closed or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self._stdin_closed = True

    def interrupt(self) -> None:
        if self.proc.poll() is not None:
            return
        if platform.system() == "Windows":
            # Windows has no direct SIGINT for another process group: this is
            # CTRL_BREAK (usually terminates rather than interrupts), and the
            # fallback kills the tree outright.
            try:
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                return
            except Exception:
                _kill_process_tree(self.proc)
                return
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
        except Exception:
            self.proc.send_signal(signal.SIGINT)

    def kill_tree(self) -> None:
        if self.proc.poll() is None:
            _kill_process_tree(self.proc)
        # Cancel path: a killed shell never wrote the state files, so drop the
        # capture (keeping the previous state) and remove the temp files.
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.cleanup()

    def _finalize_shell_state(self) -> None:
        """Apply and discard the state capture once the process has exited."""
        capture, self._capture = self._capture, None
        if capture is None:
            return
        if self._shell_state is not None:
            capture.apply(self._shell_state)
        capture.cleanup()


def resolve_command_output_window(
    head_chars: object,
    tail_chars: object,
    max_chars: int | None,
) -> tuple[int, int]:
    try:
        limit = int(max_chars) if max_chars is not None else 0
    except (TypeError, ValueError):
        limit = 0
    limit = max(0, limit)

    defaults = {
        "head_chars": limit // 2,
        "tail_chars": limit - (limit // 2),
    }
    resolved = []
    for name, value in (("head_chars", head_chars), ("tail_chars", tail_chars)):
        if value is COMMAND_OUTPUT_UNSET:
            resolved.append(defaults[name])
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        if value > MAX_COMMAND_OUTPUT_WINDOW_CHARS:
            raise ValueError(
                f"{name} must be at most {MAX_COMMAND_OUTPUT_WINDOW_CHARS} characters"
            )
        resolved.append(value)
    if resolved[0] + resolved[1] > MAX_COMMAND_OUTPUT_WINDOW_CHARS:
        raise ValueError(
            "head_chars + tail_chars must be at most "
            f"{MAX_COMMAND_OUTPUT_WINDOW_CHARS} characters"
        )
    return resolved[0], resolved[1]


def truncate_command_output(
    output: str,
    head_chars: int,
    tail_chars: int,
    label: str = "command output",
    *,
    retained_id: str | None = None,
    suggested_read_size: int | None = None,
) -> str:
    visible_chars = head_chars + tail_chars
    if len(output) <= visible_chars:
        return output

    omitted = len(output) - visible_chars
    if retained_id is None:
        notice = (
            f"[{label} truncated; showing first {head_chars} and last {tail_chars} "
            f"characters; {omitted} characters omitted from the middle]"
        )
    else:
        read_size = omitted
        if suggested_read_size is not None:
            read_size = min(omitted, max(1, suggested_read_size))
        tail_offset = len(output) - tail_chars
        notice = (
            f"[{label} truncated; id={retained_id}; total_size={len(output)} characters; "
            f"visible ranges=[0,{head_chars}) and [{tail_offset},{len(output)}); "
            f"omitted offset={head_chars} size={omitted}; "
            f"{omitted} characters omitted from the middle; "
            f'use read(input="{retained_id}", offset={head_chars}, size={read_size})]'
        )
    parts = []
    if head_chars:
        parts.append(output[:head_chars])
    parts.append(notice)
    if tail_chars:
        parts.append(output[-tail_chars:])
    return "\n\n".join(parts)


def truncate_model_output(output: str, max_chars: int | None, label: str = "tool output") -> str:
    try:
        limit = int(max_chars) if max_chars is not None else 0
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0 or len(output) <= limit:
        return output

    notice = (
        f"\n\n[{label} truncated to {limit} characters; "
        f"{len(output) - limit} characters omitted from the middle]"
    )
    body_limit = limit - len(notice)
    if body_limit <= 0:
        return output[:limit] + notice

    head = body_limit // 2
    tail = body_limit - head
    return output[:head].rstrip() + notice + "\n\n" + output[-tail:].lstrip()


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if platform.system() == "Windows":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.kill()


def execute_command(
    command: str,
    timeout: int | float = 60,
    cancellation_token: CancellationToken | None = None,
    shell_state: ShellState | None = None,
) -> CommandResult:
    try:
        timeout = float(timeout)
        if timeout <= 0:
            timeout = 60
    except (TypeError, ValueError):
        timeout = 60

    invocation = None
    try:
        invocation = build_shell_invocation(command, shell_state)
        if platform.system() == "Windows":
            proc = subprocess.Popen(
                invocation.popen_args,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=invocation.cwd,
                env=invocation.env,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        else:
            proc = subprocess.Popen(
                invocation.popen_args,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=invocation.cwd,
                env=invocation.env,
                preexec_fn=os.setsid,
            )
        unregister = (
            cancellation_token.register(lambda: _kill_process_tree(proc))
            if cancellation_token is not None else lambda: None
        )
        started = time.monotonic()
        try:
            while True:
                if cancellation_token is not None:
                    cancellation_token.throw_if_cancelled()
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stdout, stderr = proc.communicate(timeout=min(0.05, remaining))
                    if cancellation_token is not None:
                        cancellation_token.throw_if_cancelled()
                    if invocation.capture is not None and shell_state is not None:
                        invocation.capture.apply(shell_state)
                    return CommandResult(command, stdout or "", stderr or "", proc.returncode, timeout=timeout)
                except subprocess.TimeoutExpired:
                    continue
        except KeyboardInterrupt:
            _kill_process_tree(proc)
            proc.wait()
            raise
        except subprocess.TimeoutExpired:
            # Killed mid-flight: the state files are unwritten, keep the
            # previous shell state.
            _kill_process_tree(proc)
            stdout, stderr = proc.communicate()
            return CommandResult(command, stdout or "", stderr or "", proc.returncode, timed_out=True, timeout=timeout)
        finally:
            unregister()
    except KeyboardInterrupt:
        raise
    except TurnCancelled:
        raise
    except Exception as e:
        return CommandResult(command, "", f"[error: {e}]", None, timeout=timeout)
    finally:
        if invocation is not None and invocation.capture is not None:
            invocation.capture.cleanup()


def display_command_result(result: CommandResult) -> None:
    console.print(command_result_renderable(result))


def command_result_renderable(result: CommandResult):
    parts = []
    if result.stdout:
        parts.append(output_renderable(compact_command_output(result.stdout)))
    if result.stderr:
        parts.append(Text("stderr:", style="bold red"))
        parts.append(output_renderable(result.stderr.strip()))
    if result.timed_out:
        parts.append(
            Text(f"Timed out after {result.timeout:g}s", style="bold red")
        )
    elif result.exit_code not in (None, 0):
        exit_line = Text("exit ", style="bold red")
        exit_line.append(str(result.exit_code))
        parts.append(exit_line)
    else:
        parts.append(Text("exit 0", style="dim"))
    if not result.stdout and not result.stderr:
        parts.insert(0, Text("(no output)", style="dim"))
    return Group(*parts)


def compact_command_output(output: str) -> str:
    """Trim shell padding and collapse a one-row table to a compact line."""
    lines = [line.rstrip() for line in output.strip().splitlines()]
    if (
        len(lines) == 3
        and lines[0].strip()
        and lines[1].strip()
        and set(lines[1].strip()) == {"-"}
        and lines[2].strip()
    ):
        return f"{lines[0].strip()}  {lines[2].strip()}"
    return "\n".join(lines)

