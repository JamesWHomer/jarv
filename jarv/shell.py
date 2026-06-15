import os
import platform
import signal
import subprocess
import time
from dataclasses import dataclass

from .cancellation import CancellationToken, TurnCancelled
from rich.console import Group
from rich.text import Text

from .display import output_renderable, console


COMMAND_OUTPUT_UNSET = object()


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
        resolved.append(value)
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
) -> CommandResult:
    try:
        timeout = float(timeout)
        if timeout <= 0:
            timeout = 60
    except (TypeError, ValueError):
        timeout = 60

    try:
        if platform.system() == "Windows":
            # Match the shell we advertise to the model in get_system_info().
            # subprocess with shell=True uses cmd.exe on Windows, which breaks
            # PowerShell commands like Get-ChildItem.
            shell_command = [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ]
            proc = subprocess.Popen(
                shell_command,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        else:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
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
                    return CommandResult(command, stdout or "", stderr or "", proc.returncode, timeout=timeout)
                except subprocess.TimeoutExpired:
                    continue
        except KeyboardInterrupt:
            _kill_process_tree(proc)
            proc.wait()
            raise
        except subprocess.TimeoutExpired:
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


def display_command_result(result: CommandResult) -> None:
    console.print(command_result_renderable(result))


def command_result_renderable(result: CommandResult):
    parts = []
    if result.stdout:
        parts.append(output_renderable(result.stdout.rstrip()))
    if result.stderr:
        parts.append(Text("stderr:", style="bold red"))
        parts.append(output_renderable(result.stderr.rstrip()))
    if result.timed_out:
        parts.append(
            Text(f"Timed out after {result.timeout:g}s", style="bold red")
        )
    elif result.exit_code not in (None, 0):
        exit_line = Text("Exit code: ", style="bold red")
        exit_line.append(str(result.exit_code))
        parts.append(exit_line)
    else:
        parts.append(Text("Exit code: 0", style="dim"))
    if not result.stdout and not result.stderr:
        parts.insert(0, Text("(no output)", style="dim"))
    return Group(*parts)

