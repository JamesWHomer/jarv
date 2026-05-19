import os
import platform
import signal
import subprocess
from dataclasses import dataclass

from .display import display_output, console


@dataclass
class CommandResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False
    timeout: int | float = 60

    def to_model_output(self, max_chars: int | None = None) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.stderr:
            parts.append(f"[stderr] {self.stderr.rstrip()}")
        if self.timed_out:
            parts.append(f"[timed out after {self.timeout:g} seconds]")
        elif self.exit_code not in (None, 0):
            parts.append(f"[exit code {self.exit_code}]")
        output = "\n".join(parts) if parts else "(no output)"
        return truncate_model_output(output, max_chars, label="command output")


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


def execute_command(command: str, timeout: int | float = 60) -> CommandResult:
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
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return CommandResult(command, stdout or "", stderr or "", proc.returncode, timeout=timeout)
        except KeyboardInterrupt:
            _kill_process_tree(proc)
            proc.wait()
            raise
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            stdout, stderr = proc.communicate()
            return CommandResult(command, stdout or "", stderr or "", proc.returncode, timed_out=True, timeout=timeout)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return CommandResult(command, "", f"[error: {e}]", None, timeout=timeout)


def display_command_result(result: CommandResult) -> None:
    if result.stdout:
        display_output(result.stdout.rstrip())
    if result.stderr:
        if result.stdout:
            console.print()
        console.print("stderr:", style="bold red")
        display_output(result.stderr.rstrip())
    if result.timed_out:
        console.print(f"[bold red]Timed out after {result.timeout:g}s[/bold red]")
    elif result.exit_code not in (None, 0):
        console.print(f"[bold red]Exit code:[/bold red] {result.exit_code}")
    else:
        console.print("[dim]Exit code: 0[/dim]")
    if not result.stdout and not result.stderr:
        console.print("(no output)", style="dim")

