"""Install-channel-aware uninstall support for Jarv."""

from __future__ import annotations

import ntpath
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .clipboard import clipboard_image_dir
from .display import console
from .paths import CONFIG_DIR, UNINSTALL_RESULT_FILE


@dataclass(frozen=True)
class InstallChannel:
    kind: str
    executable: Path
    manual_command: str | None = None


@dataclass(frozen=True)
class UninstallOutcome:
    """What ``run_uninstall`` did: exit status plus whether anything was removed.

    ``destructive`` is True when the binary was removed/staged for removal or
    user data was purged — callers that keep running afterwards (heads-up mode)
    must exit so they don't re-persist purged data or delay the staged script.
    """

    status: int
    destructive: bool = False


def _path_parts(path: Path) -> list[str]:
    return [part.casefold() for part in path.parts]


def _has_adjacent_parts(parts: list[str], first: str, second: str) -> bool:
    return any(parts[index:index + 2] == [first, second] for index in range(len(parts) - 1))


def detect_install_channel() -> InstallChannel:
    """Detect which installer owns the currently running Jarv executable."""
    from . import commands, standalone

    executable = Path(sys.executable)
    if standalone.is_standalone_install():
        candidates = [executable]
        with suppress(OSError):
            resolved = executable.resolve()
            if resolved != executable:
                candidates.append(resolved)

        candidate_parts = [_path_parts(path) for path in candidates]
        if any(_has_adjacent_parts(parts, "microsoft", "winget") for parts in candidate_parts):
            return InstallChannel("winget", executable, "winget uninstall JamesWHomer.Jarv")
        if any(_has_adjacent_parts(parts, "scoop", "apps") for parts in candidate_parts):
            return InstallChannel("scoop", executable, "scoop uninstall jarv")
        if any("cellar" in parts or "linuxbrew" in parts for parts in candidate_parts):
            return InstallChannel("brew", executable, "brew uninstall jarv")
        return InstallChannel("standalone", executable)

    if commands._is_editable_install():
        return InstallChannel("editable", executable)
    if commands._is_pipx_env():
        return InstallChannel("pipx", executable, "pipx uninstall jarv")
    if commands._is_uv_tool_env():
        return InstallChannel("uv", executable, "uv tool uninstall jarv")
    python = str(executable)
    if " " in python:
        python = f'"{python}"'
    return InstallChannel("pip", executable, f"{python} -m pip uninstall jarv")


def _normalized_path_entry(value: str | Path) -> str:
    return str(value).strip().rstrip("\\/").casefold()


def _path_value_contains(raw_value: str | None, directory: Path) -> bool:
    """Whether a raw registry PATH value contains ``directory``.

    REG_EXPAND_SZ entries come back unexpanded, so ``%VAR%`` forms are also
    compared after expansion.
    """
    if not raw_value:
        return False
    wanted = _normalized_path_entry(directory)
    for part in str(raw_value).split(";"):
        if not part.strip():
            continue
        if _normalized_path_entry(part) == wanted:
            return True
        if "%" in part and _normalized_path_entry(ntpath.expandvars(part)) == wanted:
            return True
    return False


def _windows_user_path_contains(directory: Path) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _kind = winreg.QueryValueEx(key, "Path")
    except (ImportError, FileNotFoundError, OSError):
        return False
    return _path_value_contains(str(value), directory)


def _powershell_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _stage_windows_uninstaller(
    executable: str | Path,
    *,
    remove_path_entry: bool,
    purge: bool,
) -> subprocess.Popen:
    executable = Path(executable)
    install_dir = executable.parent
    temp_dir = Path(tempfile.mkdtemp(prefix="jarv-uninstall-"))
    script = temp_dir / "jarv-uninstall.ps1"

    path_cleanup = ""
    if remove_path_entry:
        path_cleanup = f"""
$PathEntry = {_powershell_literal(install_dir)}
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
$Wanted = $PathEntry.TrimEnd('\\', '/').Trim()
$PathParts = @($UserPath -split ';' | Where-Object {{
  $_ -and -not [string]::Equals($_.TrimEnd('\\', '/').Trim(), $Wanted, [StringComparison]::OrdinalIgnoreCase)
}})
# This matches install.ps1; SetEnvironmentVariable may expand REG_EXPAND_SZ entries.
[Environment]::SetEnvironmentVariable("Path", ($PathParts -join ';'), "User")
"""

    purge_cleanup = ""
    if purge:
        purge_cleanup = f"""
Remove-Item -LiteralPath {_powershell_literal(CONFIG_DIR)} -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath {_powershell_literal(clipboard_image_dir())} -Recurse -Force -ErrorAction SilentlyContinue
"""

    script.write_text(
        f"""
param([Parameter(Mandatory=$true)][int]$ParentPid)
$ErrorActionPreference = "SilentlyContinue"
$Executable = {_powershell_literal(executable)}
$InstallDir = {_powershell_literal(install_dir)}
$ResultFile = {_powershell_literal(UNINSTALL_RESULT_FILE)}
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Wait-Process -Id $ParentPid -ErrorAction SilentlyContinue
for ($Attempt = 0; $Attempt -lt 20; $Attempt++) {{
  Remove-Item -LiteralPath $Executable -Force -ErrorAction SilentlyContinue
  if (-not (Test-Path -LiteralPath $Executable)) {{ break }}
  Start-Sleep -Milliseconds 500
}}
if (Test-Path -LiteralPath $Executable) {{
  # Another jarv instance may still hold the file. Leave PATH and user data
  # untouched and let the next launch surface the failure.
  $Payload = '{{"status":"failed","message":"The previous uninstall could not remove jarv.exe (another instance may have been running). PATH and user data were left unchanged."}}'
  New-Item -ItemType Directory -Path (Split-Path -Parent $ResultFile) -Force | Out-Null
  Set-Content -LiteralPath $ResultFile -Value $Payload -Encoding UTF8
}} else {{
  if ((Test-Path -LiteralPath $InstallDir) -and
      -not (Get-ChildItem -LiteralPath $InstallDir -Force | Select-Object -First 1)) {{
    Remove-Item -LiteralPath $InstallDir -Force -ErrorAction SilentlyContinue
  }}
{path_cleanup}{purge_cleanup}}}
Remove-Item -LiteralPath $ScriptDir -Recurse -Force -ErrorAction SilentlyContinue
""".strip(),
        encoding="utf-8",
    )

    from .standalone import _windows_updater_creation_flags

    powershell = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
    command = [
        powershell,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-ParentPid",
        str(os.getpid()),
    ]
    popen_kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    with suppress(OSError):
        UNINSTALL_RESULT_FILE.unlink(missing_ok=True)
    try:
        try:
            return subprocess.Popen(
                command,
                creationflags=_windows_updater_creation_flags(),
                **popen_kwargs,
            )
        except OSError as exc:
            if getattr(exc, "winerror", None) != 5:
                raise
            return subprocess.Popen(
                command,
                creationflags=_windows_updater_creation_flags(allow_breakaway=False),
                **popen_kwargs,
            )
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _purge_user_data() -> None:
    shutil.rmtree(CONFIG_DIR, ignore_errors=True)
    shutil.rmtree(clipboard_image_dir(), ignore_errors=True)


def _confirm(*, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    isatty = getattr(sys.stdin, "isatty", None)
    if not callable(isatty) or not isatty():
        console.print("[red]Refusing to uninstall without confirmation.[/red] Re-run with [bold]--yes[/bold].")
        return False
    try:
        answer = console.input("[bold yellow]Proceed? [y/N] [/bold yellow]").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
        console.print()
    if answer not in {"y", "yes"}:
        console.print("[dim]Uninstall cancelled.[/dim]")
        return False
    return True


def _print_manager_instructions(channel: InstallChannel) -> None:
    if channel.kind == "editable":
        console.print("[bold]Install channel:[/bold] editable source checkout")
        console.print("Remove the source checkout or environment with your development workflow.")
        return
    console.print(f"[bold]Install channel:[/bold] {channel.kind}")
    console.print("Jarv will not run the package manager while it is active. Run:")
    console.print(f"  [bold cyan]{channel.manual_command}[/bold cyan]")


def run_uninstall(args: list[str] | None = None) -> UninstallOutcome:
    """Uninstall Jarv, or print the command for its owning package manager."""
    args = list(args or [])
    unknown = [arg for arg in args if arg not in {"--purge", "--yes"}]
    if unknown:
        console.print("[bold red]Usage:[/bold red] jarv /uninstall [--purge] [--yes]")
        console.print(f"[red]Unknown argument:[/red] {unknown[0]}")
        return UninstallOutcome(2)

    purge = "--purge" in args
    assume_yes = "--yes" in args
    channel = detect_install_channel()

    if channel.kind != "standalone":
        _print_manager_instructions(channel)
        if not purge:
            console.print("[dim]User data in ~/.jarv is kept.[/dim]")
            return UninstallOutcome(0)
        console.print("[yellow]Purge will delete ~/.jarv and cached clipboard images now.[/yellow]")
        if not _confirm(assume_yes=assume_yes):
            return UninstallOutcome(1)
        _purge_user_data()
        console.print("[bold cyan]✓[/bold cyan] User data removed. Run the command above to uninstall Jarv.")
        return UninstallOutcome(0, destructive=True)

    executable = channel.executable
    install_dir = executable.parent
    remove_path_entry = _windows_user_path_contains(install_dir)
    console.print("[bold]Standalone uninstall[/bold]")
    console.print(f"  Delete [cyan]{executable}[/cyan]")
    console.print(f"  Remove [cyan]{install_dir}[/cyan] if it is empty")
    if sys.platform == "win32" and remove_path_entry:
        console.print("  Remove the install directory from your user PATH")
    if purge:
        console.print("  Delete ~/.jarv and cached clipboard images")
    else:
        console.print("  Keep user data in ~/.jarv")

    if not _confirm(assume_yes=assume_yes):
        return UninstallOutcome(1)

    if sys.platform == "win32":
        try:
            _stage_windows_uninstaller(
                executable,
                remove_path_entry=remove_path_entry,
                purge=purge,
            )
        except OSError as exc:
            console.print(f"[red]Could not stage the uninstaller:[/red] {exc}")
            return UninstallOutcome(1)
        console.print("[bold cyan]✓[/bold cyan] Uninstall staged — completes a moment after Jarv exits.")
        console.print(
            "[dim]If it does not complete, run: "
            "irm https://github.com/JamesWHomer/jarv/releases/latest/download/uninstall.ps1 | iex[/dim]"
        )
        return UninstallOutcome(0, destructive=True)

    try:
        executable.unlink()
    except OSError as exc:
        console.print(f"[red]Could not remove {executable}:[/red] {exc}")
        return UninstallOutcome(1)
    with suppress(OSError):
        install_dir.rmdir()
    if purge:
        _purge_user_data()
    console.print("[bold cyan]✓[/bold cyan] Uninstalled.")
    return UninstallOutcome(0, destructive=True)


def cmd_uninstall(args: list[str] | None = None) -> int:
    return run_uninstall(args).status
