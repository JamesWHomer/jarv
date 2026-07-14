from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.version import Version

from .paths import CONFIG_DIR


GITHUB_OWNER = "JamesWHomer"
GITHUB_REPO = "jarv"
GITHUB_RELEASE_BASE = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases"
LATEST_MANIFEST_URL = f"{GITHUB_RELEASE_BASE}/latest/download/release-manifest.json"
MANIFEST_TIMEOUT_SECONDS = 10
DOWNLOAD_TIMEOUT_SECONDS = 60
WINDOWS_UPDATE_RESULT_FILE = CONFIG_DIR / "update-result.json"
WINDOWS_UPDATE_RETRY_COUNT = 40
WINDOWS_UPDATE_RETRY_DELAY_MS = 250


@dataclass(frozen=True)
class ReleaseAsset:
    version: str
    platform: str
    architecture: str
    name: str
    download_url: str
    sha256: str
    size: int | None = None


def is_standalone_install() -> bool:
    return bool(getattr(sys, "frozen", False))


def manifest_url(version: str | None = None) -> str:
    if not version or version == "latest":
        return LATEST_MANIFEST_URL
    tag = version if version.startswith("v") else f"v{version}"
    return f"{GITHUB_RELEASE_BASE}/download/{tag}/release-manifest.json"


def normalize_platform(value: str | None = None) -> str:
    system = (value or platform.system()).lower()
    if system.startswith("win"):
        return "windows"
    if system == "darwin" or system.startswith("mac"):
        return "macos"
    if system == "linux":
        return "linux"
    return system


def normalize_architecture(value: str | None = None, *, target_platform: str | None = None) -> str:
    machine = (value or platform.machine()).lower()
    normalized_platform = normalize_platform(target_platform)
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86-64": "x86_64",
        "arm64": "arm64",
        "aarch64": "aarch64" if normalized_platform == "linux" else "arm64",
    }
    return aliases.get(machine, machine)


def fetch_release_manifest(version: str | None = None) -> dict[str, Any] | None:
    try:
        req = urllib.request.Request(
            manifest_url(version),
            headers={"User-Agent": "jarv-standalone-updater"},
        )
        with urllib.request.urlopen(req, timeout=MANIFEST_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
        validate_manifest(data)
        return data
    except Exception:
        return None


def validate_manifest(manifest: dict[str, Any]) -> None:
    version = str(manifest.get("version") or "")
    Version(version)
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("release manifest has no assets")
    for item in assets:
        if not isinstance(item, dict):
            raise ValueError("release manifest asset must be an object")
        for key in ("platform", "architecture", "name", "download_url", "sha256"):
            if not item.get(key):
                raise ValueError(f"release manifest asset missing {key}")


def select_release_asset(
    manifest: dict[str, Any],
    *,
    target_platform: str | None = None,
    target_architecture: str | None = None,
) -> ReleaseAsset | None:
    release_version = str(manifest["version"])
    wanted_platform = normalize_platform(target_platform)
    wanted_architecture = normalize_architecture(
        target_architecture,
        target_platform=wanted_platform,
    )
    for item in manifest.get("assets", []):
        if (
            str(item.get("platform")) == wanted_platform
            and str(item.get("architecture")) == wanted_architecture
        ):
            size = item.get("size")
            return ReleaseAsset(
                version=release_version,
                platform=wanted_platform,
                architecture=wanted_architecture,
                name=str(item["name"]),
                download_url=str(item["download_url"]),
                sha256=str(item["sha256"]).lower(),
                size=int(size) if isinstance(size, int) else None,
            )
    return None


def latest_standalone_version() -> str | None:
    manifest = fetch_release_manifest()
    if not manifest:
        return None
    return str(manifest.get("version") or "") or None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(asset: ReleaseAsset, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        asset.download_url,
        headers={"User-Agent": "jarv-standalone-updater"},
    )
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp:
        with destination.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    actual = _sha256_file(destination)
    if actual.lower() != asset.sha256.lower():
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {asset.name}: expected {asset.sha256}, got {actual}"
        )
    return destination


def extract_executable(archive_path: Path, destination_dir: Path, *, windows: bool | None = None) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    executable_name = "jarv.exe" if (os.name == "nt" if windows is None else windows) else "jarv"
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extract(executable_name, destination_dir)
    else:
        with tarfile.open(archive_path) as archive:
            member = archive.getmember(executable_name)
            archive.extract(member, destination_dir)
    executable = destination_dir / executable_name
    if not executable.exists():
        raise FileNotFoundError(f"{executable_name} was not found in {archive_path.name}")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return executable


def _windows_updater_creation_flags(*, allow_breakaway: bool = True) -> int:
    # Windows PowerShell can exit before processing ``-File`` when started with
    # DETACHED_PROCESS. CREATE_NO_WINDOW isolates it from the console while
    # still allowing the handoff script to run after Jarv exits.
    flags = (
        getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    )
    if allow_breakaway:
        flags |= getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
    return flags


def _stage_windows_updater(
    source: Path,
    target: Path,
    expected_version: str,
) -> subprocess.Popen:
    script = source.parent / "jarv-update.ps1"
    script.write_text(
        """
param(
  [Parameter(Mandatory=$true)][string]$Source,
  [Parameter(Mandatory=$true)][string]$Target,
  [Parameter(Mandatory=$true)][int]$ParentPid,
  [Parameter(Mandatory=$true)][string]$ExpectedVersion,
  [Parameter(Mandatory=$true)][string]$ResultFile,
  [Parameter(Mandatory=$true)][int]$RetryCount,
  [Parameter(Mandatory=$true)][int]$RetryDelayMs
)
$ErrorActionPreference = "Stop"

function Write-UpdateResult {
  param(
    [Parameter(Mandatory=$true)][string]$Status,
    [Parameter(Mandatory=$true)][string]$Message
  )
  $payload = @{
    status = $Status
    version = $ExpectedVersion
    message = $Message
  } | ConvertTo-Json -Compress
  $temporaryResult = "$ResultFile.tmp"
  Set-Content -LiteralPath $temporaryResult -Value $payload -Encoding UTF8
  Move-Item -LiteralPath $temporaryResult -Destination $ResultFile -Force
}

try {
  Wait-Process -Id $ParentPid -ErrorAction SilentlyContinue
  $installed = $false
  $lastError = $null

  for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
    try {
      Copy-Item -LiteralPath $Source -Destination $Target -Force
      $actualVersion = (& $Target --version 2>&1 | Out-String).Trim()
      if ($LASTEXITCODE -ne 0) {
        throw "The updated executable exited with code $LASTEXITCODE."
      }
      $expectedOutput = "jarv $ExpectedVersion"
      if ($actualVersion -ne $expectedOutput) {
        throw "Expected '$expectedOutput' but got '$actualVersion'."
      }
      $installed = $true
      break
    }
    catch {
      $lastError = $_.Exception.Message
      if ($attempt -lt $RetryCount) {
        Start-Sleep -Milliseconds $RetryDelayMs
      }
    }
  }

  if (-not $installed) {
    throw "Could not replace or verify '$Target' after $RetryCount attempts. $lastError"
  }

  Write-UpdateResult -Status "updated" -Message "Updated successfully to v$ExpectedVersion."
  $temporaryRoot = Split-Path -Parent (Split-Path -Parent $Source)
  Remove-Item -LiteralPath $temporaryRoot -Recurse -Force -ErrorAction SilentlyContinue
}
catch {
  $message = "Update to v$ExpectedVersion failed: $($_.Exception.Message)"
  try {
    Write-UpdateResult -Status "failed" -Message $message
  }
  catch {}
  exit 1
}
""".strip(),
        encoding="utf-8",
    )
    powershell = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
    WINDOWS_UPDATE_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    WINDOWS_UPDATE_RESULT_FILE.unlink(missing_ok=True)
    command = [
        powershell,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Source",
        str(source),
        "-Target",
        str(target),
        "-ParentPid",
        str(os.getpid()),
        "-ExpectedVersion",
        expected_version,
        "-ResultFile",
        str(WINDOWS_UPDATE_RESULT_FILE),
        "-RetryCount",
        str(WINDOWS_UPDATE_RETRY_COUNT),
        "-RetryDelayMs",
        str(WINDOWS_UPDATE_RETRY_DELAY_MS),
    ]
    popen_kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
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


def consume_windows_update_result(
    result_file: str | Path | None = None,
) -> dict[str, str] | None:
    path = Path(result_file or WINDOWS_UPDATE_RESULT_FILE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except Exception:
        path.unlink(missing_ok=True)
        return None
    path.unlink(missing_ok=True)
    if not isinstance(payload, dict) or payload.get("status") not in {"updated", "failed"}:
        return None
    return {
        "status": str(payload["status"]),
        "version": str(payload.get("version") or ""),
        "message": str(payload.get("message") or ""),
    }


def install_standalone_asset(
    asset: ReleaseAsset,
    *,
    executable_path: str | Path | None = None,
    windows: bool | None = None,
) -> str:
    target = Path(executable_path or sys.executable).resolve()
    is_windows = os.name == "nt" if windows is None else windows
    temp_dir = Path(tempfile.mkdtemp(prefix="jarv-update-"))
    archive = temp_dir / asset.name
    try:
        download_asset(asset, archive)
        extracted = extract_executable(archive, temp_dir / "extract", windows=is_windows)
        if is_windows:
            _stage_windows_updater(extracted, target, asset.version)
            return "staged"
        os.replace(extracted, target)
        return "installed"
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
