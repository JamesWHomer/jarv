"""Shared install ownership detection for updates and uninstalls."""

from __future__ import annotations

import json
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstallChannel:
    kind: str
    executable: Path
    manual_command: str | None = None
    update_command: str | None = None


def _path_parts(path: Path) -> list[str]:
    return [part.casefold() for part in path.parts]


def _has_adjacent_parts(parts: list[str], first: str, second: str) -> bool:
    return any(parts[index:index + 2] == [first, second] for index in range(len(parts) - 1))


def _is_pipx_env() -> bool:
    """Keep the interpreter's symlink path, which identifies its tool environment."""
    return "pipx" in _path_parts(Path(sys.executable))


def _is_uv_tool_env() -> bool:
    return _has_adjacent_parts(_path_parts(Path(sys.executable)), "uv", "tools")


def _is_editable_install() -> bool:
    import importlib.metadata

    try:
        direct_url = importlib.metadata.distribution("jarv").read_text("direct_url.json")
        if not direct_url:
            return False
        metadata = json.loads(direct_url)
        return bool(metadata.get("dir_info", {}).get("editable"))
    except Exception:
        return False


def detect_install_channel() -> InstallChannel:
    """Detect which installer owns the currently running Jarv executable."""
    from .standalone import is_standalone_install

    executable = Path(sys.executable)
    if is_standalone_install():
        candidates = [executable]
        with suppress(OSError):
            resolved = executable.resolve()
            if resolved != executable:
                candidates.append(resolved)

        candidate_parts = [_path_parts(path) for path in candidates]
        if any(_has_adjacent_parts(parts, "microsoft", "winget") for parts in candidate_parts):
            return InstallChannel(
                "winget", executable,
                "winget uninstall JamesWHomer.Jarv",
                "winget upgrade --id JamesWHomer.Jarv --exact",
            )
        if any(_has_adjacent_parts(parts, "scoop", "apps") for parts in candidate_parts):
            return InstallChannel(
                "scoop", executable, "scoop uninstall jarv", "scoop update jarv",
            )
        if any("cellar" in parts or "linuxbrew" in parts for parts in candidate_parts):
            return InstallChannel(
                "brew", executable, "brew uninstall jarv", "brew upgrade jarv",
            )
        return InstallChannel("standalone", executable)

    if _is_editable_install():
        return InstallChannel("editable", executable)
    if _is_pipx_env():
        return InstallChannel("pipx", executable, "pipx uninstall jarv")
    if _is_uv_tool_env():
        return InstallChannel("uv", executable, "uv tool uninstall jarv")
    python = str(executable)
    if " " in python:
        python = f'"{python}"'
    return InstallChannel("pip", executable, f"{python} -m pip uninstall jarv")
