import json
import time
import urllib.request

from packaging.version import InvalidVersion, Version

from . import __version__
from .config import CONFIG_DIR
from .display import console
from .standalone import is_standalone_install, latest_standalone_version

PYPI_VERSION_URL = "https://pypi.org/pypi/jarv/json"
UPDATE_FLAG_FILE = CONFIG_DIR / "update_available.txt"
LAST_CHECK_FILE = CONFIG_DIR / "last_update_check.txt"
UPDATE_CHECK_INTERVAL_HOURS = 24
UPDATE_CHECK_TIMEOUT_SECONDS = 5


def _fetch_latest_pypi_release() -> tuple[str, str] | None:
    try:
        req = urllib.request.Request(PYPI_VERSION_URL, headers={"User-Agent": "jarv-updater"})
        with urllib.request.urlopen(req, timeout=UPDATE_CHECK_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
        version = str(data["info"]["version"])
        Version(version)
        if not data.get("urls"):
            return None
        return version, f"jarv=={version}"
    except Exception:
        return None


def _fetch_latest_pypi_version() -> str | None:
    release = _fetch_latest_pypi_release()
    return release[0] if release else None


def _is_newer_version(candidate: str, current: str) -> bool:
    try:
        return Version(candidate) > Version(current)
    except InvalidVersion:
        return False


def _should_check_now() -> bool:
    """Return True if enough time has passed since the last update check."""
    if not LAST_CHECK_FILE.exists():
        return True
    try:
        last = float(LAST_CHECK_FILE.read_text().strip())
        return (time.time() - last) >= UPDATE_CHECK_INTERVAL_HOURS * 3600
    except Exception:
        return True


def _record_check_time() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LAST_CHECK_FILE.write_text(str(time.time()), encoding="utf-8")


def _fetch_latest_update_version() -> str | None:
    if is_standalone_install():
        return latest_standalone_version()
    return _fetch_latest_pypi_version()


def _check_update_background() -> None:
    """Check the active install channel for a newer version and write a flag file."""
    if not _should_check_now():
        return
    latest = _fetch_latest_update_version()
    if not latest:
        return
    _record_check_time()
    if _is_newer_version(latest, __version__):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        UPDATE_FLAG_FILE.write_text(latest, encoding="utf-8")
    else:
        UPDATE_FLAG_FILE.unlink(missing_ok=True)


def maybe_print_update_available() -> None:
    """Show a pending update notification written by a previous run's background check."""
    if not UPDATE_FLAG_FILE.exists():
        return
    try:
        latest = UPDATE_FLAG_FILE.read_text(encoding="utf-8").strip()
        UPDATE_FLAG_FILE.unlink(missing_ok=True)
        if latest and _is_newer_version(latest, __version__):
            console.print(
                f"[yellow]Update available![/yellow] [dim]v{__version__} -> v{latest}[/dim]  "
                "Run [bold]jarv /update[/bold] to install."
            )
    except Exception:
        pass
