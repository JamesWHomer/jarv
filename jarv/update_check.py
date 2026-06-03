import json
import time
import urllib.request

from . import __version__
from .config import CONFIG_DIR
from .display import console

PYPI_VERSION_URL = "https://pypi.org/pypi/jarv/json"
UPDATE_FLAG_FILE = CONFIG_DIR / "update_available.txt"
LAST_CHECK_FILE = CONFIG_DIR / "last_update_check.txt"
UPDATE_CHECK_INTERVAL_HOURS = 24


def _fetch_latest_pypi_version() -> str | None:
    try:
        req = urllib.request.Request(PYPI_VERSION_URL, headers={"User-Agent": "jarv-updater"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:
        return None


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
    CONFIG_DIR.mkdir(exist_ok=True)
    LAST_CHECK_FILE.write_text(str(time.time()))


def _check_update_background() -> None:
    """Check PyPI for a newer version and write a flag file if one is available."""
    if not _should_check_now():
        return
    _record_check_time()
    latest = _fetch_latest_pypi_version()
    if not latest:
        return
    if latest != __version__:
        CONFIG_DIR.mkdir(exist_ok=True)
        UPDATE_FLAG_FILE.write_text(latest)


def maybe_print_update_available() -> None:
    """Show a pending update notification written by a previous run's background check."""
    if not UPDATE_FLAG_FILE.exists():
        return
    try:
        latest = UPDATE_FLAG_FILE.read_text().strip()
        UPDATE_FLAG_FILE.unlink(missing_ok=True)
        if latest and latest != __version__:
            console.print(
                f"[yellow]Update available![/yellow] [dim]v{__version__} -> v{latest}[/dim]  "
                "Run [bold]jarv /update[/bold] to install."
            )
    except Exception:
        pass
