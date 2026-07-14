"""Filesystem locations used by Jarv."""

from pathlib import Path


CONFIG_DIR = Path.home() / ".jarv"
CONFIG_FILE = CONFIG_DIR / "config.json"
SESSIONS_FILE = CONFIG_DIR / "sessions.json"
SESSIONS_DIR = CONFIG_DIR / "sessions"
ARCHIVE_DIR = CONFIG_DIR / "archive"
UNINSTALL_RESULT_FILE = CONFIG_DIR / "uninstall-result.json"
