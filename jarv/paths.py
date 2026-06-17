"""Filesystem locations used by Jarv."""

from pathlib import Path


CONFIG_DIR = Path.home() / ".jarv"
CONFIG_FILE = CONFIG_DIR / "config.json"
