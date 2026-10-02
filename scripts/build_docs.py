"""Build the docs with Read the Docs' version-specific URL when available."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    command = [sys.executable, "-m", "zensical", "build", "--clean", "--strict"]
    site_url = os.environ.get("READTHEDOCS_CANONICAL_URL")
    if not site_url:
        if os.environ.get("READTHEDOCS") == "True":
            raise SystemExit("Read the Docs did not supply READTHEDOCS_CANONICAL_URL")
        return subprocess.call(command, cwd=ROOT)

    # Zensical's TOML does not interpolate environment variables. Use a sibling
    # config so relative docs/site paths still resolve from the repository root.
    source = (ROOT / "zensical.toml").read_text(encoding="utf-8")
    config, count = re.subn(
        r"(?m)^site_url\s*=.*$",
        lambda _: "site_url = " + json.dumps(site_url.rstrip("/") + "/"),
        source,
    )
    if count != 1:
        raise SystemExit("Expected exactly one site_url in zensical.toml")

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".toml", prefix=".zensical-", dir=ROOT,
        delete=False,
    ) as temporary:
        temporary.write(config)
        config_path = Path(temporary.name)
    try:
        return subprocess.call(command + ["--config-file", str(config_path)], cwd=ROOT)
    finally:
        config_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
