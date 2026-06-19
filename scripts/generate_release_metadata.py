from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


ASSET_PATTERN = re.compile(
    r"^jarv-(?P<version>.+)-(?P<platform>windows|macos|linux)-(?P<architecture>x86_64|arm64|aarch64)\.(?:zip|tar\.gz)$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--installer", type=Path, action="append", default=[])
    args = parser.parse_args()

    version = args.version.removeprefix("v")
    tag = f"v{version}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assets = []
    for source in sorted(args.asset_dir.iterdir()):
        match = ASSET_PATTERN.match(source.name)
        if not match:
            continue
        if match.group("version") != version:
            raise SystemExit(f"{source.name} does not match release version {version}")
        destination = args.output_dir / source.name
        shutil.copy2(source, destination)
        assets.append(
            {
                "version": version,
                "platform": match.group("platform"),
                "architecture": match.group("architecture"),
                "name": destination.name,
                "download_url": (
                    f"https://github.com/{args.repository}/releases/download/{tag}/{destination.name}"
                ),
                "sha256": sha256(destination),
                "size": destination.stat().st_size,
            }
        )

    if not assets:
        raise SystemExit("No standalone binary assets found")

    for installer in args.installer:
        shutil.copy2(installer, args.output_dir / installer.name)

    manifest_path = args.output_dir / "release-manifest.json"
    manifest_path.write_text(
        json.dumps({"version": version, "assets": assets}, indent=2) + "\n",
        encoding="utf-8",
    )

    checksum_lines = []
    for path in sorted(args.output_dir.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        checksum_lines.append(f"{sha256(path)}  {path.name}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

