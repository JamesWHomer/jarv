"""Regenerate the vendored models.dev snapshot shipped inside the package.

Usage:
    python scripts/update_models_dev.py [--url URL] [--out PATH]

Downloads the full models.dev catalog, keeps only the providers Jarv can talk
to and the fields Jarv reads, and writes a compact snapshot. The snapshot is the
offline floor for pricing, context limits, modalities, and reasoning options;
``jarv.models_dev.refresh`` layers a newer copy on top at runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarv.models_dev import DEFAULT_URL, MODEL_FIELDS, PROVIDER_FIELDS, prune_catalog
from jarv.provider_catalog import models_dev_provider_ids


def _download(url: str) -> tuple[dict, str]:
    import httpx

    with httpx.Client(timeout=httpx.Timeout(60, connect=10), follow_redirects=True) as client:
        response = client.get(url, headers={"User-Agent": "jarv-snapshot-updater"})
        response.raise_for_status()
        return response.json(), response.headers.get("ETag", "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "jarv" / "data" / "models_dev.json"),
    )
    parser.add_argument("--source", help="Read from a local api.json instead of the network")
    args = parser.parse_args()

    if args.source:
        payload = json.loads(Path(args.source).read_text(encoding="utf-8"))
        etag = ""
    else:
        payload, etag = _download(args.url)

    wanted = models_dev_provider_ids()
    snapshot = {
        "source": args.url,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "etag": etag,
        "providers": prune_catalog(payload, wanted),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(snapshot, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    models = sum(len(entry.get("models", {})) for entry in snapshot["providers"].values())
    size = out.stat().st_size
    print(f"wrote {out} ({size:,} bytes)")
    print(f"providers: {len(snapshot['providers'])}  models: {models}")
    print(f"model fields kept: {', '.join(sorted(MODEL_FIELDS))}")
    print(f"provider fields kept: {', '.join(sorted(PROVIDER_FIELDS))}")
    for name in sorted(wanted - set(snapshot["providers"])):
        print(f"warning: models.dev has no provider '{name}'", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
