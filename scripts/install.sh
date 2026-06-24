#!/usr/bin/env sh
set -eu

OWNER="JamesWHomer"
REPO="jarv"
VERSION="latest"
INSTALL_DIR="${JARV_INSTALL_DIR:-$HOME/.local/bin}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --version)
      VERSION="${2:?--version requires a value}"
      shift 2
      ;;
    --dir)
      INSTALL_DIR="${2:?--dir requires a value}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: install.sh [--version VERSION] [--dir DIR]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$os" in
  darwin) platform="macos" ;;
  linux) platform="linux" ;;
  *) echo "Unsupported OS: $os" >&2; exit 1 ;;
esac

machine="$(uname -m | tr '[:upper:]' '[:lower:]')"
case "$machine" in
  x86_64|amd64) arch="x86_64" ;;
  arm64) arch="arm64" ;;
  aarch64) if [ "$platform" = "linux" ]; then arch="aarch64"; else arch="arm64"; fi ;;
  *) echo "Unsupported architecture: $machine" >&2; exit 1 ;;
esac

if [ "$VERSION" = "latest" ]; then
  manifest_url="https://github.com/$OWNER/$REPO/releases/latest/download/release-manifest.json"
else
  case "$VERSION" in v*) tag="$VERSION" ;; *) tag="v$VERSION" ;; esac
  manifest_url="https://github.com/$OWNER/$REPO/releases/download/$tag/release-manifest.json"
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT INT TERM

manifest="$tmp_dir/release-manifest.json"
archive="$tmp_dir/jarv.tar.gz"

curl -fsSL "$manifest_url" -o "$manifest"

asset_info="$(python3 - "$manifest" "$platform" "$arch" <<'PY'
import json
import sys

manifest_path, platform, arch = sys.argv[1:]
data = json.load(open(manifest_path, encoding="utf-8"))
for asset in data.get("assets", []):
    if asset.get("platform") == platform and asset.get("architecture") == arch:
        print(asset["download_url"])
        print(asset["sha256"])
        print(asset["name"])
        break
else:
    raise SystemExit(f"No jarv release asset for {platform}/{arch}")
PY
)"

download_url="$(printf '%s\n' "$asset_info" | sed -n '1p')"
expected_sha="$(printf '%s\n' "$asset_info" | sed -n '2p')"
asset_name="$(printf '%s\n' "$asset_info" | sed -n '3p')"
archive="$tmp_dir/$asset_name"

curl -fL "$download_url" -o "$archive"
actual_sha="$(shasum -a 256 "$archive" | awk '{print $1}')"
if [ "$actual_sha" != "$expected_sha" ]; then
  echo "Checksum mismatch for $asset_name" >&2
  echo "expected: $expected_sha" >&2
  echo "actual:   $actual_sha" >&2
  exit 1
fi

tar -xzf "$archive" -C "$tmp_dir"
mkdir -p "$INSTALL_DIR"
install -m 0755 "$tmp_dir/jarv" "$INSTALL_DIR/jarv"

echo "Installed jarv to $INSTALL_DIR/jarv"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *) echo "Add this to PATH: export PATH=\"$INSTALL_DIR:\$PATH\"" ;;
esac
"$INSTALL_DIR/jarv" --version

