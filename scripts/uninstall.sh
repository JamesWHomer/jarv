#!/usr/bin/env sh
set -eu

INSTALL_DIR="${JARV_INSTALL_DIR:-$HOME/.local/bin}"
PURGE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dir)
      INSTALL_DIR="${2:?--dir requires a value}"
      shift 2
      ;;
    --purge)
      PURGE=1
      shift
      ;;
    -h|--help)
      echo "Usage: uninstall.sh [--dir DIR] [--purge]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

rm -f "$INSTALL_DIR/jarv"
rmdir "$INSTALL_DIR" 2>/dev/null || true

if [ "$PURGE" -eq 1 ]; then
  printf 'Delete user data in %s/.jarv and cached clipboard images? [y/N] ' "$HOME"
  read -r answer || answer=""
  case "$answer" in
    y|Y|yes|YES|Yes)
      rm -rf "$HOME/.jarv" "${TMPDIR:-/tmp}/jarv-clipboard"
      echo "Removed Jarv user data."
      ;;
    *) echo "Kept Jarv user data." ;;
  esac
fi

echo "Uninstalled Jarv from $INSTALL_DIR."
