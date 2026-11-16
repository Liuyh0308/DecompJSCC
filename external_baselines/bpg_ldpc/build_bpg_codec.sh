#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SOURCE_DIR="$ROOT_DIR/external_baselines/bpg_ldpc/upstream/libbpg-0.9.8"
BIN_DIR="${BIN_DIR:-$ROOT_DIR/external_baselines/bpg_ldpc/bin}"
FIX_X265_MULTIPOOL_SHUTDOWN="${FIX_X265_MULTIPOOL_SHUTDOWN:-1}"

case "$FIX_X265_MULTIPOOL_SHUTDOWN" in
  0|1) ;;
  *) echo "FIX_X265_MULTIPOOL_SHUTDOWN must be 0 or 1" >&2; exit 1 ;;
esac

if [[ -n "${BUILD_DIR:-}" ]]; then
  mkdir -p "$BUILD_DIR"
else
  BUILD_DIR=$(mktemp -d /tmp/decompjscc-libbpg-0.9.8-build.XXXXXX)
  trap 'rm -rf "$BUILD_DIR"' EXIT
fi

if [[ ! -f "$SOURCE_DIR/VERSION" ]]; then
  echo "Missing frozen libbpg-0.9.8 source: $SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$BIN_DIR"
cp -a "$SOURCE_DIR/." "$BUILD_DIR/"

# Apply the adopted shutdown fix only to the disposable build copy.
if [[ "$FIX_X265_MULTIPOOL_SHUTDOWN" == 1 ]]; then
  patch --batch --forward --fuzz=0 -d "$BUILD_DIR" -p1 < \
    "$ROOT_DIR/external_baselines/bpg_ldpc/patches/x265-multipool-shutdown.patch"
fi

# x265 predates current CMake policy defaults. This patch is applied only to
# the temporary build copy; the frozen upstream snapshot remains unchanged.
python - "$BUILD_DIR/x265/source/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if "cmake_minimum_required(VERSION 3.5)" not in text:
    text = text.replace("# vim: syntax=cmake\n", "# vim: syntax=cmake\ncmake_minimum_required(VERSION 3.5)\n", 1)
text = text.replace("cmake_policy(SET CMP0025 OLD)", "cmake_policy(SET CMP0025 NEW)")
text = text.replace("cmake_policy(SET CMP0054 OLD)", "cmake_policy(SET CMP0054 NEW)")
text = text.replace("cmake_minimum_required (VERSION 2.8.8) # OBJECT libraries require 2.8.8\n", "")
path.write_text(text, encoding="utf-8")
PY

PATH=/usr/bin:/bin make -C "$BUILD_DIR" x265_clean
PATH=/usr/bin:/bin make -C "$BUILD_DIR" -j"${JOBS:-8}" USE_BPGVIEW= bpgenc bpgdec
cp "$BUILD_DIR/bpgenc" "$BUILD_DIR/bpgdec" "$BIN_DIR/"
sha256sum "$BIN_DIR/bpgenc" "$BIN_DIR/bpgdec"
