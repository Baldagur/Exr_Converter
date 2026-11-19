#!/usr/bin/env bash
# Build the standalone Linux executable: dist/ExrConverter
# Needs internet on the BUILD machine only. The result runs offline on Rocky Linux 8/9.
#   PYTHON=python3.11 bash packaging/build_linux.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
FFMPEG_URL=${FFMPEG_URL:-}

"$PY" -m venv .venv-build
# shellcheck disable=SC1091
. .venv-build/bin/activate
python -m pip install -q -U pip
python -m pip install -q -r requirements.txt pyinstaller pytest

echo "== unit tests"
python -m pytest -q tests

mkdir -p build_deps
if [ ! -x build_deps/ffmpeg ]; then
  echo "== downloading static ffmpeg"
  if [ -z "$FFMPEG_URL" ]; then
    # Newest release-branch static GPL build from BtbN (has libx264 + prores_ks), else master.
    FFMPEG_URL=$(curl -fsSL https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest | python -c '
import json, re, sys
assets = {a["name"]: a["browser_download_url"] for a in json.load(sys.stdin)["assets"]}
rel = [n for n in assets if re.fullmatch(r"ffmpeg-n[\d.]+-latest-linux64-gpl-[\d.]+\.tar\.xz", n)]
key = lambda n: [int(x) for x in re.findall(r"\d+", n.split("-")[1])]
name = max(rel, key=key) if rel else "ffmpeg-master-latest-linux64-gpl.tar.xz"
print(assets[name])')
  fi
  echo "$FFMPEG_URL"
  curl -fsSL -o build_deps/ffmpeg.tar.xz "$FFMPEG_URL"
  tar -xJf build_deps/ffmpeg.tar.xz -C build_deps --strip-components=2 --wildcards '*/bin/ffmpeg'
  rm -f build_deps/ffmpeg.tar.xz
  chmod +x build_deps/ffmpeg
fi
build_deps/ffmpeg -hide_banner -version | head -1

echo "== self-test from source"
rm -rf /tmp/exrconv_selftest_src
EXRCONV_FFMPEG="$PWD/build_deps/ffmpeg" QT_QPA_PLATFORM=offscreen python run.py --selftest /tmp/exrconv_selftest_src

echo "== pyinstaller"
pyinstaller --noconfirm --clean --distpath dist --workpath build packaging/exr_converter.spec

echo "== self-test of the built app"
rm -rf /tmp/exrconv_selftest
PYTHONUNBUFFERED=1 QT_QPA_PLATFORM=offscreen ./dist/ExrConverter --selftest /tmp/exrconv_selftest
