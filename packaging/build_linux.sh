#!/usr/bin/env bash
# Build the standalone Linux executable: dist/ExrConverter
# Needs internet on the BUILD machine only. The result runs offline on Rocky Linux 8/9.
#   PYTHON=python3.11 bash packaging/build_linux.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
FFMPEG_URL=${FFMPEG_URL:-https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz}

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
  curl -fsSL -o build_deps/ffmpeg.tar.xz "$FFMPEG_URL"
  tar -xJf build_deps/ffmpeg.tar.xz -C build_deps --strip-components=2 --wildcards '*/bin/ffmpeg'
  rm -f build_deps/ffmpeg.tar.xz
  chmod +x build_deps/ffmpeg
fi
build_deps/ffmpeg -hide_banner -version | head -1

echo "== pyinstaller"
pyinstaller --noconfirm --clean --distpath dist --workpath build packaging/exr_converter.spec

echo "== self-test of the built app"
rm -rf /tmp/exrconv_selftest
QT_QPA_PLATFORM=offscreen ./dist/ExrConverter --selftest /tmp/exrconv_selftest
