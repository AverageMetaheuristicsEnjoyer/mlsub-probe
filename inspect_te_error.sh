#!/usr/bin/env bash
set -eu

ROOT=/home/jovyan/hmoe-cloud
LATEST=$(find "$ROOT/logs" -maxdepth 1 -type f -name 'te-build-*.log' | sort | tail -n 1)
echo "LOG=$LATEST"
echo "=== compiler failures ==="
grep -i -E -C 8 '(fatal error:|error: |FAILED:|killed|out of memory|internal compiler|no such file|undefined reference|collect2:)' "$LATEST" | tail -n 500 || true
echo "=== remaining extension objects ==="
BUILD=$ROOT/src/TransformerEngine/build/temp.linux-x86_64-cpython-310
find "$BUILD" -type f -name '*.o' | wc -l
du -sh "$ROOT/src/TransformerEngine/build" "$ROOT/wheels"
