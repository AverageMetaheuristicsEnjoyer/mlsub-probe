#!/usr/bin/env bash

ROOT=/home/jovyan/hmoe-cloud
VENV=$ROOT/te-venv
LOG=$ROOT/logs/repair-torch-runtime-$(date +%F_%H%M%S).log
mkdir -p "$ROOT/logs"

(
    set -eu
    export PYTHONNOUSERSITE=1
    "$VENV/bin/python" -m pip install --upgrade --force-reinstall nvidia-nvjitlink-cu12==12.6.20
    nvidia_libs=$(find "$VENV/lib/python3.10/site-packages/nvidia" -type d -name lib -printf '%p:' 2>/dev/null)
    export LD_LIBRARY_PATH="$nvidia_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$VENV/bin/python" - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
PY
) > "$LOG" 2>&1
CODE=$?

echo "EXIT=$CODE"
echo "LOG=$LOG"
tail -n 200 "$LOG"
exit 0
