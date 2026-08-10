#!/usr/bin/env bash

ROOT=/home/jovyan/hmoe-cloud
VENV=$ROOT/torch251-clean
LOG=$ROOT/logs/bootstrap-clean-torch-$(date +%F_%H%M%S).log
mkdir -p "$ROOT/logs"

(
    set -eu
    if [ ! -x "$VENV/bin/python" ]; then
        python -m venv "$VENV"
    fi
    export PYTHONNOUSERSITE=1
    "$VENV/bin/python" -m pip install --upgrade pip
    timeout 1200 "$VENV/bin/python" -m pip install \
        --index-url https://download.pytorch.org/whl/cu121 \
        --timeout 60 --retries 2 torch==2.5.1
    nvidia_libs=$(find "$VENV/lib/python3.10/site-packages/nvidia" -type d -name lib -printf '%p:' 2>/dev/null)
    export LD_LIBRARY_PATH="$nvidia_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$VENV/bin/python" - <<'PY'
import sys
import torch
print("python=", sys.version)
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
print("torch_import=PASS")
PY
    "$VENV/bin/python" -m pip check
) > "$LOG" 2>&1
CODE=$?

echo "EXIT=$CODE"
echo "LOG=$LOG"
tail -n 220 "$LOG"
exit 0
