#!/usr/bin/env bash

ROOT=/home/jovyan/hmoe-cloud
LOG=$ROOT/logs/conda-torch-$(date +%F_%H%M%S).log
mkdir -p "$ROOT/logs"

(
    set -eu
    export PYTHONNOUSERSITE=1
    timeout 90 /home/user/conda/bin/python - <<'PY'
import sys
import torch
print("python=", sys.executable)
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
print("torch_import=PASS")
PY
) > "$LOG" 2>&1
CODE=$?

echo "EXIT=$CODE"
echo "LOG=$LOG"
tail -n 120 "$LOG"
exit 0
