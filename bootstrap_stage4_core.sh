#!/usr/bin/env bash

ROOT=/home/jovyan/hmoe-cloud
SRC=$ROOT/src
VENV=$ROOT/stage4-venv
LOG=$ROOT/logs/bootstrap-stage4-core-$(date +%F_%H%M%S).log
MCORE_SHA=571370c829ca768fe37244f4e2e7f28d8accc4ab
EO_SHA=1effa026ff096b7fa1063ca2fba19d98be6e6cdf
mkdir -p "$ROOT/logs" "$SRC"

(
    set -eu
    export PYTHONNOUSERSITE=1
    if [ ! -d "$SRC/Megatron-LM/.git" ]; then
        git clone https://github.com/NVIDIA/Megatron-LM.git "$SRC/Megatron-LM"
    fi
    if [ ! -d "$SRC/emerging-optimizers/.git" ]; then
        git clone https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git "$SRC/emerging-optimizers"
    fi
    git -C "$SRC/Megatron-LM" checkout --detach "$MCORE_SHA"
    git -C "$SRC/emerging-optimizers" checkout --detach "$EO_SHA"
    if [ ! -x "$VENV/bin/python" ]; then
        /home/user/conda/bin/python -m venv --system-site-packages "$VENV"
    fi
    "$VENV/bin/python" -m pip install --no-deps "$SRC/Megatron-LM"
    "$VENV/bin/python" -m pip install --no-deps "$SRC/emerging-optimizers"
    "$VENV/bin/python" - <<'PY'
import torch
import megatron.core
from megatron.core.optimizer.emerging_optimizers import HAVE_EMERGING_OPTIMIZERS
import emerging_optimizers
print("torch=", torch.__version__)
print("mcore=", megatron.core.__version__)
print("emerging_optimizers=", emerging_optimizers.__version__)
print("have_emerging_optimizers=", HAVE_EMERGING_OPTIMIZERS)
print("stage4_core=PASS")
PY
) > "$LOG" 2>&1
CODE=$?

echo "EXIT=$CODE"
echo "LOG=$LOG"
tail -n 240 "$LOG"
exit 0
