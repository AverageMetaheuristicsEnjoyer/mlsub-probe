#!/usr/bin/env bash

ROOT=/home/jovyan/hmoe-cloud
SRC=$ROOT/src-stage4-import-v1
LOG=$ROOT/logs/import-stage4-sources-$(date +%F_%H%M%S).log
MCORE_SHA=571370c829ca768fe37244f4e2e7f28d8accc4ab
EO_SHA=1effa026ff096b7fa1063ca2fba19d98be6e6cdf
HMO_REPO=https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud.git
mkdir -p "$ROOT/logs" "$SRC"

(
    set -eu
    export PYTHONNOUSERSITE=1
    echo "phase=fetch_mcore"
    git init "$SRC/Megatron-LM"
    git -C "$SRC/Megatron-LM" remote add origin https://github.com/NVIDIA/Megatron-LM.git
    git -C "$SRC/Megatron-LM" fetch --depth=1 origin "$MCORE_SHA"
    git -C "$SRC/Megatron-LM" checkout --detach FETCH_HEAD
    echo "phase=fetch_emerging_optimizers"
    git init "$SRC/emerging-optimizers"
    git -C "$SRC/emerging-optimizers" remote add origin https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git
    git -C "$SRC/emerging-optimizers" fetch --depth=1 origin "$EO_SHA"
    git -C "$SRC/emerging-optimizers" checkout --detach FETCH_HEAD
    echo "phase=clone_stage4"
    git clone --depth=1 "$HMO_REPO" "$SRC/H-MoE-Part-cloud"
    echo "phase=imports"
    PYTHONPATH="$SRC/Megatron-LM:$SRC/emerging-optimizers:$SRC/H-MoE-Part-cloud" /home/user/conda/bin/python - <<'PY'
import torch
import megatron.core.optimizer.emerging_optimizers
import megatron.training.arguments
import stage4.fp8_optimizer_states
print("torch=", torch.__version__)
print("stage4_source_import=PASS")
PY
) > "$LOG" 2>&1
CODE=$?

echo "EXIT=$CODE"
echo "LOG=$LOG"
tail -n 240 "$LOG"
exit 0
