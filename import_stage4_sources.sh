#!/usr/bin/env bash

ROOT=/home/jovyan/hmoe-cloud
SRC=$ROOT/src-stage4-import-v2
LOG=$ROOT/logs/import-stage4-sources-$(date +%F_%H%M%S).log
MCORE_SHA=571370c829ca768fe37244f4e2e7f28d8accc4ab
EO_SHA=1effa026ff096b7fa1063ca2fba19d98be6e6cdf
HMO_REPO=https://github.com/AverageMetaheuristicsEnjoyer/H-MoE-Part-cloud.git
mkdir -p "$ROOT/logs" "$SRC"

(
    set -eu
    export PYTHONNOUSERSITE=1
    echo "phase=download_mcore"
    mkdir "$SRC/Megatron-LM"
    /home/user/conda/bin/python -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' \
        "https://github.com/NVIDIA/Megatron-LM/archive/$MCORE_SHA.tar.gz" "$SRC/mcore.tar.gz"
    tar --strip-components=1 -xzf "$SRC/mcore.tar.gz" -C "$SRC/Megatron-LM"
    echo "phase=download_emerging_optimizers"
    mkdir "$SRC/emerging-optimizers"
    /home/user/conda/bin/python -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' \
        "https://github.com/NVIDIA-NeMo/Emerging-Optimizers/archive/$EO_SHA.tar.gz" "$SRC/emerging-optimizers.tar.gz"
    tar --strip-components=1 -xzf "$SRC/emerging-optimizers.tar.gz" -C "$SRC/emerging-optimizers"
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
