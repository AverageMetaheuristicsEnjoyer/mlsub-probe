#!/usr/bin/env bash

ROOT=/home/jovyan/hmoe-cloud
CUDA_PREFIX=$ROOT/cuda-12.1.1
VENV=$ROOT/te-venv
LOG=$ROOT/logs/te-import-$(date +%F_%H%M%S).log
mkdir -p "$ROOT/logs"

(
    set -u
    export PYTHONNOUSERSITE=1
    export CUDA_HOME=$CUDA_PREFIX
    export CUDA_PATH=$CUDA_PREFIX
    export PATH=$VENV/bin:$CUDA_PREFIX/bin:$PATH
    nvidia_libs=$(find "$VENV/lib/python3.10/site-packages/nvidia" /home/user/conda/lib/python3.10/site-packages/nvidia -type d -name lib -printf '%p:' 2>/dev/null)
    export LD_LIBRARY_PATH="$nvidia_libs$CUDA_PREFIX/lib:$CUDA_PREFIX/lib64:$CUDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    echo "=== runtime ==="
    "$CUDA_PREFIX/bin/nvcc" --version | tail -3
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "=== torch ==="
    timeout 90 "$VENV/bin/python" -X faulthandler - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
PY
    echo "=== transformer engine ==="
    timeout 90 "$VENV/bin/python" -X faulthandler - <<'PY'
import transformer_engine
import transformer_engine.pytorch
import transformer_engine_torch
print("te=", transformer_engine.__version__)
print("te_torch=", transformer_engine_torch.__file__)
PY
) > "$LOG" 2>&1
CODE=$?

echo "EXIT=$CODE"
echo "LOG=$LOG"
tail -n 200 "$LOG"
exit 0
