#!/usr/bin/env bash
set -u

ROOT=/home/jovyan/hmoe-cloud
CUDA_PREFIX=$ROOT/cuda-12.1.1
VENV=$ROOT/te-venv

echo "=== persistent state ==="
df -h /home/jovyan
du -sh "$CUDA_PREFIX" "$VENV" "$ROOT/wheels" 2>/dev/null || true
find "$ROOT/wheels" -maxdepth 1 -type f -name '*.whl' -print -exec sha256sum {} \;

echo "=== latest build log ==="
LATEST=$(find "$ROOT/logs" -maxdepth 1 -type f -name 'te-build-*.log' | sort | tail -n 1)
echo "LOG=$LATEST"
tail -n 500 "$LATEST"

echo "=== isolated CPU import ==="
export PYTHONNOUSERSITE=1
export CUDA_HOME=$CUDA_PREFIX
export CUDA_PATH=$CUDA_PREFIX
export CUDAToolkit_ROOT=$CUDA_PREFIX
export CUDNN_HOME=$CUDA_PREFIX
export CUDNN_PATH=$CUDA_PREFIX
export PATH=$VENV/bin:$CUDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PREFIX/lib:$CUDA_PREFIX/lib64:$CUDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
"$VENV/bin/python" - <<'PY'
import torch
import transformer_engine
import transformer_engine.pytorch
import transformer_engine_torch

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("transformer_engine:", transformer_engine.__version__)
print("transformer_engine_torch:", transformer_engine_torch.__file__)
print("TE_CPU_IMPORT=PASS")
PY
