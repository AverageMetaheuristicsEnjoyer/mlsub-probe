#!/usr/bin/env bash

ROOT=/home/jovyan/hmoe-cloud
CUDA_PREFIX=$ROOT/cuda-12.1.1
VENV=$ROOT/te-venv
LOG=$ROOT/logs/te-gpu-smoke-$(date +%F_%H%M%S).log
mkdir -p "$ROOT/logs"

(
    set -eu
    export PYTHONNOUSERSITE=1
    export CUDA_HOME=$CUDA_PREFIX
    export CUDA_PATH=$CUDA_PREFIX
    export CUDAToolkit_ROOT=$CUDA_PREFIX
    export CUDNN_HOME=$CUDA_PREFIX
    export CUDNN_PATH=$CUDA_PREFIX
    export PATH=$VENV/bin:$CUDA_PREFIX/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_PREFIX/lib:$CUDA_PREFIX/lib64:$CUDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1

    hostname
    nvidia-smi -L
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
    "$CUDA_PREFIX/bin/nvcc" --version
    "$VENV/bin/python" - <<'PY'
import sys

import torch
import transformer_engine
import transformer_engine.pytorch as te
import transformer_engine_torch
from transformer_engine.common.recipe import DelayedScaling, Float8BlockScaling, Format

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("transformer_engine:", transformer_engine.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("memory_gb:", torch.cuda.get_device_properties(0).total_memory / 2**30)
print("cublaslt:", transformer_engine_torch.get_cublasLt_version())

available, reason = te.is_fp8_block_scaling_available(return_reason=True)
print("block_scaling_available:", available)
print("block_scaling_reason:", reason)

def check_linear(recipe, name):
    linear = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with te.autocast(enabled=True, recipe=recipe):
        y = linear(x)
    y.float().square().mean().backward()
    assert torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert linear.weight.grad is not None and torch.isfinite(linear.weight.grad).all()
    print(f"{name}=PASS dtype={y.dtype}")

if available:
    check_linear(Float8BlockScaling(fp8_format=Format.E4M3), "block_fp8_linear")
else:
    print("block_fp8_linear=SKIP")

check_linear(DelayedScaling(fp8_format=Format.E4M3), "delayed_fp8_linear")
print("TE_GPU_SMOKE=PASS")
PY
) > "$LOG" 2>&1
CODE=$?

echo "EXIT=$CODE"
echo "LOG=$LOG"
tail -n 300 "$LOG"
exit 0
