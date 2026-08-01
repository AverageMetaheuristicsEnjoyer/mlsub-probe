#!/usr/bin/env bash

ROOT=/home/jovyan/hmoe-cloud
PREFIX=$ROOT/cuda-12.1.1
LOG=$ROOT/logs/nvcc-conda-$(date +%F_%H%M%S).log
mkdir -p "$ROOT/logs"

(
    set -eu

    echo "=== base image ==="
    hostname
    python - <<'PY'
import torch
print("python executable:", __import__("sys").executable)
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
PY
    df -h /home/jovyan

    CONDA=/home/user/conda/bin/conda
    test -x "$CONDA"
    "$CONDA" --version

    echo "=== install isolated CUDA 12.1.1 compiler prefix ==="
    if [ ! -x "$PREFIX/bin/nvcc" ]; then
        "$CONDA" create -y -p "$PREFIX" --override-channels \
            -c nvidia/label/cuda-12.1.1 \
            cuda-nvcc=12.1.105 cuda-cudart-dev=12.1.105 cuda-cccl=12.1.109
    fi

    echo "=== nvcc ==="
    "$PREFIX/bin/nvcc" --version
    du -sh "$PREFIX"

    echo "=== SM90 PTX compile ==="
    cat > /tmp/mlsub_cuda121_probe.cu <<'CU'
__global__ void add_one(float *x) { x[threadIdx.x] += 1.0f; }
CU
    "$PREFIX/bin/nvcc" -ptx /tmp/mlsub_cuda121_probe.cu \
        -o /tmp/mlsub_cuda121_probe.ptx -arch=sm_90
    test -s /tmp/mlsub_cuda121_probe.ptx
    head -n 12 /tmp/mlsub_cuda121_probe.ptx

    echo "=== development headers currently visible ==="
    find /home/user/conda "$PREFIX" -type f \
        \( -name cuda_runtime.h -o -name cublas_v2.h -o -name cudnn.h \) \
        -print 2>/dev/null | head -n 50

    echo "NVCC_CONDA_PROBE=PASS"
) > "$LOG" 2>&1
CODE=$?

echo "EXIT=$CODE"
echo "LOG=$LOG"
cat "$LOG"
exit 0
