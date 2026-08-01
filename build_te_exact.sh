#!/usr/bin/env bash

ROOT=/home/jovyan/hmoe-cloud
CUDA_PREFIX=$ROOT/cuda-12.1.1
VENV=$ROOT/te-venv
WHEEL_DIR=$ROOT/wheels
TE_SOURCE=$ROOT/src/TransformerEngine
LOG=$ROOT/logs/te-build-$(date +%F_%H%M%S).log
TE_COMMIT=b9d690e042b1c4e455214e7dab65d6d3512c05d6
mkdir -p "$ROOT/logs" "$ROOT/src" "$WHEEL_DIR"

(
    set -eu
    export PYTHONNOUSERSITE=1

    echo "=== preflight ==="
    hostname
    df -h /home/jovyan
    test -x "$CUDA_PREFIX/bin/nvcc"
    "$CUDA_PREFIX/bin/nvcc" --version

    echo "=== CUDA 12.1 development libraries and cuDNN 9.3 ==="
    CONDA=/home/user/conda/bin/conda
    "$CONDA" install -y -p "$CUDA_PREFIX" \
        -c nvidia/label/cudnn-9.3.0 \
        -c nvidia/label/cuda-12.1.1 \
        -c defaults \
        cuda-version=12.1 libcublas-dev=12.1.3.1 \
        cuda-cudart-static=12.1.105 \
        cuda-driver-dev=12.1.105 cuda-nvrtc-dev=12.1.105 \
        cuda-nvml-dev=12.1.105 \
        cuda-profiler-api=12.1.105 \
        cuda-nvtx=12.1.105 \
        libcufft-dev=11.0.2.54 \
        libcusolver-dev=11.4.5.107 \
        libcusparse-dev=12.1.0.106 \
        libcurand-dev=10.3.2.106 \
        nccl=2.27.7 \
        cudnn=9.3.0.75
    "$CONDA" list -p "$CUDA_PREFIX" | grep -E \
        '^(cuda-(cccl|cudart|driver-dev|nvcc|nvml-dev|nvrtc|nvrtc-dev|nvtx|profiler-api|version)|libcublas|libcublas-dev|libcufft|libcufft-dev|libcurand|libcurand-dev|libcusolver|libcusolver-dev|libcusparse|libcusparse-dev|cudnn|nccl)[[:space:]]'
    du -sh "$CUDA_PREFIX"

    echo "=== isolated Python build environment ==="
    if [ ! -x "$VENV/bin/python" ]; then
        python -m venv --system-site-packages "$VENV"
    fi
    "$VENV/bin/python" -m pip install --upgrade \
        pip setuptools wheel 'cmake>=3.21' ninja 'pybind11>=2.6.0'
    "$VENV/bin/python" - <<'PY'
import site
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("user_site_enabled:", site.ENABLE_USER_SITE)
PY

    echo "=== exact TransformerEngine source ==="
    if [ ! -d "$TE_SOURCE/.git" ]; then
        git clone --recursive https://github.com/NVIDIA/TransformerEngine.git "$TE_SOURCE"
    fi
    git -C "$TE_SOURCE" checkout --detach "$TE_COMMIT"
    git -C "$TE_SOURCE" submodule update --init --recursive
    git -C "$TE_SOURCE" rev-parse HEAD

    export CUDA_HOME=$CUDA_PREFIX
    export CUDA_PATH=$CUDA_PREFIX
    export CUDAToolkit_ROOT=$CUDA_PREFIX
    export CUDNN_HOME=$CUDA_PREFIX
    export CUDNN_PATH=$CUDA_PREFIX
    export PATH=$VENV/bin:$CUDA_PREFIX/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_PREFIX/lib:$CUDA_PREFIX/lib64:$CUDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export CMAKE_PREFIX_PATH=$CUDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
    export NVTE_FRAMEWORK=pytorch
    export NVTE_CUDA_ARCHS=90
    export NVCC_PREPEND_FLAGS=--forward-unknown-to-host-compiler
    export MAX_JOBS=8
    export NVTE_BUILD_MAX_JOBS=8
    export NVTE_BUILD_THREADS_PER_JOB=1

    echo "=== build exact TE wheel for SM90 ==="
    cd "$TE_SOURCE"
    "$VENV/bin/python" -m pip wheel --no-build-isolation --no-deps . -w "$WHEEL_DIR"
    WHEEL=$(find "$WHEEL_DIR" -maxdepth 1 -type f \
        -name 'transformer_engine-2.16.0+b9d690e*.whl' | sort | tail -n 1)
    test -n "$WHEEL"
    sha256sum "$WHEEL"
    "$VENV/bin/python" -m pip install --no-deps --force-reinstall "$WHEEL"
    echo "TE_BUILD=PASS"

    echo "=== CPU import check ==="
    if "$VENV/bin/python" - <<'PY'
import transformer_engine
import transformer_engine.pytorch
import transformer_engine_torch
print("transformer_engine:", transformer_engine.__version__)
print("transformer_engine_torch:", transformer_engine_torch.__file__)
PY
    then
        echo "TE_CPU_IMPORT=PASS"
    else
        echo "TE_CPU_IMPORT=FAIL"
    fi
    "$VENV/bin/python" -m pip check || true
    du -sh "$VENV" "$WHEEL_DIR"
    df -h /home/jovyan
) > "$LOG" 2>&1
CODE=$?

echo "EXIT=$CODE"
echo "LOG=$LOG"
tail -n 300 "$LOG"
exit 0
