#!/usr/bin/env bash

ROOT=/home/jovyan/hmoe-cloud
VENV=$ROOT/te-venv
LOG=$ROOT/logs/torch-import-trace-$(date +%F_%H%M%S).log
mkdir -p "$ROOT/logs"

(
    set -u
    export PYTHONNOUSERSITE=1
    nvidia_libs=$(find "$VENV/lib/python3.10/site-packages/nvidia" /home/user/conda/lib/python3.10/site-packages/nvidia -type d -name lib -printf '%p:' 2>/dev/null)
    export LD_LIBRARY_PATH="$nvidia_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "=== loader trace ==="
    timeout 45 env LD_DEBUG=libs "$VENV/bin/python" -c 'import torch; print(torch.__version__)' > "$ROOT/logs/torch-loader-debug.log" 2>&1
    echo "loader_exit=$?"
    tail -n 120 "$ROOT/logs/torch-loader-debug.log"
    if command -v strace >/dev/null; then
        echo "=== syscall trace ==="
        timeout 45 strace -f -tt -o "$ROOT/logs/torch-import.strace" "$VENV/bin/python" -c 'import torch; print(torch.__version__)'
        echo "strace_exit=$?"
        tail -n 160 "$ROOT/logs/torch-import.strace"
    else
        echo "strace=unavailable"
    fi
) > "$LOG" 2>&1

CODE=$?
echo "EXIT=$CODE"
echo "LOG=$LOG"
tail -n 260 "$LOG"
exit 0
