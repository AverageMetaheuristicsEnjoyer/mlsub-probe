#!/usr/bin/env bash
# Inventory: disk volumes, workspace state, leftovers of the TE build campaign.
ROOT=/home/jovyan/hmoe-cloud

{
    echo "=== host ==="
    hostname
    date -u
    nproc
    free -g | head -2

    echo "=== df -h ==="
    df -h

    echo "=== volumes ==="
    for d in /home/jovyan /workspace /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
        echo "--- $d"
        ls -ld "$d" 2>&1
        ls -la "$d" 2>&1 | head -25
        if touch "$d/.probe_write_test" 2>/dev/null; then
            echo "WRITE=OK $d"
            rm -f "$d/.probe_write_test"
        else
            echo "WRITE=FAIL $d"
        fi
    done

    echo "=== hmoe-cloud campaign state ==="
    du -sh "$ROOT" 2>&1
    du -sh "$ROOT"/* 2>&1 | head -20
    "$ROOT/cuda-12.1.1/bin/nvcc" --version 2>&1 | tail -3
    ls -la "$ROOT/wheels" 2>&1 | head
    ls -la "$ROOT/logs" 2>&1 | tail -10

    echo "=== torch inside te-venv ==="
    "$ROOT/te-venv/bin/python" -c "import torch; print('torch', torch.__version__, torch.version.cuda)" 2>&1 | tail -5

    echo "=== libnvJitLink locations ==="
    find "$ROOT/te-venv" -name 'libnvJitLink*' 2>/dev/null | head
    find /home/user/conda -name 'libnvJitLink*' 2>/dev/null | head
    find "$ROOT/cuda-12.1.1" -name 'libnvJitLink*' 2>/dev/null | head
} 2>&1

exit 0
