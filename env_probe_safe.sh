#!/usr/bin/env bash
# Part A of the inventory: everything that cannot touch an unresponsive NFS export.
# Reads /proc/mounts (kernel memory, never blocks) instead of walking every mount.
ROOT=/home/jovyan/hmoe-cloud

run() {   # run <seconds> <label> <command...>
    local t=$1 label=$2
    shift 2
    echo "--- $label"
    timeout -s KILL "$t" "$@" 2>&1
    [ $? -eq 137 ] && echo "KILLED after ${t}s: $label"
    return 0
}

echo "=== host ==="
hostname
date -u
nproc
free -g | head -2

echo "=== /proc/mounts (no I/O) ==="
grep -vE '^(proc|sysfs|devpts|tmpfs|cgroup|mqueue|overlay|shm) ' /proc/mounts | head -40

echo "=== df, local filesystems only ==="
run 20 "df -hl" df -hl

echo "=== known-good volume ==="
run 20 "df /home/jovyan" df -h /home/jovyan
run 20 "df /tmp" df -h /tmp
run 20 "ls /home/jovyan" bash -c "ls -la /home/jovyan | head -30"
run 20 "write test" bash -c "touch /home/jovyan/.probe_write_test && echo WRITE=OK && rm -f /home/jovyan/.probe_write_test || echo WRITE=FAIL"

echo "=== hmoe-cloud campaign state ==="
run 20 "top level" ls -la "$ROOT"
run 90 "du depth 1" du -sh --max-depth=1 "$ROOT"
run 30 "nvcc" bash -c "'$ROOT/cuda-12.1.1/bin/nvcc' --version | tail -3"
run 20 "wheels" bash -c "ls -la '$ROOT/wheels' | head"
run 20 "logs" bash -c "ls -la '$ROOT/logs' | tail -12"
run 60 "torch in te-venv" bash -c "'$ROOT/te-venv/bin/python' -c \"import torch; print('torch', torch.__version__, torch.version.cuda)\""
run 30 "nvidia lib dirs in venv" bash -c "ls -d '$ROOT'/te-venv/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null"
run 20 "nvjitlink" bash -c "ls -l '$ROOT'/te-venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib 2>/dev/null || echo NO_NVJITLINK_WHEEL"

echo "=== done ==="
exit 0
