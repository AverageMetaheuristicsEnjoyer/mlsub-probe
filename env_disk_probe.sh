#!/usr/bin/env bash
# Inventory: disk volumes, workspace state, leftovers of the TE build campaign.
# Every step is wrapped in `timeout` — a stale NFS mount must not hang the job.
ROOT=/home/jovyan/hmoe-cloud

run() {   # run <seconds> <label> <command...>
    local t=$1 label=$2
    shift 2
    echo "--- $label"
    timeout "$t" "$@" 2>&1
    local rc=$?
    [ $rc -eq 124 ] && echo "TIMEOUT after ${t}s: $label"
    return 0
}

echo "=== host ==="
hostname
date -u
nproc
free -g | head -2

run 20 "df -h (local only)" df -hl
run 25 "df -h (all, may include NFS)" df -h
run 20 "mount | grep -i nfs" bash -c "mount | grep -iE 'nfs|workspace' | head -20"

for d in /home/jovyan /workspace /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
    run 15 "stat $d" ls -ld "$d"
    run 15 "df $d" df -h "$d"
    run 15 "ls $d" bash -c "ls -la '$d' | head -25"
    run 15 "write test $d" bash -c "touch '$d/.probe_write_test' && echo WRITE=OK && rm -f '$d/.probe_write_test' || echo WRITE=FAIL"
done

echo "=== hmoe-cloud campaign state ==="
run 15 "top level" ls -la "$ROOT"
run 90 "du depth 1" du -sh --max-depth=1 "$ROOT"
run 30 "nvcc" bash -c "'$ROOT/cuda-12.1.1/bin/nvcc' --version | tail -3"
run 15 "wheels" bash -c "ls -la '$ROOT/wheels' | head"
run 15 "logs" bash -c "ls -la '$ROOT/logs' | tail -12"
run 60 "torch in te-venv" bash -c "'$ROOT/te-venv/bin/python' -c \"import torch; print('torch', torch.__version__, torch.version.cuda)\""
run 60 "nvjitlink in venv" bash -c "ls -d '$ROOT'/te-venv/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | head -20"
run 30 "nvjitlink files" bash -c "ls -l '$ROOT'/te-venv/lib/python3.10/site-packages/nvidia/nvjitlink/lib 2>/dev/null"

echo "=== done ==="
exit 0
