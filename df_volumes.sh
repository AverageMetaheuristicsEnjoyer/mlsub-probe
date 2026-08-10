#!/usr/bin/env bash
# Ephemeral: free space and write access on the two extra NFS exports.
for d in /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
    echo "--- $d"
    timeout -s KILL 20 df -h "$d" 2>&1
    timeout -s KILL 20 bash -c "ls -la '$d' | head -15" 2>&1
    timeout -s KILL 20 bash -c "touch '$d/.probe_w' && echo WRITE=OK && rm -f '$d/.probe_w' || echo WRITE=FAIL" 2>&1
done
echo "=== done ==="
exit 0
