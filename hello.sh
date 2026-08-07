#!/usr/bin/env bash
# Control: touches no filesystem beyond the cloned repo. If this hangs, the platform
# is at fault, not our probes.
echo "HELLO"
hostname
date -u
nproc
echo "DONE"
exit 0
