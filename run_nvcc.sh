#!/bin/bash
# Точка входа для nvcc-пробы: пишет полный лог на диск и всегда выходит с 0.
export PYTHONUNBUFFERED=1
export PYTHONUSERBASE=/home/jovyan/.local
export PATH=$PYTHONUSERBASE/bin:$PATH

mkdir -p /home/jovyan/logs
LOG=/home/jovyan/logs/nvcc_$(date +%F_%H%M%S).log

python nvcc_probe.py "$@" > "$LOG" 2>&1
CODE=$?

echo "EXIT=$CODE"
echo "LOG=$LOG"
echo "=== полный вывод nvcc-пробы ==="
cat "$LOG"
exit 0
