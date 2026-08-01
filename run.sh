#!/bin/bash
# Точка входа для mlsub. Пишет полный лог на сетевой диск и всегда завершается
# с кодом 0 — иначе платформа спрячет вывод и статус Failed придёт без логов.
export PYTHONUNBUFFERED=1

mkdir -p /home/jovyan/logs
LOG=/home/jovyan/logs/probe_$(date +%F_%H%M%S).log

python probe.py "$@" > "$LOG" 2>&1
CODE=$?

echo "EXIT=$CODE"
echo "LOG=$LOG"
echo "=== полный вывод пробы ==="
cat "$LOG"
exit 0
