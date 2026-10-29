#!/bin/bash
# 等待 q1 板端 512 队列完成，自动生成汇总并提交（无需人工值守）
set -u
cd /media/pi/DATA/rknn-llm/nac_paper || exit 1
MARK=results/q1_board512.DONE
LOG=results/q1_postprocess.log

echo "[$(date '+%F %T')] watcher start" >> "$LOG"
while [ ! -f "$MARK" ]; do sleep 60; done
echo "[$(date '+%F %T')] DONE marker found" >> "$LOG"

.venv/bin/python scripts/q1_report.py >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] report rc=$rc" >> "$LOG"

git add -A results scripts analysis paper >> "$LOG" 2>&1
git -c user.name=pi -c user.email=pi@local commit -q -m "auto: board 512-token results collected (q1_report) + summaries" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] committed" >> "$LOG"
touch results/q1_postprocess.DONE
