#!/bin/bash
# 等 sim 重跑结束 -> 生成汇总 -> 提交
set -u
cd /media/pi/DATA/rknn-llm/nac_paper || exit 1
LOG=results/q1_sim_postprocess.log
echo "[$(date '+%F %T')] sim watcher start" >> "$LOG"
while [ ! -f results/q1_sim.DONE ]; do sleep 60; done
echo "[$(date '+%F %T')] sim DONE" >> "$LOG"
.venv/bin/python scripts/q1_sim_report.py >> "$LOG" 2>&1
git add -A results >> "$LOG" 2>&1
git -c user.name=pi -c user.email=pi@local commit -q -m "auto: sim rerun (stop-truncated) results + summary" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] committed" >> "$LOG"
