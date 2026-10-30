#!/bin/bash
# 等 GPTQ 重跑完成 -> 生成截断协议对比 -> 提交（用于判定 Finding 1 是否需重写）
set -u
cd /media/pi/DATA/rknn-llm/nac_paper || exit 1
LOG=results/q1_gptq_report.log
echo "[$(date '+%F %T')] watcher start" >> "$LOG"
while [ ! -f results/acc_gsm8k_sim_g64_stop_gptq.json ]; do sleep 60; done
# 等对应进程退出，确保文件写完
while pgrep -f 'configs gptq' > /dev/null 2>&1; do sleep 30; done
echo "[$(date '+%F %T')] gptq done" >> "$LOG"
.venv/bin/python analysis/sim_compare.py >> "$LOG" 2>&1
git add -A results/acc_gsm8k_sim_g64_stop_gptq.json results/acc_gsm8k_sim_g64_stop_gptq.records.gptq.json results/q1_sim_compare.md analysis/sim_compare.py >> "$LOG" 2>&1
git -c user.name=pi -c user.email=pi@local commit -q -m "auto: GPTQ truncated-protocol result + fp16 comparison (Finding-1 decision gate)" >> "$LOG" 2>&1
echo "[$(date '+%F %T')] committed" >> "$LOG"
touch results/q1_gptq_report.DONE
