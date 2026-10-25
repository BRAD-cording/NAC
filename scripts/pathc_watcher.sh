#!/usr/bin/env bash
# 看门狗：等 nac 评测落满 969 → 暂停队列 → 跑 QuantLinear 捕获探针 → 恢复队列。
# 目的：内存窗口利用（探针需 ~7GB，不能与 7.5GB 的 sim 评测并存）。
set -u
cd /media/pi/DATA/rknn-llm/nac_paper
LOG=results/pathc_watcher.log
echo "[$(date '+%F %T')] watcher armed" >> "$LOG"

while true; do
  n=$(python3 -c "import json;print(len(json.load(open('results/acc_mmlu_sim_g64.records.nac.json'))))" 2>/dev/null || echo 0)
  if [ "${n:-0}" -ge 969 ]; then
    echo "[$(date '+%F %T')] nac eval complete ($n/969), pausing queue" >> "$LOG"
    systemctl --user stop nac-queue 2>/dev/null
    sleep 30
    echo "[$(date '+%F %T')] starting ql capture probe" >> "$LOG"
    timeout 2400 env QEMU_LD_PREFIX=/media/pi/DATA/rktoolkit/sysroot \
      taskset -c 0-3 /media/pi/DATA/rktoolkit/x86py/bin/python3 \
      scripts/ql_capture_probe.py >> results/build_ql_probe.log 2>&1
    echo "[$(date '+%F %T')] probe exit=$?, resuming queue" >> "$LOG"
    systemctl --user start nac-queue 2>/dev/null
    break
  fi
  sleep 300
done
echo "[$(date '+%F %T')] watcher done" >> "$LOG"
