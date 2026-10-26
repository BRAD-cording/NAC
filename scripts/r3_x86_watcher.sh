#!/usr/bin/env bash
# r3-x86 watcher：等笔记本 x86 原生构建的模型 scp 落盘（>1G 即视为传完），
# 然后停掉 qemu 兜底构建（r3-plain），拉起板端评测链 r3-x86-eval。
# 抗重启：由 boot_resume.sh 第 8 段重新挂载。
set -u
PROJ=/media/pi/DATA/rknn-llm/nac_paper
MODEL=$PROJ/models/local_w4a16_plain_x86.rkllm
LOG=$PROJ/results/r3_x86_pipeline.log
ts() { date '+%F %T'; }

echo "[$(ts)] r3-x86-wd start (waiting for x86 model)" >> "$LOG"

# 已完成 → 直接退出，不留守
grep -qF 'R3 X86 ALL DONE' "$LOG" 2>/dev/null && { echo "[$(ts)] x86 eval already done, exit" >> "$LOG"; exit 0; }

# 等模型落盘：>1G 且大小连续 3 轮（约 3min）不再增长（scp 传完前文件持续增长；
# 10-26 事故：1.41GB 模型传到 1GB 即被误判传完，评测拉起读半截模型全灭）
last=0; stable=0
while :; do
    sz=$(stat -c%s "$MODEL" 2>/dev/null || echo 0)
    if [ "$sz" -gt 1000000000 ] && [ "$sz" -eq "$last" ]; then
        stable=$((stable+1))
        [ "$stable" -ge 3 ] && break
    else
        stable=0
    fi
    last=$sz
    sleep 60
done

echo "[$(ts)] x86 model detected ($(du -h "$MODEL" | cut -f1)), stop qemu r3-plain" >> "$LOG"
systemctl --user stop r3-plain 2>/dev/null || true
systemctl --user reset-failed r3-plain 2>/dev/null || true

if systemd-run --user --unit=r3-x86-eval --working-directory="$PROJ" \
        bash "$PROJ/scripts/r3_x86_pipeline.sh" >> "$LOG" 2>&1; then
    echo "[$(ts)] r3-x86-eval STARTED" >> "$LOG"
else
    echo "[$(ts)] r3-x86-eval skipped (already active/exists)" >> "$LOG"
fi
