#!/usr/bin/env bash
# r4a watcher：等笔记本 R4-a 构建完成（判据：日志含 "[r4a] DONE" 且大小稳定；含 mtime 停更报警，10-26 事故教训），
# 然后回传模型（md5 校验，失败重试 ≤3），拉起 r4a-eval。
set -u
PROJ=/media/pi/DATA/rknn-llm/nac_paper
MODEL=$PROJ/models/local_w4a16_r4e.rkllm
LOG=$PROJ/results/r4a_watch.log
SSH="sshpass -p 131468 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 Administrator@192.168.3.16"
W() { $SSH "wsl -d Ubuntu -- bash -c \"$1\"" 2>/dev/null | tr -d '\r\000'; }
ts() { date '+%F %T'; }

echo "[$(ts)] r4a-wd start" >> "$LOG"
grep -qF 'R4A X86 ALL DONE' "$LOG" 2>/dev/null && { echo "[$(ts)] already done, exit" >> "$LOG"; exit 0; }

# 等构建完成：DONE 标记，或日志大小停更 40min（报警但不中止——蒸发场景人工介入）
last_sz=0; stable=0; done_seen=0
while :; do
    sz=$(W "stat -c%s /home/dministrator/r3x86/build_r4a.log 2>/dev/null || echo 0")
    tail1=$(W "tail -c 2000 /home/dministrator/r3x86/build_r4a.log 2>/dev/null | tr -d '\000' | tail -1")
    case "$tail1" in *"[r4a] DONE"*) done_seen=1; break;; esac
    if [ "$sz" = "$last_sz" ] && [ "$sz" != "0" ]; then
        stable=$((stable+1))
        if [ "$stable" -ge 4 ]; then
            echo "[$(ts)] WARN: build log stalled at $sz B for ~40min without DONE (proc check below)" >> "$LOG"
            nproc=$(W "ps aux | grep r4a_build | grep -v grep | wc -l")
            echo "[$(ts)] r4a_build procs=$nproc" >> "$LOG"
            [ "$nproc" = "0" ] && { echo "[$(ts)] FATAL: build process gone, no DONE. ABORT (人工检查 build_r4a.log)" >> "$LOG"; exit 1; }
        fi
    else
        stable=0
    fi
    last_sz=$sz
    sleep 600
done

echo "[$(ts)] build DONE detected, transfer model" >> "$LOG"

# 回传：WSL cp 到 Windows 中转 → 板上 sftp 拉回（scp 拉方向确定性截断 204800B，10-27 实锤）→ md5 校验，重试 3 次
for i in 1 2 3; do
    W "cp /home/dministrator/r3x86/models/local_w4a16_r4e.rkllm /mnt/c/Users/Administrator/r4a_model.rkllm && sync && ls -la /mnt/c/Users/Administrator/r4a_model.rkllm" >> "$LOG" 2>&1
    sleep 10
    rm -f "$MODEL"
    sshpass -p '131468' sftp -o StrictHostKeyChecking=no 'Administrator@192.168.3.16' <<< "get r4a_model.rkllm $MODEL" >> "$LOG" 2>&1
    sleep 5
    md5_src=$(W "md5sum /home/dministrator/r3x86/models/local_w4a16_r4e.rkllm | cut -d' ' -f1")
    md5_dst=$(md5sum "$MODEL" 2>/dev/null | cut -d' ' -f1)
    echo "[$(ts)] try$i md5 src=$md5_src dst=$md5_dst" >> "$LOG"
    if [ -n "$md5_src" ] && [ "$md5_src" = "$md5_dst" ]; then
        echo "[$(ts)] model transfer OK ($(du -h "$MODEL" | cut -f1))" >> "$LOG"
        TRANSFER_OK=1
        break
    fi
done
[ "${TRANSFER_OK:-0}" = "1" ] || { echo "[$(ts)] FATAL: md5 mismatch after 3 tries, ABORT" >> "$LOG"; exit 1; }

if systemd-run --user --unit=r4e-eval --working-directory="$PROJ" \
        bash "$PROJ/scripts/r4e_pipeline.sh" >> "$LOG" 2>&1; then
    echo "[$(ts)] r4e-eval STARTED" >> "$LOG"
else
    echo "[$(ts)] r4a-eval skipped (already active/exists)" >> "$LOG"
fi
