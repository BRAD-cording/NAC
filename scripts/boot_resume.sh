#!/usr/bin/env bash
# 开机自动恢复：整机重启后若 P0 实验未全部完成，自动拉起 nac-queue 队列与 mmctrl-wd watcher。
# 设计约束：
#   - 幂等：单元已存在/已完成则跳过，不重复拉起、不重跑已完成阶段
#   - 队列各阶段本身支持断点续跑（MMLU/GSM8K sim 按 records 续跑，GPTQ g128 校准按产物文件跳过）
#   - E3 完成后不再拉起 watcher（watcher 的触发标记在累积日志中永久存在，需此守卫防重放 E3）
#   - 由 user 级 timer（OnBootSec）触发，不阻塞登录/启动链
set -u
PROJ=/media/pi/DATA/rknn-llm/nac_paper
LOG=$PROJ/results/boot_resume.log
ts() { date '+%F %T'; }

echo "[$(ts)] boot resume check begin" >> "$LOG"

# 1) 等 DATA 挂载就绪（最多 5 分钟）
for _ in $(seq 1 60); do
    [ -d "$PROJ/results" ] && break
    sleep 5
done
if [ ! -d "$PROJ/results" ]; then
    echo "[$(ts)] ABORT: $PROJ/results not mounted" >> "$LOG"
    exit 1
fi

# 2) 等 30s 让系统开机峰值（fsck/索引等）过去
sleep 30

# 3) 队列全部完成 → 跳过队列/watcher 拉起，但仍考虑 NAC-v2 流水线
QUEUE_DONE=0
if grep -qF 'queue ALL DONE' "$PROJ/results/p0_queue.log" 2>/dev/null; then
    echo "[$(ts)] queue ALL DONE already, queue/mmctrl resume skipped" >> "$LOG"
    QUEUE_DONE=1
fi

systemctl --user reset-failed nac-queue nac-p0 mmctrl-wd nac-v2-wd 2>/dev/null || true

# 4) 拉起 P0 队列（断点续跑；已 active 时 systemd-run 报错 → 记为 skipped）
if [ "$QUEUE_DONE" -eq 0 ]; then
    if systemd-run --user --unit=nac-queue --working-directory="$PROJ" \
            bash "$PROJ/scripts/run_p0_queue.sh" >> "$LOG" 2>&1; then
        echo "[$(ts)] nac-queue STARTED" >> "$LOG"
    else
        echo "[$(ts)] nac-queue skipped (already active/exists)" >> "$LOG"
    fi
fi

# 5) E3 未完成才挂 watcher（防止累积日志中的 DONE 标记导致 E3 重放）
if [ "$QUEUE_DONE" -eq 0 ]; then
    if grep -qF 'mmctrl ALL DONE' "$PROJ/results/mmctrl_pipeline.log" 2>/dev/null; then
        echo "[$(ts)] mmctrl E3 already done, watcher not needed" >> "$LOG"
    else
        if systemd-run --user --unit=mmctrl-wd \
                bash "$PROJ/scripts/mmctrl_watcher.sh" >> "$LOG" 2>&1; then
            echo "[$(ts)] mmctrl-wd STARTED" >> "$LOG"
        else
            echo "[$(ts)] mmctrl-wd skipped (already active/exists)" >> "$LOG"
        fi
    fi
fi

# 6) NAC-v2 流水线未完成才挂 watcher（守卫：最终 MMLU v2 结果落盘则跳过）
if [ -f "$PROJ/results/acc_mmlu_sim_g64_nacv2.json" ]; then
    echo "[$(ts)] nacv2 already done, watcher not needed" >> "$LOG"
else
    if systemd-run --user --unit=nac-v2-wd \
            bash "$PROJ/scripts/nac_v2_watcher.sh" >> "$LOG" 2>&1; then
        echo "[$(ts)] nac-v2-wd STARTED" >> "$LOG"
    else
        echo "[$(ts)] nac-v2-wd skipped (already active/exists)" >> "$LOG"
    fi
fi

# 7) R3 本地 w4a16 原生构建复现基线（幂等：R3 DONE 或结果落盘则跳过，未完成拉起续跑）
#    x86 模型已回传则 qemu 兜底构建不再有意义，跳过
if grep -qF 'R3 ALL DONE' "$PROJ/results/r3_plain_pipeline.log" 2>/dev/null; then
    echo "[$(ts)] R3 already done, resume skipped" >> "$LOG"
elif [ -f "$PROJ/results/acc_gsm8k_qwen1.5b_w4a16plain_zs.json" ]; then
    echo "[$(ts)] R3 result exists, resume skipped" >> "$LOG"
elif [ -f "$PROJ/models/local_w4a16_plain_x86.rkllm" ]; then
    echo "[$(ts)] x86 model present, qemu r3-plain resume skipped" >> "$LOG"
else
    if systemd-run --user --unit=r3-plain --working-directory="$PROJ" \
            bash "$PROJ/scripts/r3_plain_pipeline.sh" >> "$LOG" 2>&1; then
        echo "[$(ts)] r3-plain STARTED" >> "$LOG"
    else
        echo "[$(ts)] r3-plain skipped (already active/exists)" >> "$LOG"
    fi
fi

# 8) R3-x86：模型已回传但评测未完 → 直接拉评测；未回传 → 挂 watcher 等落盘
if [ -f "$PROJ/results/acc_mmlu_qwen1.5b_w4a16plain_x86_zs.json" ]; then
    echo "[$(ts)] r3-x86 eval already done, skipped" >> "$LOG"
elif [ -f "$PROJ/models/local_w4a16_plain_x86.rkllm" ]; then
    if systemd-run --user --unit=r3-x86-eval --working-directory="$PROJ" \
            bash "$PROJ/scripts/r3_x86_pipeline.sh" >> "$LOG" 2>&1; then
        echo "[$(ts)] r3-x86-eval STARTED" >> "$LOG"
    else
        echo "[$(ts)] r3-x86-eval skipped (already active/exists)" >> "$LOG"
    fi
else
    if systemd-run --user --unit=r3-x86-wd \
            bash "$PROJ/scripts/r3_x86_watcher.sh" >> "$LOG" 2>&1; then
        echo "[$(ts)] r3-x86-wd STARTED" >> "$LOG"
    else
        echo "[$(ts)] r3-x86-wd skipped (already active/exists)" >> "$LOG"
    fi
fi

echo "[$(ts)] boot resume check end" >> "$LOG"
