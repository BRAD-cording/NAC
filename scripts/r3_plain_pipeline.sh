#!/usr/bin/env bash
# R3：本地 w4a16 原生构建（无 extra_qparams，toolkit 自算）复现厂商基线 0.555。
#
# 判据（outline §13 R3）：
#   GSM8K ≈0.555 → 问题在 g64（本地 w4a16 正常，g64 是元凶）
#   GSM8K ≈0.005 → 问题在"本地构建 vs 厂商预转"的构建配置（dataset/optimization/版本）
#
# 与 E3/P0 的区别：本链**不传 extra_qparams**，dtype=w4a16（非 g64），
# 对标厂商预转基线 Qwen2.5-1.5B-Instruct_RK3576_w4a16.rkllm（GSM8K 0.555）。
# ⚠️ 2026-10-25 实测：qemu 下 Optimizing 从未在之前构建中运行过（extra_qparams 命中即跳过），
# 1.5B 无注入构建 Optimizing 0/28 跑满 3h timeout 仍未出第一个 pass（CPU 满负荷、无 OOM，纯慢非卡死）。
# 故 timeout 放大到 12h，并加内存/耗时采样（results/r3_mem_trace.log）。
#
# 用法：
#   systemd-run --user --unit=r3-plain --working-directory=/media/pi/DATA/rknn-llm/nac_paper \
#     bash /media/pi/DATA/rknn-llm/nac_paper/scripts/r3_plain_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
LOG=results/r3_plain_pipeline.log
exec >> "$LOG" 2>&1

RKT="${RKT_BASE:-/media/pi/DATA/rktoolkit}"
PY="$PWD/.venv/bin/python"
X86PY="$RKT/x86py/bin/python3"
MODEL_OUT="models/local_w4a16_plain.rkllm"

echo "=== R3 plain w4a16 pipeline start $(date '+%F %T') ==="

# ---------- 阶段 A：无注入 w4a16 原生构建（toolkit 自算，含 Optimizing） ----------
if [ -s "$MODEL_OUT" ] && [ "$(stat -c%s "$MODEL_OUT")" -gt 1000000000 ]; then
    echo "--- [A] model exists ($(du -h "$MODEL_OUT" | cut -f1)), skip build ---"
else
    free_g=$(df -BG --output=avail /media/pi | tail -1 | tr -dc '0-9')
    echo "--- [A] build start, free=${free_g}G ---"
    [ "${free_g:-0}" -ge 4 ] || { echo "!!! 磁盘不足 4G，中止"; exit 1; }
    # 采样器：每 10 分钟记录 qemu python 进程 RSS/已运行时长，直至模型落盘
    ( while [ ! -s "$MODEL_OUT" ]; do
          sleep 600
          ps -eo rss,etime,args | grep 'x86py/bin/python3' | grep -v grep | head -1 >> results/r3_mem_trace.log
      done ) &
    timeout 43200 env QEMU_LD_PREFIX="$RKT/sysroot" taskset -c 0-3 "$X86PY" scripts/r3_plain_build.py
    if [ ! -s "$MODEL_OUT" ]; then
        echo "!!! build FAILED $(date '+%F %T')"; exit 1
    fi
    echo "=== [A] build OK $(du -h "$MODEL_OUT" | cut -f1) $(date '+%F %T') ==="
fi

# ---------- 阶段 B：板端冒烟（加载 + 一次生成） ----------
echo "--- [B] smoke ---"
"$PY" scripts/smoke_test.py --model "$MODEL_OUT" > results/r3_plain_smoke.log 2>&1
grep -q "PASS" results/r3_plain_smoke.log \
    && echo "=== [B] smoke PASS $(date '+%F %T') ===" \
    || echo "!!! [B] smoke 未见 PASS（继续评测，需复查 results/r3_plain_smoke.log）"

# ---------- 阶段 C：板端 GSM8K zs 200（核心判别，协议同基线 zs 直答） ----------
echo "--- [C] GSM8K zs 200 start $(date '+%F %T') ---"
HF_HUB_OFFLINE=1 "$PY" -m src.eval.accuracy --model "$MODEL_OUT" \
    --task gsm8k --num-samples 200 --fewshot-num 0 \
    --out results/acc_gsm8k_qwen1.5b_w4a16plain_zs.json > results/acc_w4a16plain_gsmzs.log 2>&1
echo "=== [C] GSM8K done $(date '+%F %T') ==="

# ---------- 阶段 D：板端 MMLU zs 969（完整对照，协议同基线） ----------
echo "--- [D] MMLU zs 969 start $(date '+%F %T') ---"
HF_HUB_OFFLINE=1 "$PY" -m src.eval.accuracy --model "$MODEL_OUT" \
    --task mmlu --num-samples 969 --fewshot-num 0 \
    --out results/acc_mmlu_qwen1.5b_w4a16plain_zs.json > results/acc_w4a16plain_mmluzs.log 2>&1
echo "=== [D] MMLU done $(date '+%F %T') ==="

echo "=== R3 ALL DONE $(date '+%F %T') ==="
