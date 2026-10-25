#!/usr/bin/env bash
# R3-x86 板端评测链：smoke → GSM8K zs 200 → MMLU zs 969（模型由笔记本 x86 原生构建回传）。
# 判据（outline §13 R3）：GSM8K ≈0.555 → g64 是元凶；≈0.005 → 本地构建配置问题。
# 幂等：各阶段产物已存在则跳过；全部完成打 R3 X86 ALL DONE 标记。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
LOG=results/r3_x86_pipeline.log
exec >> "$LOG" 2>&1

PY="$PWD/.venv/bin/python"
MODEL="models/local_w4a16_plain_x86.rkllm"
GSM_JSON=results/acc_gsm8k_qwen1.5b_w4a16plain_x86_zs.json
MMLU_JSON=results/acc_mmlu_qwen1.5b_w4a16plain_x86_zs.json

echo "=== R3 x86 eval pipeline start $(date '+%F %T') ==="

# 模型完整性守卫
[ -s "$MODEL" ] && [ "$(stat -c%s "$MODEL")" -gt 1000000000 ] \
    || { echo "!!! x86 model missing/incomplete, ABORT"; exit 1; }

# ---------- 阶段 B：板端冒烟（加载 + 一次生成） ----------
echo "--- [B] smoke ---"
"$PY" scripts/smoke_test.py --model "$MODEL" > results/r3_x86_smoke.log 2>&1
grep -q "PASS" results/r3_x86_smoke.log \
    && echo "=== [B] smoke PASS $(date '+%F %T') ===" \
    || echo "!!! [B] smoke 未见 PASS（继续评测，需复查 results/r3_x86_smoke.log）"

# ---------- 阶段 C：板端 GSM8K zs 200（核心判别，协议同基线 zs 直答） ----------
if [ -s "$GSM_JSON" ]; then
    echo "--- [C] GSM8K result exists, skip ---"
else
    echo "--- [C] GSM8K zs 200 start $(date '+%F %T') ---"
    HF_HUB_OFFLINE=1 "$PY" -m src.eval.accuracy --model "$MODEL" \
        --task gsm8k --num-samples 200 --fewshot-num 0 \
        --out "$GSM_JSON" > results/acc_w4a16plain_x86_gsmzs.log 2>&1
    echo "=== [C] GSM8K done $(date '+%F %T') ==="
fi

# ---------- 阶段 D：板端 MMLU zs 969（完整对照，协议同基线） ----------
if [ -s "$MMLU_JSON" ]; then
    echo "--- [D] MMLU result exists, skip ---"
else
    echo "--- [D] MMLU zs 969 start $(date '+%F %T') ---"
    HF_HUB_OFFLINE=1 "$PY" -m src.eval.accuracy --model "$MODEL" \
        --task mmlu --num-samples 969 --fewshot-num 0 \
        --out "$MMLU_JSON" > results/acc_w4a16plain_x86_mmluzs.log 2>&1
    echo "=== [D] MMLU done $(date '+%F %T') ==="
fi

echo "=== R3 X86 ALL DONE $(date '+%F %T') ==="
