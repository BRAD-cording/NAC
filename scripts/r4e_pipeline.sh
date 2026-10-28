#!/usr/bin/env bash
# R4-e 板端评测链：smoke → GSM8K zs 200 → MMLU zs 969。
# R4-e = 与 R3 唯一变量 quantized_algorithm normal→grq（B2 其他形态；R4-d gdq 已证伪：0.01/0.5387）。
# 判据：GSM8K 恢复（≈0.5+）→ vendor 算法=grq（B2 证实）；仍 ≈0.0x → B2 主形态整族出清
#       （normal/gdq/grq 全崩），转 target_platform/B4 版本/二相行为归因。
# 幂等：各阶段产物已存在则跳过；全部完成打 R4E X86 ALL DONE 标记。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
LOG=results/r4e_pipeline.log
exec >> "$LOG" 2>&1

PY="$PWD/.venv/bin/python"
MODEL="models/local_w4a16_r4e.rkllm"
GSM_JSON=results/acc_gsm8k_qwen1.5b_r4e_zs.json
MMLU_JSON=results/acc_mmlu_qwen1.5b_r4e_zs.json

echo "=== R4E x86 eval pipeline start $(date '+%F %T') ==="

[ -s "$MODEL" ] && [ "$(stat -c%s "$MODEL")" -gt 1000000000 ] \
    || { echo "!!! r4e model missing/incomplete, ABORT"; exit 1; }

echo "--- [B] smoke ---"
"$PY" scripts/smoke_test.py --model "$MODEL" > results/r4e_smoke.log 2>&1
grep -q "PASS" results/r4e_smoke.log \
    && echo "=== [B] smoke PASS $(date '+%F %T') ===" \
    || echo "!!! [B] smoke 未见 PASS（继续评测，需复查 results/r4e_smoke.log）"

if [ -s "$GSM_JSON" ]; then
    echo "--- [C] GSM8K result exists, skip ---"
else
    echo "--- [C] GSM8K zs 200 start $(date '+%F %T') ---"
    HF_HUB_OFFLINE=1 "$PY" -m src.eval.accuracy --model "$MODEL" \
        --task gsm8k --num-samples 200 --fewshot-num 0 \
        --out "$GSM_JSON" > results/acc_r4e_gsmzs.log 2>&1
    echo "=== [C] GSM8K done $(date '+%F %T') ==="
fi

if [ -s "$MMLU_JSON" ]; then
    echo "--- [D] MMLU result exists, skip ---"
else
    echo "--- [D] MMLU zs 969 start $(date '+%F %T') ---"
    HF_HUB_OFFLINE=1 "$PY" -m src.eval.accuracy --model "$MODEL" \
        --task mmlu --num-samples 969 --fewshot-num 0 \
        --out "$MMLU_JSON" > results/acc_r4e_mmluzs.log 2>&1
    echo "=== [D] MMLU done $(date '+%F %T') ==="
fi

echo "=== R4E X86 ALL DONE $(date '+%F %T') ==="
