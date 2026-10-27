#!/usr/bin/env bash
# R4-c 板端评测链：smoke → GSM8K zs 200 → MMLU zs 969。
# R4-c = 与 R3 唯一变量 num_npu_core 2→1（toolkit 默认；数据集同 R3 原版 128 条 wikitext；docs/r4_bisect_plan.md B7）。
# 判据：GSM8K 恢复（≈0.5+）→ npu_core=2 是元凶（B7 证实）；仍 ≈0.0x → npu_core 排除，转 B2（algorithm）/target_platform。
# 幂等：各阶段产物已存在则跳过；全部完成打 R4C X86 ALL DONE 标记。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
LOG=results/r4c_pipeline.log
exec >> "$LOG" 2>&1

PY="$PWD/.venv/bin/python"
MODEL="models/local_w4a16_r4c.rkllm"
GSM_JSON=results/acc_gsm8k_qwen1.5b_r4c_zs.json
MMLU_JSON=results/acc_mmlu_qwen1.5b_r4c_zs.json

echo "=== R4C x86 eval pipeline start $(date '+%F %T') ==="

[ -s "$MODEL" ] && [ "$(stat -c%s "$MODEL")" -gt 1000000000 ] \
    || { echo "!!! r4c model missing/incomplete, ABORT"; exit 1; }

echo "--- [B] smoke ---"
"$PY" scripts/smoke_test.py --model "$MODEL" > results/r4c_smoke.log 2>&1
grep -q "PASS" results/r4c_smoke.log \
    && echo "=== [B] smoke PASS $(date '+%F %T') ===" \
    || echo "!!! [B] smoke 未见 PASS（继续评测，需复查 results/r4c_smoke.log）"

if [ -s "$GSM_JSON" ]; then
    echo "--- [C] GSM8K result exists, skip ---"
else
    echo "--- [C] GSM8K zs 200 start $(date '+%F %T') ---"
    HF_HUB_OFFLINE=1 "$PY" -m src.eval.accuracy --model "$MODEL" \
        --task gsm8k --num-samples 200 --fewshot-num 0 \
        --out "$GSM_JSON" > results/acc_r4c_gsmzs.log 2>&1
    echo "=== [C] GSM8K done $(date '+%F %T') ==="
fi

if [ -s "$MMLU_JSON" ]; then
    echo "--- [D] MMLU result exists, skip ---"
else
    echo "--- [D] MMLU zs 969 start $(date '+%F %T') ---"
    HF_HUB_OFFLINE=1 "$PY" -m src.eval.accuracy --model "$MODEL" \
        --task mmlu --num-samples 969 --fewshot-num 0 \
        --out "$MMLU_JSON" > results/acc_r4c_mmluzs.log 2>&1
    echo "=== [D] MMLU done $(date '+%F %T') ==="
fi

echo "=== R4C X86 ALL DONE $(date '+%F %T') ==="
