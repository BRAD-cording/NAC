#!/usr/bin/env bash
# P0：板端 int-zp 注入复评（NAC-v2 qparams，验证问题 B 修复）
#
# 依据：E3 对照（minmax float-zp，同一 build 参数/同一评测协议）板端 GSM8K 0.005。
# 本链把 nacv2（整值 zp）注入同一 w4a16_g64 构建流程，板端跑 MMLU 969 + GSM8K 200。
# 预期读法（大纲 §10 P0）：
#   - GSM8K 显著高于 0.005 → 整值 zp 修复有效（板端叙事可升级为"诊断+修复"）
#   - GSM8K 仍 ≈0.005     → 整值 zp 非充分条件，问题 B 判定降级
#
# 用法（必须带 --working-directory）：
#   systemd-run --user --unit=p0-intzp --working-directory="$PWD" \
#     bash scripts/p0_intzp_pipeline.sh
#
# 幂等：产物已存在则跳过对应阶段。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
LOG=results/p0_intzp_pipeline.log
exec >> "$LOG" 2>&1

RKT="${RKT_BASE:-/media/pi/DATA/rktoolkit}"
PY="$PWD/.venv/bin/python"
X86PY="$RKT/x86py/bin/python3"
QPARAMS="results/nacv2_g64_intzp.qparams"
MODEL_OUT="models/nacv2_qwen1.5b_g64_intzp.rkllm"

echo "=== P0 int-zp pipeline start $(date '+%F %T') ==="

# ---------- 阶段 0：qparams 转换 + 保真度审计 ----------
if [ ! -s "$QPARAMS" ]; then
    echo "--- [0] convert nacv2 qparams -> extra_qparams schema ---"
    "$PY" scripts/gen_nacv2_extra_qparams.py || { echo "!!! convert FAILED"; exit 1; }
else
    echo "--- [0] qparams exists, skip convert ---"
fi

# ---------- 阶段 A：qemu 注入构建（复用 E3 模板，仅换 qparams/产物名） ----------
if [ -s "$MODEL_OUT" ] && [ "$(stat -c%s "$MODEL_OUT")" -gt 1000000000 ]; then
    echo "--- [A] model exists ($(du -h "$MODEL_OUT" | cut -f1)), skip build ---"
else
    free_g=$(df -BG --output=avail /media/pi | tail -1 | tr -dc '0-9')
    echo "--- [A] build start, free=${free_g}G ---"
    [ "${free_g:-0}" -ge 4 ] || { echo "!!! 磁盘不足 4G，中止"; exit 1; }
    timeout 5400 env QEMU_LD_PREFIX="$RKT/sysroot" taskset -c 0-3 "$X86PY" - <<PY
from rkllm.api import rkllm_base as rb
print("[p0] === int-zp injection build start ===", flush=True)
llm = rb.RKLLMBase()
ret = llm.load_huggingface("models/Qwen2.5-1.5B-Instruct-hf", None, "cpu", "float32", None, True)
print(f"[p0] load ret={ret}", flush=True)
ret = llm.build(do_quantization=True, quantized_dtype="w4a16_g64",
                quantized_algorithm="normal", optimization_level=1,
                target_platform="rk3576", num_npu_core=2,
                dataset="data/quant_dataset.json",
                extra_qparams="$QPARAMS")
print(f"[p0] build ret={ret}", flush=True)
if ret == 0:
    ret = llm.export_rkllm("$MODEL_OUT")
    print(f"[p0] export ret={ret}", flush=True)
print("[p0] BUILD_EXIT", flush=True)
PY
    if [ ! -s "$MODEL_OUT" ]; then
        echo "!!! build FAILED $(date '+%F %T')"; exit 1
    fi
    echo "=== [A] build OK $(du -h "$MODEL_OUT" | cut -f1) $(date '+%F %T') ==="
fi

# ---------- 阶段 B：板端冒烟（加载 + 一次生成） ----------
echo "--- [B] smoke ---"
"$PY" scripts/smoke_test.py --model "$MODEL_OUT" > results/p0_intzp_smoke.log 2>&1
grep -q "PASS" results/p0_intzp_smoke.log \
    && echo "=== [B] smoke PASS $(date '+%F %T') ===" \
    || echo "!!! [B] smoke 未见 PASS（继续评测，但需复查 results/p0_intzp_smoke.log）"

# ---------- 阶段 C：板端 MMLU zs 969（协议同 E3：zs / 32 tok） ----------
echo "--- [C] MMLU zs 969 start $(date '+%F %T') ---"
HF_HUB_OFFLINE=1 "$PY" -m src.eval.accuracy --model "$MODEL_OUT" \
    --task mmlu --num-samples 969 --fewshot-num 0 \
    --out results/acc_mmlu_qwen1.5b_g64_nacv2intzp_zs.json > results/acc_g64nacv2intzp_mmluzs.log 2>&1
echo "=== [C] MMLU done $(date '+%F %T') ==="

# ---------- 阶段 D：板端 GSM8K zs 200（协议同 E3：zs 直答 / 32 tok） ----------
echo "--- [D] GSM8K zs 200 start $(date '+%F %T') ---"
HF_HUB_OFFLINE=1 "$PY" -m src.eval.accuracy --model "$MODEL_OUT" \
    --task gsm8k --num-samples 200 --fewshot-num 0 \
    --out results/acc_gsm8k_qwen1.5b_g64_nacv2intzp_zs.json > results/acc_g64nacv2intzp_gsmzs.log 2>&1
echo "=== [D] GSM8K done $(date '+%F %T') ==="

echo "=== P0 ALL DONE $(date '+%F %T') ==="
