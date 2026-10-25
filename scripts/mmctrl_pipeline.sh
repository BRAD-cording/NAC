#!/usr/bin/env bash
# E3: minmax float-zp 对照注入构建 + 板端评测（须在内存窗口内运行，勿与 sim 评测并行）
set -uo pipefail
cd "$(dirname "$0")/.."
exec >> results/mmctrl_pipeline.log 2>&1
echo "=== mmctrl pipeline start $(date '+%F %T') ==="

# 阶段A: qemu x86 注入构建（复用 inject2 成功模板，仅换 qparams 与产物名）
timeout 5400 env QEMU_LD_PREFIX=/media/pi/DATA/rktoolkit/sysroot taskset -c 0-3 \
/media/pi/DATA/rktoolkit/x86py/bin/python3 - <<'PY'
from rkllm.api import rkllm_base as rb
print("[mmctrl] === minmax-control extra_qparams build start ===", flush=True)
llm = rb.RKLLMBase()
ret = llm.load_huggingface("models/Qwen2.5-1.5B-Instruct-hf", None, "cpu", "float32", None, True)
print(f"[mmctrl] load ret={ret}", flush=True)
ret = llm.build(do_quantization=True, quantized_dtype="w4a16_g64",
                quantized_algorithm="normal", optimization_level=1,
                target_platform="rk3576", num_npu_core=2,
                dataset="data/quant_dataset.json",
                extra_qparams="/media/pi/DATA/rknn-llm/nac_paper/results/minmax_g64_full.qparams")
print(f"[mmctrl] build ret={ret}", flush=True)
if ret == 0:
    ret = llm.export_rkllm("models/nac_qwen1.5b_g64_mmctrl.rkllm")
    print(f"[mmctrl] export ret={ret}", flush=True)
print("[mmctrl] BUILD_EXIT", flush=True)
PY
if [ ! -f models/nac_qwen1.5b_g64_mmctrl.rkllm ]; then
    echo "!!! mmctrl build FAILED, abort board eval $(date '+%F %T')"
    exit 1
fi
echo "=== mmctrl build OK $(date '+%F %T') ==="

# 阶段B: 板端 MMLU zs (969)
.venv/bin/python -m src.eval.accuracy --model models/nac_qwen1.5b_g64_mmctrl.rkllm \
    --task mmlu --num-samples 969 --fewshot-num 0 \
    --out results/acc_mmlu_qwen1.5b_g64mmctrl_zs.json >> results/acc_g64mmctrl_mmluzs.log 2>&1
echo "=== mmctrl MMLU zs DONE $(date '+%F %T') ==="

# 阶段C: 板端 GSM8K zs (200)
.venv/bin/python -m src.eval.accuracy --model models/nac_qwen1.5b_g64_mmctrl.rkllm \
    --task gsm8k --num-samples 200 --fewshot-num 0 \
    --out results/acc_gsm8k_qwen1.5b_g64mmctrl_zs.json >> results/acc_g64mmctrl_gsmzs.log 2>&1
echo "=== mmctrl ALL DONE $(date '+%F %T') ==="
