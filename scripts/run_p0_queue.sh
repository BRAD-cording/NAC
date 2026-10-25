#!/usr/bin/env bash
# P0 自治实验队列（抗断网/断连：systemd --user + linger 已开启）
# 阶段0: 等待/自愈 MMLU g64 四配置（单元停止则自动断点重启）
# 阶段1: GSM8K sim g64 三配置（minmax/nac/gptq）
# 阶段2: GPTQ g128 校准
# 阶段3: MMLU g128 三配置（fp16 复用 g64 锚点）
# 全程增量落盘，任意阶段可断点续跑
set -uo pipefail
cd "$(dirname "$0")/.."
exec >> results/p0_queue.log 2>&1
MODEL=models/Qwen2.5-1.5B-Instruct-hf
MMLU64='.venv/bin/python scripts/eval_mmlu_sim.py --model '"$MODEL"' --qparams results/nac_qwen1.5b_g64.qparams.pt --group-size 64 --configs minmax,nac,gptq=results/gptq_qwen1.5b_g64.qparams.pt,fp16 --batch-size 16 --out results/acc_mmlu_sim_g64.json'

echo "=== queue start $(date '+%F %T') ==="

# ---- 阶段0: 自愈等待 MMLU g64 ----
while true; do
    done_cfgs=$(.venv/bin/python -c "import json;print(len(json.load(open('results/acc_mmlu_sim_g64.full.json'))['configs']))" 2>/dev/null || echo 0)
    [ "${done_cfgs:-0}" -ge 4 ] && break
    if ! systemctl --user is-active --quiet nac-p0; then
        echo "[queue] nac-p0 inactive but MMLU g64 incomplete(${done_cfgs}/4) -> resume $(date '+%T')"
        systemctl --user reset-failed nac-p0 2>/dev/null || true
        systemd-run --user --unit=nac-p0 --working-directory="$PWD" \
            bash -c "taskset -c 4-7 $MMLU64 >> results/p0_pipeline.log 2>&1" || true
        sleep 90
    fi
    sleep 300
done
echo "=== [1/3] MMLU g64 4-configs DONE $(date '+%F %T') ==="

# ---- 阶段1: GSM8K g64 ----
taskset -c 4-7 .venv/bin/python scripts/eval_gsm8k_sim.py \
    --model "$MODEL" \
    --qparams results/nac_qwen1.5b_g64.qparams.pt --group-size 64 \
    --configs minmax,nac,gptq=results/gptq_qwen1.5b_g64.qparams.pt \
    --batch-size 16 --out results/acc_gsm8k_sim_g64.json
echo "=== [2/3] GSM8K g64 3-configs DONE $(date '+%F %T') ==="

# ---- 阶段2: GPTQ g128 校准 ----
[ -f results/gptq_qwen1.5b_g128.qparams.pt ] || \
taskset -c 4-7 .venv/bin/python scripts/gptq_calibrate.py \
    --model "$MODEL" --group-size 128 --out results/gptq_qwen1.5b_g128
[ -f results/gptq_qwen1.5b_g128.qparams.pt ] || { echo "!!! gptq g128 FAILED, stop queue"; exit 1; }
echo "=== GPTQ g128 calib DONE $(date '+%F %T') ==="

# ---- 阶段3: MMLU g128（fp16 复用 g64 锚点）----
taskset -c 4-7 .venv/bin/python scripts/eval_mmlu_sim.py \
    --model "$MODEL" \
    --qparams results/nac_qwen1.5b_g128.qparams.pt --group-size 128 \
    --configs minmax,nac,gptq=results/gptq_qwen1.5b_g128.qparams.pt \
    --batch-size 16 --out results/acc_mmlu_sim_g128.json
echo "=== queue ALL DONE $(date '+%F %T') ==="
