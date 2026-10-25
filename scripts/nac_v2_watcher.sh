#!/usr/bin/env bash
# NAC-v2 自动流水线（问题 A 修复，docs/review_publication_20261017.md §3）：
# 等 queue ALL DONE 释放内存窗口 → v2 g64 校准（Hessian+激活加权+int-zp）
# → sim GSM8K v2（判别任务）→ sim MMLU v2。各阶段按产物幂等跳过。
# 开机自愈：boot_resume.sh 第 6 节在重启后重新拉起本 watcher。
set -uo pipefail
cd "$(dirname "$0")/.."
exec >> results/nac_v2.log 2>&1
echo "=== nacv2 watcher start $(date '+%F %T') ==="

# 1) 等 queue ALL DONE（sim 评测与 E3/E4 构建窗口全部结束后内存才够校准）
while ! grep -qF 'queue ALL DONE' results/p0_queue.log 2>/dev/null; do
    sleep 300
done
echo "=== queue done -> nacv2 window open $(date '+%F %T') ==="

MODEL=models/Qwen2.5-1.5B-Instruct-hf
QP=results/nacv2_qwen1.5b_g64.qparams.pt

# 2) v2 g64 校准（int-zp 默认开，同服问题 B；~1.5-2h）
if [ ! -f "$QP" ]; then
    taskset -c 4-7 .venv/bin/python scripts/nac_v2_calibrate.py \
        --model "$MODEL" --group-size 64 --out results/nacv2_qwen1.5b_g64
    [ -f "$QP" ] || { echo "!!! nacv2 calib FAILED $(date '+%F %T')"; exit 1; }
fi
echo "=== nacv2 calib OK $(date '+%F %T') ==="

# 3) sim GSM8K v2（判别任务：对照 gptq 0.5625 / nac 0.070 / minmax 0.075；~10.5h）
if [ ! -f results/acc_gsm8k_sim_g64_nacv2.json ]; then
    taskset -c 4-7 .venv/bin/python scripts/eval_gsm8k_sim.py \
        --model "$MODEL" --qparams "$QP" --group-size 64 \
        --configs "nacv2=$QP" --batch-size 16 \
        --out results/acc_gsm8k_sim_g64_nacv2.json
fi
echo "=== nacv2 gsm8k sim done $(date '+%F %T') ==="

# 4) sim MMLU v2（969 题，~40min；达标参考：nac 0.3932 / gptq 0.4520 / minmax 0.4675）
if [ ! -f results/acc_mmlu_sim_g64_nacv2.json ]; then
    taskset -c 4-7 .venv/bin/python scripts/eval_mmlu_sim.py \
        --model "$MODEL" --qparams "$QP" --group-size 64 \
        --configs "nacv2=$QP" --batch-size 16 \
        --out results/acc_mmlu_sim_g64_nacv2.json
fi
echo "=== nacv2 ALL DONE $(date '+%F %T') ==="

# 5) PPL 扩样 64 seq（论文口径 P2-7；E4 今晚自动跑的仍是无参数 16-seq 口径，互不干扰）
if [ ! -f results/ppl_g64_64seq.json ]; then
    taskset -c 4-7 .venv/bin/python scripts/ppl_coherence.py --num-seqs 64 \
        --modes fp16,minmax,nac,gptq \
        --gptq-qparams results/gptq_qwen1.5b_g64.qparams.pt \
        --out results/ppl_g64_64seq.json
fi
echo "=== nacv2 ppl64 done $(date '+%F %T') ==="

# 6) AWQ 基线校准（激活感知 scale + 折叠 + clip 网格 + int-zp；~2-2.5h）
AWQ=results/awq_qwen1.5b_g64
if [ ! -f "$AWQ.qparams.pt" ] || [ ! -f "$AWQ.fold.pt" ]; then
    taskset -c 4-7 .venv/bin/python scripts/awq_calibrate.py \
        --model "$MODEL" --group-size 64 --out "$AWQ"
    [ -f "$AWQ.qparams.pt" ] && [ -f "$AWQ.fold.pt" ] || { echo "!!! awq calib FAILED $(date '+%F %T')"; exit 1; }
fi
echo "=== awq calib OK $(date '+%F %T') ==="

# 7) sim GSM8K AWQ（判别任务；~10.5h）
if [ ! -f results/acc_gsm8k_sim_g64_awq.json ]; then
    taskset -c 4-7 .venv/bin/python scripts/eval_gsm8k_sim.py \
        --model "$MODEL" --qparams "$AWQ.qparams.pt" --group-size 64 \
        --configs "awq=$AWQ.qparams.pt" --fold "$AWQ.fold.pt" \
        --batch-size 16 \
        --out results/acc_gsm8k_sim_g64_awq.json
fi
echo "=== awq gsm8k sim done $(date '+%F %T') ==="

# 8) sim MMLU AWQ（969 题，~40min）
if [ ! -f results/acc_mmlu_sim_g64_awq.json ]; then
    taskset -c 4-7 .venv/bin/python scripts/eval_mmlu_sim.py \
        --model "$MODEL" --qparams "$AWQ.qparams.pt" --group-size 64 \
        --configs "awq=$AWQ.qparams.pt" --fold "$AWQ.fold.pt" \
        --batch-size 16 \
        --out results/acc_mmlu_sim_g64_awq.json
fi
echo "=== awq ALL DONE $(date '+%F %T') ==="

# 9) v2 PPL 64-seq（补 §3.1 v2 行 PPL 空缺；v1/v2 qparams 格式一致，nac 模式注入 v2 文件；~2.2h）
if [ ! -f results/ppl_g64_64seq_nacv2.json ]; then
    taskset -c 4-7 .venv/bin/python scripts/ppl_coherence.py --num-seqs 64 \
        --modes nac \
        --qparams results/nacv2_qwen1.5b_g64.qparams.pt \
        --out results/ppl_g64_64seq_nacv2.json
fi
echo "=== nacv2 ppl64(v2) done $(date '+%F %T') ==="
