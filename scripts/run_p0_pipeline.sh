#!/usr/bin/env bash
# P0 实验链（2026-10-13）：GPTQ 全量校准 → PPL 门槛体检 → MMLU sim 四配置评测
# 用法: systemd-run --user --unit=nac-p0 --working-directory=<项目根> bash scripts/run_p0_pipeline.sh
set -uo pipefail
cd "$(dirname "$0")/.."
exec >> results/p0_pipeline.log 2>&1

echo "=== [1/3] GPTQ calib g64 (28 blocks) start $(date '+%F %T') ==="
taskset -c 4-7 .venv/bin/python scripts/gptq_calibrate.py \
    --model models/Qwen2.5-1.5B-Instruct-hf --group-size 64 \
    --out results/gptq_qwen1.5b_g64
[ -f results/gptq_qwen1.5b_g64.qparams.pt ] || { echo "!!! GPTQ FAILED, abort"; exit 1; }

echo "=== [2/3] PPL 门槛体检 start $(date '+%F %T') ==="
taskset -c 4-7 .venv/bin/python -c "
import torch, sys
sys.path.insert(0, '.')
from src.nac.validate_sim import apply_qparams, restore_from_disk, eval_ppl, load_heldout
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('models/Qwen2.5-1.5B-Instruct-hf')
model = AutoModelForCausalLM.from_pretrained('models/Qwen2.5-1.5B-Instruct-hf', dtype=torch.float32)
model.eval()
seqs = load_heldout(tok, 4, 512)
qp = torch.load('results/gptq_qwen1.5b_g64.qparams.pt', weights_only=True)
assert not torch.isnan(qp[0]['self_attn.q_proj']['scale']).any(), 'NaN in qparams'
apply_qparams(model, 64, 'minmax', backup_path='/tmp/p0_bak.pt')
ppl_mm = eval_ppl(model, seqs)
restore_from_disk(model, '/tmp/p0_bak.pt')
apply_qparams(model, 64, 'nac', qp, backup_path='/tmp/p0_bak.pt')
ppl_gq = eval_ppl(model, seqs)
restore_from_disk(model, '/tmp/p0_bak.pt')
ppl_fp = eval_ppl(model, seqs)
print(f'GATE 4-seq PPL: fp={ppl_fp:.3f} minmax={ppl_mm:.3f} gptq={ppl_gq:.3f}')
assert ppl_gq < ppl_mm, f'GPTQ({ppl_gq}) not better than minmax({ppl_mm}) — 量化实现可疑, 阻断 MMLU'
print('GATE PASS')
" || { echo "!!! PPL GATE FAILED, abort"; exit 1; }

echo "=== [3/3] MMLU sim (minmax,nac,gptq,fp16) 969q start $(date '+%F %T') ==="
taskset -c 4-7 .venv/bin/python scripts/eval_mmlu_sim.py \
    --model models/Qwen2.5-1.5B-Instruct-hf \
    --qparams results/nac_qwen1.5b_g64.qparams.pt --group-size 64 \
    --configs minmax,nac,gptq=results/gptq_qwen1.5b_g64.qparams.pt,fp16 \
    --batch-size 16 --out results/acc_mmlu_sim_g64.json
echo "=== pipeline done $(date '+%F %T') ==="
