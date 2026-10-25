"""PPL 一致性/扩样评测（E4 复核 + 论文 64-seq 口径共用脚本）。

E4 口径（无参数运行，与 10-08 锚点可比）：fp16/minmax/nac，16 seqs × 1024。
论文扩样口径（review_publication_20261017.md P2-7）：
  python scripts/ppl_coherence.py --num-seqs 64 --modes fp16,minmax,nac,gptq \
      --gptq-qparams results/gptq_qwen1.5b_g64.qparams.pt --out results/ppl_g64_64seq.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import torch
from src.nac.validate_sim import apply_qparams, restore, eval_ppl, load_heldout
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser(description="PPL coherence / expanded-seq eval")
ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct-hf")
ap.add_argument("--num-seqs", type=int, default=16, help="E4 口径 16；论文口径 ≥64")
ap.add_argument("--seqlen", type=int, default=1024)
ap.add_argument("--group-size", type=int, default=64)
ap.add_argument("--modes", default="fp16,minmax,nac")
ap.add_argument("--qparams", default="results/nac_qwen1.5b_g64.qparams.pt")
ap.add_argument("--gptq-qparams", default=None, help="modes 含 gptq 时必填")
ap.add_argument("--out", default=None, help="可选 JSON 落盘路径")
args = ap.parse_args()

torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
model.eval()
seqs = load_heldout(tok, args.num_seqs, args.seqlen)
print(f"[pplchk] heldout {seqs.shape}", flush=True)
qps = torch.load(args.qparams, map_location="cpu", weights_only=True)
gqps = torch.load(args.gptq_qparams, map_location="cpu", weights_only=True) \
    if args.gptq_qparams else None

summary = {"model": args.model, "num_seqs": args.num_seqs, "seqlen": args.seqlen,
           "group_size": args.group_size, "ppl": {}}
for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
    t0 = time.time()
    if mode == "fp16":
        ppl = eval_ppl(model, seqs, bs=1)
    else:
        if mode == "gptq":
            if gqps is None:
                raise SystemExit("--gptq-qparams required for mode gptq")
            bk = apply_qparams(model, args.group_size, "nac", qparams=gqps)
        else:
            bk = apply_qparams(model, args.group_size, mode,
                               qparams=qps if mode == "nac" else None)
        ppl = eval_ppl(model, seqs, bs=1)
        restore(model, bk)
        del bk
    summary["ppl"][mode] = round(float(ppl), 4)
    print(f"[pplchk] {mode}: ppl={ppl:.3f} ({time.time()-t0:.0f}s)", flush=True)

if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[pplchk] -> {args.out}", flush=True)
print("[pplchk] DONE", flush=True)
