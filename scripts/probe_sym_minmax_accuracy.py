#!/usr/bin/env python3
"""验证 sym_minmax qparams 的反量化精度：w_int * scale 是否还原原 fp32 权重。

若 maxdiff/相对误差达正常 4bit 水平 → 量化 OK（问题在 g64/配置）；
若误差巨大 → 我的对称 minmax 量化损坏。
"""
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
QP = ROOT / "results/sym_minmax_g64.qparams"
MODEL_DIR = ROOT / "models/Qwen2.5-1.5B-Instruct-hf"
GROUP = 64

qp = torch.load(QP, map_location="cpu")

orig = {}
for f in sorted(MODEL_DIR.glob("*.safetensors")):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            orig[k] = sf.get_tensor(k)

worst = []
for k in sorted(qp.keys()):
    if not k.endswith(".weight") or k.endswith(".weight_qscale"):
        continue
    base = k[: -len(".weight")]
    s_key = base + ".weight_qscale"
    o_key = base + ".weight"
    if s_key not in qp or o_key not in orig:
        continue
    w_int = qp[k].float()
    s = qp[s_key].float()
    w_orig = orig[o_key].float()
    out, ind = w_int.shape
    n_g = s.shape[1]
    gs = ind // n_g
    s_b = s.repeat_interleave(gs, dim=1)
    w_pred = w_int * s_b
    diff = (w_pred - w_orig).abs()
    rel = diff / (w_orig.abs() + 1e-8)
    worst.append((rel.max().item(), diff.max().item(), base))
    print(f"{base}: max_abs={diff.max().item():.4f} max_rel={rel.max().item():.3f} "
          f"mean_abs={diff.mean().item():.5f}", flush=True)

worst.sort(reverse=True)
print("\nTOP5 worst (max_rel, max_abs, name):", flush=True)
for r in worst[:5]:
    print(f"  {r[0]:.3f}  {r[1]:.4f}  {r[2]}", flush=True)
print("DONE", flush=True)
