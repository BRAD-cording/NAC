#!/usr/bin/env python3
"""关键探针：验证 tk w4a16 的 weight 反量化语义。

gdq 落盘的 weight 值域为 [-8,7]（int4 有符号 = 对称）。本脚本验证：
    w_pred = weight * weight_qscale（对称，zp=0）
是否还原原始 fp32 权重。若 maxdiff 达量化误差级 → 对称成立（无 zp）；
若系统性偏差 → 非对称（需 zp，但 gdq 未落盘 zp，说明 w4a16 实为对称）。
"""
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
GDQ = ROOT / "gdq.qparams"
MODEL_DIR = ROOT / "models/tiny_qwen2"

qp = torch.load(GDQ, map_location="cpu")

orig = {}
for f in sorted(MODEL_DIR.glob("*.safetensors")):
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            orig[k] = sf.get_tensor(k)

print(f"n_qp={len(qp)} n_orig={len(orig)}", flush=True)
for k in sorted(qp.keys()):
    if not k.endswith(".weight"):
        continue
    base = k[: -len(".weight")]
    s_key = base + ".weight_qscale"
    o_key = base + ".weight"
    if s_key not in qp or o_key not in orig:
        print(f"skip {base}", flush=True)
        continue
    w_int = qp[k].float()
    s = qp[s_key].float()
    w_orig = orig[o_key].float()
    out, in_d = w_int.shape
    n_g = s.shape[1]
    gs = in_d // n_g
    s_b = s.repeat_interleave(gs, dim=1)  # (out, in)
    w_pred = w_int * s_b
    diff = (w_pred - w_orig).abs()
    rel = diff / (w_orig.abs() + 1e-6)
    print(f"{base}: maxdiff={diff.max().item():.4f} "
          f"mean_abs={diff.mean().item():.4f} "
          f"w_int[{w_int.min().item():.0f},{w_int.max().item():.0f}]", flush=True)
print("DONE", flush=True)
