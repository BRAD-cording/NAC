#!/usr/bin/env python3
"""篡改 gdq.qparams 中单层 scale，用于验证 extra_qparams 是否真正进入产物。

默认：把 model.layers.0.self_attn.q_proj.weight_qscale 乘 8。
"""
import os

import torch

SRC = os.environ.get("SRC", "gdq.qparams")
DST = os.environ.get("DST", "results/tiny_gdq_mut.qparams")
LAYER = os.environ.get("LAYER", "model.layers.0.self_attn.q_proj")
FACTOR = float(os.environ.get("FACTOR", "8"))
MODE = os.environ.get("MODE", "scale")

qp = torch.load(SRC, map_location="cpu")
key_s = f"{LAYER}.weight_qscale"
key_w = f"{LAYER}.weight"
if MODE == "scale":
    qp[key_s] = qp[key_s] * FACTOR
    print(f"[mut] {key_s}: x{FACTOR} -> mean={qp[key_s].mean().item():.6f}", flush=True)
elif MODE == "weight":
    w = qp[key_w]
    qp[key_w] = torch.clamp(w + int(FACTOR), -8, 7).to(w.dtype)
    print(f"[mut] {key_w}: +{int(FACTOR)} (clamped) -> "
          f"range [{qp[key_w].min().item()}, {qp[key_w].max().item()}]", flush=True)
else:
    raise SystemExit(f"bad MODE {MODE}")
torch.save(qp, DST)
print(f"[mut] saved {len(qp)} keys -> {DST}", flush=True)
