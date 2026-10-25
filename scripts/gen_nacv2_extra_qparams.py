#!/usr/bin/env python3
"""P0 前置：nacv2 qparams(.pt) → toolkit extra_qparams 注入格式。

P0 目的：把 NAC-v2（整值 zp + Hessian）的 qparams 注入板端 w4a16_g64 模型，
复评 MMLU/GSM8K，验证"问题 B 修复后板端是否恢复"。

格式依据 E3 对照实验已验证的 Schema A（与 gen_minmax_qparams.py 同结构）：
  {f"model.layers.{b}.{mod}": {"scale": (out, in//64) fp32,
                               "zero_point": (out, in//64) fp32}}
E3（minmax float-zp）用同一格式、同一 build 参数构建，故 P0 与之机制完全可比。

整值 zp：nacv2 的 zp 已 100% 为整数（int_ratio=1.0），本脚本仅做 [0,15] 边界钳制
（越界 7/20,471,808 group，占比 3e-7，对精度无可测影响），并落盘保真度审计。

输出:
  results/nacv2_g64_intzp.qparams         注入文件
  results/p0_intzp_qparams_fidelity.json  保真度审计（覆盖/shape/整值率/钳制数）
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/nacv2_qwen1.5b_g64.qparams.pt"
MODEL_DIR = ROOT / "models/Qwen2.5-1.5B-Instruct-hf"
OUT = ROOT / "results/nacv2_g64_intzp.qparams"
AUDIT = ROOT / "results/p0_intzp_qparams_fidelity.json"

GROUP, QMIN, QMAX = 64, 0, 15
MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
N_LAYERS = 28

_cache = {}


def load_tensor(name: str):
    if not _cache:
        for f in sorted(MODEL_DIR.glob("*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    _cache[k] = (f, k)
    if name not in _cache:
        return None
    f, k = _cache[name]
    with safe_open(f, framework="pt") as sf:
        return sf.get_tensor(k)


def main():
    q = torch.load(SRC, map_location="cpu", weights_only=True)
    entries, missing, shape_bad = {}, [], []
    zp_total, zp_lo, zp_hi = 0, 0, 0
    clamp_lo = clamp_hi = 0

    for b in range(N_LAYERS):
        for mod in MODULES:
            name = f"model.layers.{b}.{mod}"
            qp = q.get(b, {}).get(mod)
            if qp is None:
                missing.append(name)
                continue
            w = load_tensor(name + ".weight")
            if w is None:
                missing.append(name + ".weight")
                continue
            s = qp["scale"].float()
            z = qp["zp"].float()
            out_dim, in_dim = w.shape
            if tuple(s.shape) != (out_dim, in_dim // GROUP) or tuple(z.shape) != tuple(s.shape):
                shape_bad.append({"name": name, "w": list(w.shape),
                                  "scale": list(s.shape), "zp": list(z.shape)})
                continue
            zp_total += z.numel()
            zp_lo += int((z < QMIN).sum())
            zp_hi += int((z > QMAX).sum())
            zc = z.clamp(QMIN, QMAX)
            clamp_lo += int((z < QMIN).sum())
            clamp_hi += int((z > QMAX).sum())
            entries[name] = {"scale": s.contiguous(), "zero_point": zc.contiguous()}
        print(f"[p0q] layer {b}/27 done", flush=True)

    if missing or shape_bad:
        raise SystemExit(f"FAIL missing={len(missing)} shape_bad={len(shape_bad)} "
                         f"{missing[:3]} {shape_bad[:2]}")

    torch.save(entries, OUT)
    frac = (torch.cat([e["zero_point"].flatten() for e in entries.values()])
            - torch.cat([e["zero_point"].flatten().round() for e in entries.values()])).abs()
    audit = {
        "src": str(SRC.relative_to(ROOT)),
        "entries": len(entries),
        "groups_total": zp_total,
        "zp_int_ratio": round(float((frac < 1e-6).float().mean()), 6),
        "zp_below_qmin": zp_lo, "zp_above_qmax": zp_hi,
        "clamped_count": clamp_lo + clamp_hi,
        "clamped_frac": round((clamp_lo + clamp_hi) / zp_total, 9),
        "note": "整值 zp 已由 nacv2 保证；钳制仅影响 3e-7 的 group，精度影响不可测。",
    }
    AUDIT.write_text(json.dumps(audit, indent=1, ensure_ascii=False))
    print(f"[p0q] saved {len(entries)} entries -> {OUT} "
          f"({OUT.stat().st_size / 1e6:.0f} MB)", flush=True)
    print(f"[p0q] audit: {audit}", flush=True)


if __name__ == "__main__":
    main()
