#!/usr/bin/env python3
"""生成 tk W4A16（非 g64，per-channel 非对称）的 minmax qparams（R3 复现厂商基线用）。

关键发现（probe_tiny_gdq_w4a16 + 反量化验证）：
  - W4A16（非 g64）= per-channel 非对称，反量化 w = weight_qscale * (weight - weight_qzero)
  - weight 值域 [-8,7]（int4 有符号），weight_qscale / weight_qzero 形状 (out, 1)
  - weight_qzero 为 float（非整数），tk 原生支持 float zp
  - 厂商预转基线即 W4A16 非对称（GSM8K 0.555）

本脚本产出 per-channel 非对称 minmax：
  scale = (wmax-wmin)/15, zp = -8 - wmin/scale, w_int = clamp(round(w/scale+zp), -8, 7)
"""
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/Qwen2.5-1.5B-Instruct-hf"
OUT = ROOT / "results/asym_minmax_w4a16.qparams"
MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
N_LAYERS = 28
QMIN, QMAX = -8, 7

_cache = {}


def load_tensor(name):
    if not _cache:
        files = sorted(MODEL_DIR.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors under {MODEL_DIR}")
        for f in files:
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    _cache[k] = (f, k)
    if name not in _cache:
        return None
    f, k = _cache[name]
    with safe_open(f, framework="pt") as sf:
        return sf.get_tensor(k)


def asym_minmax_per_channel(w):
    """per-channel 非对称 minmax：scale/zp (out,1)，w_int (out,in) ∈ [-8,7]。"""
    wmin = w.amin(dim=1, keepdim=True)
    wmax = w.amax(dim=1, keepdim=True)
    wmax = torch.maximum(wmax, wmin + 1e-8)
    scale = (wmax - wmin) / float(QMAX - QMIN)  # /15
    zp = QMIN - wmin / scale  # -8 - wmin/scale
    w_int = torch.clamp(torch.round(w / scale + zp), QMIN, QMAX).to(torch.int8)
    return w_int, scale.contiguous(), zp.contiguous()


def main():
    entries = {}
    missing = []
    for b in range(N_LAYERS):
        for mod in MODULES:
            name = f"model.layers.{b}.{mod}"
            w = load_tensor(name + ".weight")
            if w is None:
                missing.append(name + ".weight")
                continue
            w_int, scale, zp = asym_minmax_per_channel(w.float())
            entries[name + ".weight"] = w_int
            entries[name + ".weight_qscale"] = scale.to(torch.float32)
            entries[name + ".weight_qzero"] = zp.to(torch.float32)
        print(f"[asym] layer {b}/27 done", flush=True)

    # bias（q/k/v）+ layernorm + norm
    for b in range(N_LAYERS):
        for mod in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
            t = load_tensor(f"model.layers.{b}.{mod}.bias")
            if t is not None:
                entries[f"model.layers.{b}.{mod}.bias"] = t.float()
        for ln in ["input_layernorm", "post_attention_layernorm"]:
            t = load_tensor(f"model.layers.{b}.{ln}.weight")
            if t is not None:
                entries[f"model.layers.{b}.{ln}.weight"] = t.float()
    t = load_tensor("model.norm.weight")
    if t is not None:
        entries["model.norm.weight"] = t.float()

    if missing:
        print(f"[asym] MISSING {len(missing)}: {missing[:3]}...", file=sys.stderr)
        sys.exit(1)

    torch.save(entries, OUT)
    print(f"[asym] saved {len(entries)} keys -> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)",
          flush=True)

    # sanity
    wmin, wmax = 10, -10
    for k, v in entries.items():
        if k.endswith(".weight") and v.dtype == torch.int8:
            wmin = min(wmin, v.min().item())
            wmax = max(wmax, v.max().item())
    print(f"[asym] weight int4 range [{wmin}, {wmax}]", flush=True)


if __name__ == "__main__":
    main()
