#!/usr/bin/env python3
"""生成 minmax float-zp 对照 qparams（注入格式 Schema A）。

目的：E3 对照实验——minmax 的 zp = -wmin/scale 同为 float（不取整），
与 NAC zp 语义一致。若板端对照也崩 → float-zp 注入路径普遍失真；
若对照正常 → NAC 特有问题。

流式读 safetensors，峰值内存 <1GB，可与 sim 评测并存。
输出: /tmp/minmax_g64_full.qparams  {f"model.layers.{b}.{mod}": {"scale","zero_point"}}
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/Qwen2.5-1.5B-Instruct-hf"
OUT = ROOT / "results/minmax_g64_full.qparams"
GROUP = 64
QMIN, QMAX = 0, 15
MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
N_LAYERS = 28


def init_minmax(w: torch.Tensor, group_size: int):
    """与 src/nac/quantize.py 完全一致的 minmax 语义（float zp 不取整）。"""
    out_dim, in_dim = w.shape
    assert in_dim % group_size == 0
    wg = w.reshape(out_dim, in_dim // group_size, group_size)
    wmin = wg.amin(dim=-1)
    wmax = wg.amax(dim=-1)
    wmax = torch.maximum(wmax, wmin + 1e-8)
    scale = (wmax - wmin) / (QMAX - QMIN)
    zp = -wmin / scale
    return scale, zp


def main():
    entries = {}
    missing = []
    for b in range(N_LAYERS):
        for mod in MODULES:
            name = f"model.layers.{b}.{mod}.weight"
            t = _load_tensor(name)
            if t is None:
                missing.append(name)
                continue
            w = t.float()
            scale, zp = init_minmax(w, GROUP)
            entries[f"model.layers.{b}.{mod}"] = {
                "scale": scale.contiguous().to(torch.float32),
                "zero_point": zp.contiguous().to(torch.float32),
            }
        print(f"[mmq] layer {b}/27 done", flush=True)
    if missing:
        print(f"[mmq] MISSING {len(missing)}: {missing[:3]}...", file=sys.stderr)
        sys.exit(1)
    torch.save(entries, OUT)
    n = len(entries)
    sz = OUT.stat().st_size / 1e6
    print(f"[mmq] saved {n} entries -> {OUT} ({sz:.0f} MB)", flush=True)
    # sanity: zp 分布（应为 float 连续，与 NAC 同语义）
    zps = torch.cat([e["zero_point"].flatten() for e in entries.values()])
    frac = (zps - zps.round()).abs()
    stats = {"n": int(zps.numel()), "int_ratio": round((frac < 1e-6).float().mean().item(), 4),
             "frac_mean": round(frac.mean().item(), 4), "zp_min": round(zps.min().item(), 3),
             "zp_max": round(zps.max().item(), 3)}
    (ROOT / "results/mmctrl_zp_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[mmq] zp stats: {stats}", flush=True)


_cache = {}


def _load_tensor(name: str):
    """safetensors 流式读：只映射需要的 tensor。"""
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


if __name__ == "__main__":
    main()
