#!/usr/bin/env python3
"""生成 tk w4a16 对称格式的 minmax qparams（R3 复现厂商基线用）。

关键发现（probe_symmetry.py 实证）：tk w4a16 的 weight 是 int4 对称 [-8,7]，
反量化 w = w_int * weight_qscale（zp=0，无 zp 通道）。extra_qparams 权威格式为
扁平 state-dict：{name.weight: int4-as-int8, name.weight_qscale: fp32(out, in/64),
bias/layernorm: fp32}。

本脚本产出 minmax 对称量化 qparams：scale = amax/7（严格对称，w_int∈[-7,7]）。
bias / layernorm / norm 从原始模型原样复制（与 gdq 落盘行为一致）。
"""
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/Qwen2.5-1.5B-Instruct-hf"
OUT = ROOT / "results/sym_minmax_g64.qparams"
GROUP = 64
MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
N_LAYERS = 28

_cache = {}


def load_tensor(name: str):
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


def sym_minmax(w: torch.Tensor, group_size: int):
    """对称 minmax：scale = amax/8，w_int = clamp(round(w/scale), -8, 7) ∈ [-8,7]。

    注意：tk w4a16 要求 weight 值域为 int4 完整范围 [-8,7]（probe_tiny_range.py 实证：
    [-7,7] 的 weight 被"识别"计数但值被忽略，产物与 0-loaded 相同）。故必须用 amax/8，
    让 w=amax 时 round 到 8、clamp 到 7，值域覆盖 [-8,7]。
    """
    out_dim, in_dim = w.shape
    assert in_dim % group_size == 0
    wg = w.reshape(out_dim, in_dim // group_size, group_size)
    amax = wg.abs().amax(dim=-1)
    amax = torch.clamp(amax, min=1e-8)
    scale = amax / 8.0
    w_int = torch.clamp(torch.round(wg / scale.unsqueeze(-1)), -8, 7).to(torch.int8)
    return w_int.reshape(out_dim, in_dim), scale.contiguous()


def main():
    entries = {}
    missing = []

    # 1) 量化 7 模块 × 28 层的 weight
    for b in range(N_LAYERS):
        for mod in MODULES:
            name = f"model.layers.{b}.{mod}"
            w = load_tensor(name + ".weight")
            if w is None:
                missing.append(name + ".weight")
                continue
            w_int, scale = sym_minmax(w.float(), GROUP)
            entries[name + ".weight"] = w_int
            entries[name + ".weight_qscale"] = scale.to(torch.float32)
        print(f"[sym] layer {b}/27 done", flush=True)

    # 2) 复制 bias（q/k/v 有 bias，o/gate/up/down 无）+ layernorm + norm
    for b in range(N_LAYERS):
        for mod in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
            name = f"model.layers.{b}.{mod}.bias"
            t = load_tensor(name)
            if t is not None:
                entries[name] = t.float()
        for ln in ["input_layernorm", "post_attention_layernorm"]:
            name = f"model.layers.{b}.{ln}.weight"
            t = load_tensor(name)
            if t is not None:
                entries[name] = t.float()
    t = load_tensor("model.norm.weight")
    if t is not None:
        entries["model.norm.weight"] = t.float()

    if missing:
        print(f"[sym] MISSING {len(missing)}: {missing[:3]}...", file=sys.stderr)
        sys.exit(1)

    torch.save(entries, OUT)
    n_w = sum(1 for k in entries if k.endswith(".weight") and not k.endswith(".weight_qscale"))
    n_s = sum(1 for k in entries if k.endswith(".weight_qscale"))
    print(f"[sym] saved {len(entries)} keys (weight={n_w} qscale={n_s}) -> {OUT} "
          f"({OUT.stat().st_size / 1e6:.0f} MB)", flush=True)

    # sanity：weight 值域应为 int4 完整范围 [-8,7]（流式，避免 torch.cat OOM）
    wmin, wmax = 10, -10
    for k, v in entries.items():
        if k.endswith(".weight") and not k.endswith(".weight_qscale"):
            wmin = min(wmin, v.min().item())
            wmax = max(wmax, v.max().item())
    print(f"[sym] weight int4 range [{wmin}, {wmax}]", flush=True)


if __name__ == "__main__":
    main()
