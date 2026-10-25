#!/usr/bin/env python3
"""tiny 对照：定位"1.5B 337 loaded 但 md5 == 0-loaded"的根因。

假设 H1：tk w4a16 要求 weight 值域 [-8,7]（int4 完整范围），我给的 sym_minmax 是
[-7,7]（amax/7），缺了 -8 档 → weight 被"识别"（计数）但值被忽略（回退默认）。
假设 H2：1.5B 上 extra_qparams 的 weight 根本没进产物（tiny 是特例）。

本脚本在 tiny 上跑 3 组：sym7([-7,7]) / sym8([-8,7]) / empty(0-loaded 参照)，
对比 md5。若 sym7==empty 且 sym8!=empty → H1 成立；若 sym7、sym8 均 != empty
→ sym_minmax 格式本身有效，问题在 1.5B 特有（H2）。
"""
import hashlib
import os

import torch
from safetensors import safe_open
from rkllm.api import rkllm_base as rb

MODEL = "models/tiny_qwen2"
MODULES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
N_LAYERS = 2
GROUP = 64

_cache = {}


def load(name):
    if not _cache:
        import pathlib
        for f in sorted(pathlib.Path(MODEL).glob("*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    _cache[k] = (f, k)
    if name not in _cache:
        return None
    f, k = _cache[name]
    with safe_open(f, framework="pt") as sf:
        return sf.get_tensor(k)


def gen(mode):
    entries = {}
    for b in range(N_LAYERS):
        for mod in MODULES:
            name = f"model.layers.{b}.{mod}"
            w = load(name + ".weight").float()
            out, ind = w.shape
            wg = w.reshape(out, ind // GROUP, GROUP)
            amax = wg.abs().amax(-1).clamp(min=1e-8)
            qmax = 7 if mode == "sym7" else 8
            scale = amax / qmax
            w_int = torch.clamp(torch.round(wg / scale.unsqueeze(-1)), -8, 7).to(torch.int8)
            entries[name + ".weight"] = w_int.reshape(out, ind)
            entries[name + ".weight_qscale"] = scale.float()
    for b in range(N_LAYERS):
        for mod in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
            t = load(f"model.layers.{b}.{mod}.bias")
            if t is not None:
                entries[f"model.layers.{b}.{mod}.bias"] = t.float()
        for ln in ["input_layernorm", "post_attention_layernorm"]:
            t = load(f"model.layers.{b}.{ln}.weight")
            if t is not None:
                entries[f"model.layers.{b}.{ln}.weight"] = t.float()
    t = load("model.norm.weight")
    if t is not None:
        entries["model.norm.weight"] = t.float()
    return entries


def build_md5(qp_path, out_path):
    llm = rb.RKLLMBase()
    llm.load_huggingface(MODEL, None, "cpu", "float32", None, True)
    llm.build(do_quantization=True, quantized_dtype="w4a16_g64",
              quantized_algorithm="normal", optimization_level=1,
              target_platform="rk3576", num_npu_core=2, extra_qparams=qp_path)
    llm.export_rkllm(out_path)
    h = hashlib.md5()
    with open(out_path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    for mode in ["sym7", "sym8", "empty"]:
        qp_path = f"results/tiny_range_{mode}.qparams"
        out = f"models/tiny_range_{mode}.rkllm"
        if mode == "empty":
            torch.save({}, qp_path)
        else:
            entries = gen(mode)
            ws = torch.cat([v.flatten() for k, v in entries.items()
                            if k.endswith(".weight") and not k.endswith(".weight_qscale")])
            torch.save(entries, qp_path)
            print(f"[{mode}] n_keys={len(entries)} w_range=[{ws.min().item()},{ws.max().item()}]",
                  flush=True)
        md5 = build_md5(qp_path, out)
        print(f"[{mode}] md5={md5} size={os.path.getsize(out)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
