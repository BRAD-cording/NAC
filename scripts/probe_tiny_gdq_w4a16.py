#!/usr/bin/env python3
"""探针：tiny + gdq + w4a16（非 g64）落盘 qparams，确认非 g64 的 weight_qscale 布局。"""
import time

import torch
from rkllm.api import rkllm_base as rb

MODEL = "models/tiny_qwen2"
DATASET = "results/mini_quant_dataset.json"
OUT = "results/tiny_gdq_w4a16.qparams"

llm = rb.RKLLMBase()
print(f"load ret={llm.load_huggingface(MODEL, None, 'cpu', 'float32', None, True)}", flush=True)
ret = llm.build(do_quantization=True, quantized_dtype="w4a16",
                quantized_algorithm="gdq", optimization_level=1,
                target_platform="rk3576", num_npu_core=2, dataset=DATASET)
print(f"build ret={ret}", flush=True)
print(f"save ret={llm.save_qparams(OUT)}", flush=True)

obj = torch.load(OUT + ".qparams", map_location="cpu")
print(f"type={type(obj)} n={len(obj)}", flush=True)
for k in sorted(obj.keys()):
    v = obj[k]
    if "weight_qscale" in k or k.endswith(".weight"):
        print(f"  {k}: {tuple(v.shape)} {v.dtype} range[{v.min().item()},{v.max().item()}]", flush=True)
    if list(obj.keys()).index(k) > 15:
        break
print("DONE", flush=True)
