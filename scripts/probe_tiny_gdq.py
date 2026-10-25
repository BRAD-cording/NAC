#!/usr/bin/env python3
"""探针：tiny 模型用 gdq 算法构建后 save_qparams，确认 extra_qparams 的权威 Schema。

背景：normal 算法下 save_qparams 只落盘 bias/layernorm 等"非量化参数"（11 条），
无法反映 per-group scale/zp 的真实键名与结构。文档称 extra_qparams 亦可承接
"gdq/grq 缓存的量化参数"，故用 gdq 跑一遍 tiny，观察它写出的结构。
"""
import time

import torch
from rkllm.api import rkllm_base as rb

MODEL = "models/tiny_qwen2"
DATASET = "results/mini_quant_dataset.json"
OUT = "results/tiny_gdq.qparams"

llm = rb.RKLLMBase()
t0 = time.time()
print(f"[gdq] load ret={llm.load_huggingface(MODEL, None, 'cpu', 'float32', None, True)} "
      f"({time.time() - t0:.0f}s)", flush=True)

t0 = time.time()
ret = llm.build(do_quantization=True, quantized_dtype="w4a16_g64",
                quantized_algorithm="gdq", optimization_level=1,
                target_platform="rk3576", num_npu_core=2, dataset=DATASET)
print(f"[gdq] build ret={ret} ({time.time() - t0:.0f}s)", flush=True)

print(f"[gdq] save_qparams ret={llm.save_qparams(OUT)}", flush=True)

obj = torch.load(OUT, map_location="cpu")
print(f"[gdq] dumped type={type(obj)} n={len(obj)}", flush=True)
for i, (k, v) in enumerate(obj.items()):
    if i >= 6:
        break
    if isinstance(v, dict):
        print(f"  {k}: dict{list(v.keys())}", flush=True)
        for kk, vv in v.items():
            print(f"     {kk}: {type(vv).__name__} "
                  f"{tuple(vv.shape) if hasattr(vv, 'shape') else vv} "
                  f"{getattr(vv, 'dtype', '')}", flush=True)
    else:
        print(f"  {k}: {type(v).__name__} "
              f"{tuple(v.shape) if hasattr(v, 'shape') else v}", flush=True)
print("[gdq] DONE", flush=True)
