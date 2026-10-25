#!/usr/bin/env python3
"""tiny 注入验证：用 gdq 产出的 qparams 文件（或篡改版）走 normal 构建 + 导出。

目的：
1) 确认 extra_qparams 的真实格式被接受（日志应出现 N extra qparams already found and loaded!）；
2) 篡改 scale 后导出，与基准导出 md5 对比，判断 qparams 是否真正进入产物。

环境变量：
  QP   qparams 文件路径（默认 gdq.qparams）
  OUT  导出路径
"""
import hashlib
import os
import time

from rkllm.api import rkllm_base as rb

QP = os.environ.get("QP", "gdq.qparams")
OUT = os.environ.get("OUT", "models/tiny_inject.rkllm")

llm = rb.RKLLMBase()
t0 = time.time()
print(f"[inj] load ret={llm.load_huggingface('models/tiny_qwen2', None, 'cpu', 'float32', None, True)} "
      f"({time.time() - t0:.0f}s)", flush=True)

t0 = time.time()
ret = llm.build(do_quantization=True, quantized_dtype="w4a16_g64",
                quantized_algorithm="normal", optimization_level=1,
                target_platform="rk3576", num_npu_core=2,
                dataset="results/mini_quant_dataset.json", extra_qparams=QP)
print(f"[inj] build ret={ret} ({time.time() - t0:.0f}s)", flush=True)

ret = llm.export_rkllm(OUT)
print(f"[inj] export ret={ret} -> {OUT}", flush=True)

if os.path.exists(OUT):
    h = hashlib.md5()
    with open(OUT, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print(f"[inj] md5={h.hexdigest()} size={os.path.getsize(OUT)}", flush=True)
print("[inj] DONE", flush=True)
