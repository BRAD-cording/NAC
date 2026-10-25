#!/usr/bin/env python3
"""通用注入构建：load 1.5B → build(w4a16_g64, normal, extra_qparams=QP) → export。

R3 用：QP=results/sym_minmax_g64.qparams，验证注入生效（日志计数）并产出
板端评测所需的 .rkllm。

环境变量：
  MODEL  模型路径（默认 models/Qwen2.5-1.5B-Instruct-hf）
  QP     extra_qparams 文件路径（必需）
  OUT    导出 .rkllm 路径
  DATASET 校准集（extra_qparams 命中后跳过 Optimizing，通常用不到，仅传参兼容）
  DTYPE  quantized_dtype（默认 w4a16_g64；W4A16 非 g64 用 "w4a16"）
"""
import hashlib
import os
import time

from rkllm.api import rkllm_base as rb

MODEL = os.environ.get("MODEL", "models/Qwen2.5-1.5B-Instruct-hf")
QP = os.environ.get("QP")
OUT = os.environ.get("OUT")
DATASET = os.environ.get("DATASET", "")
DTYPE = os.environ.get("DTYPE", "w4a16_g64")

if not QP or not OUT:
    raise SystemExit("QP 和 OUT 环境变量必填")

llm = rb.RKLLMBase()
t0 = time.time()
ret = llm.load_huggingface(MODEL, None, "cpu", "float32", None, True)
print(f"[inj] load ret={ret} ({time.time() - t0:.0f}s)", flush=True)
if ret != 0:
    raise SystemExit("load failed")

kw = dict(do_quantization=True, quantized_dtype=DTYPE,
          quantized_algorithm="normal", optimization_level=1,
          target_platform="rk3576", num_npu_core=2, extra_qparams=QP)
if DATASET:
    kw["dataset"] = DATASET

t0 = time.time()
ret = llm.build(**kw)
print(f"[inj] build ret={ret} ({time.time() - t0:.0f}s)", flush=True)
if ret != 0:
    raise SystemExit("build failed")

t0 = time.time()
ret = llm.export_rkllm(OUT)
print(f"[inj] export ret={ret} ({time.time() - t0:.0f}s) -> {OUT}", flush=True)

if os.path.exists(OUT):
    h = hashlib.md5()
    with open(OUT, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print(f"[inj] md5={h.hexdigest()} size={os.path.getsize(OUT)}", flush=True)
print("[inj] DONE", flush=True)
