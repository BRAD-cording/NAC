#!/usr/bin/env python3
"""探针：让 toolkit 自己构建一次并落盘 qparams，以取得 extra_qparams 的权威 Schema。

背景：10-15/10-18/10-22 三次"注入"构建均打印 `0 extra qparams already found and loaded!`，
三个产物 md5 完全相同（3bd8c0ba…），说明注入从未生效，板端 NAC 结论全部作废。
本脚本只做"构建 → save_qparams → 退出"，不导出（省 ~45min）。
产物：results/probe_g64.qparams（toolkit 自算的 w4a16_g64 qparams）
"""
import json
import os
import time

from rkllm.api import rkllm_base as rb

MODEL = "models/Qwen2.5-1.5B-Instruct-hf"
OUT = os.environ.get("PROBE_OUT", "results/probe_g64.qparams")
DATASET = os.environ.get("PROBE_DATASET", "data/quant_dataset.json")
OPT = int(os.environ.get("PROBE_OPT", "1"))

llm = rb.RKLLMBase()
t0 = time.time()
ret = llm.load_huggingface(MODEL, None, "cpu", "float32", None, True)
print(f"[probe] load ret={ret} ({time.time() - t0:.0f}s)", flush=True)

t0 = time.time()
ret = llm.build(do_quantization=True, quantized_dtype="w4a16_g64",
                quantized_algorithm="normal", optimization_level=OPT,
                target_platform="rk3576", num_npu_core=2,
                dataset=DATASET)
print(f"[probe] build ret={ret} ({time.time() - t0:.0f}s)", flush=True)

ret = llm.save_qparams(OUT)
print(f"[probe] save_qparams ret={ret} -> {OUT}", flush=True)
print("[probe] DONE", flush=True)
