"""R3 基线复现 + R1 Schema 获取（一次构建两用）。

流程：load → build（无 extra_qparams，toolkit 自算，opt=1，真实校准集）
     → save_qparams（拿到 toolkit 权威 qparams 文件 = extra_qparams 的读入格式）
     → export_rkllm（得到"本地普通构建"模型，用于板端 GSM8K/MMLU 复现厂商 0.555 基线）

产物：
  results/tk_g64_full.qparams      toolkit 自算 qparams（R1 权威 Schema）
  models/local_g64_plain.rkllm     本地普通构建模型（R3 板端对照）
"""
import os
import time

from rkllm.api import rkllm_base as rb

MODEL = "models/Qwen2.5-1.5B-Instruct-hf"
QP = os.environ.get("R3_QP", "results/tk_w4a16_full.qparams")
OUT = os.environ.get("R3_OUT", "models/local_w4a16_plain.rkllm")
# optim 0 跳过病态 Optimizing（10-12 rebuild.log 记录的历史捷径；probe_qparam_attrs 曾用 optim 0 + dataset 走通）
OPT = int(os.environ.get("R3_OPT", "1"))

llm = rb.RKLLMBase()
t0 = time.time()
ret = llm.load_huggingface(MODEL, None, "cpu", "float32", None, True)
print(f"[r3] load ret={ret} ({time.time() - t0:.0f}s)", flush=True)
if ret != 0:
    raise SystemExit("load failed")

t0 = time.time()
ret = llm.build(do_quantization=True, quantized_dtype="w4a16",
                quantized_algorithm="normal", optimization_level=OPT,
                target_platform="rk3576", num_npu_core=2,
                dataset="data/quant_dataset.json")
print(f"[r3] build ret={ret} (opt={OPT}) ({time.time() - t0:.0f}s)", flush=True)
if ret != 0:
    raise SystemExit("build failed")

try:
    ret = llm.save_qparams(QP)
    print(f"[r3] save_qparams ret={ret} -> {QP}", flush=True)
except Exception as e:
    print(f"[r3] save_qparams FAIL {type(e).__name__}: {e}", flush=True)

t0 = time.time()
ret = llm.export_rkllm(OUT)
print(f"[r3] export ret={ret} ({time.time() - t0:.0f}s) -> {OUT}", flush=True)
print("[r3] DONE", flush=True)
