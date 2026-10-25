"""路径C 阶段1探针：原始 fp16 模型 × plain w4a16 × gdq 算法。

目的：
1. 验证「原始模型 × plain w4a16」组合能否通过 Optimizing（此前唯一未测组合；
   original×g64、dequant×g64、dequant×plain 均卡死 Optimizing 0/28）
2. 让 gdq 算法缓存 .qparams 文件，逆向其格式 → 为 NAC scale/zp 注入做格式映射
3. 若全程通过，导出探针产物 _gdq_probe.rkllm（可弃），证明该组合全链可用
"""
import time
from pathlib import Path

from rkllm.api import RKLLM

MODEL = "models/Qwen2.5-1.5B-Instruct-hf"
DATASET = "data/quant_dataset.json"

print(f"[probe] cwd={Path.cwd()}", flush=True)
llm = RKLLM()
t0 = time.time()
ret = llm.load_huggingface(model=MODEL, model_lora=None, device="cpu")
print(f"[probe] load_huggingface ret={ret} ({time.time()-t0:.0f}s)", flush=True)
if ret != 0:
    raise SystemExit("load failed")

before = set(Path(".").glob("*.qparams")) | set(Path("models").glob("*.qparams"))
t1 = time.time()
ret = llm.build(do_quantization=True, quantized_dtype="w4a16",
                quantized_algorithm="gdq", optimization_level=1,
                target_platform="rk3576", num_npu_core=2, dataset=DATASET)
print(f"[probe] build ret={ret} ({time.time()-t1:.0f}s)", flush=True)
if ret != 0:
    raise SystemExit("build failed")

after = set(Path(".").glob("*.qparams")) | set(Path("models").glob("*.qparams"))
new = after - before
print(f"[probe] new qparams files: {[str(p) for p in new]}", flush=True)
for p in new:
    print(f"[probe] qparams {p}: {p.stat().st_size} bytes", flush=True)

t2 = time.time()
ret = llm.export_rkllm("models/_gdq_probe.rkllm")
print(f"[probe] export ret={ret} ({time.time()-t2:.0f}s)", flush=True)
print("[probe] DONE", flush=True)
