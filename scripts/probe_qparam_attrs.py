"""R1 补充探针 3：在 build 进行中（Building 阶段完成后）从旁路线程扫描
QuantLinear/Linear 上的 qparams 属性，直接取得 toolkit 的权威键名/形状/布局。

动机：full build 的 "Optimizing model" 在 qemu 下耗时不可估；但 qparam 属性
在 Building 阶段（~12.5 min）就已挂到模块上，旁路读取即可，无需等 build 返回。
"""
import json
import os
import threading
import time
from pathlib import Path

from rkllm.api import rkllm_base as rb

MODEL = "models/Qwen2.5-1.5B-Instruct-hf"
DUMP = Path("results/probe_attrs.json")
QP_OUT = "results/probe_g64_watchdog.qparams"
WAIT = int(os.environ.get("PROBE_WAIT", "900"))

llm = rb.RKLLMBase()


def dump_attrs():
    time.sleep(WAIT)
    out = {"ts": time.strftime("%H:%M:%S"), "modules": [], "n_q": 0, "errors": []}
    try:
        for name, mod in llm.model.named_modules():
            attrs = {}
            for k in ("weight_qscale", "weight_qzero", "qweight", "qweight_int4", "qparams"):
                v = getattr(mod, k, None)
                if v is None:
                    continue
                if hasattr(v, "shape"):
                    attrs[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                else:
                    attrs[k] = {"type": type(v).__name__, "repr": str(v)[:200]}
            if attrs:
                out["n_q"] += 1
                if len(out["modules"]) < 5:
                    out["modules"].append({"name": name, "cls": type(mod).__name__, "attrs": attrs})
        try:
            ret = llm.save_qparams(QP_OUT)
            out["save_qparams_ret"] = ret
        except Exception as e:
            out["errors"].append(f"save_qparams: {e}")
    except Exception as e:
        out["errors"].append(f"dump: {e}")
    DUMP.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[watchdog] dumped n_q={out['n_q']} -> {DUMP}", flush=True)
    os._exit(0)


threading.Thread(target=dump_attrs, daemon=True).start()
print(f"[watchdog] armed, dump at +{WAIT}s", flush=True)

t0 = time.time()
ret = llm.load_huggingface(MODEL, None, "cpu", "float32", None, True)
print(f"[watchdog] load ret={ret} ({time.time() - t0:.0f}s)", flush=True)

t0 = time.time()
ret = llm.build(do_quantization=True, quantized_dtype="w4a16_g64",
                quantized_algorithm="normal", optimization_level=0,
                target_platform="rk3576", num_npu_core=2,
                dataset="results/mini_quant_dataset.json")
print(f"[watchdog] build ret={ret} ({time.time() - t0:.0f}s)", flush=True)
print("[watchdog] build finished before watchdog; dumping now", flush=True)
dump_attrs()
