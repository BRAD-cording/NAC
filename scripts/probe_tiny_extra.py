"""R1 收尾探针：用 tiny_qwen2（2 层）复验 extra_qparams 的真实语义与注入机制。

用法：
  QP=<qparams 文件> 时：build 时传 extra_qparams，预期日志出现 "N extra qparams ..."，
  用于确认 save_qparams 产物的读入格式（flat {param_name: tensor}）。
  未给 QP 时：patch calc_qparams，打印 toolkit 真正使用的 scale/zp 形状与 dtype，
  用于确认"外部注入 scale/zp"只能走 calc_qparams/set_qparams 补丁路径。
"""
import os

import torch

from rkllm.api import rkllm_base as rb

MODEL = os.environ.get("PROBE_MODEL", "models/tiny_qwen2")
DATASET = os.environ.get("PROBE_DATASET", "results/mini_quant_dataset.json")
QP = os.environ.get("QP", "")
OUT = os.environ.get("PROBE_OUT", "results/tiny_probe.qparams")

if not QP:
    orig_calc = rb.QuantLinear.calc_qparams
    orig_set = rb.QuantLinear.set_qparams
    state = {"calc": 0, "set": 0, "diag": []}

    def patched_calc(self):
        r = orig_calc(self)
        state["calc"] += 1
        if len(state["diag"]) < 3:
            d = {"hidden": [self.in_features, self.out_features]}
            for k in ("scale", "zp", "qweight", "q_group", "weight_qscale", "weight_qzero"):
                v = getattr(self, k, None)
                d[k] = (list(v.shape), str(v.dtype)) if hasattr(v, "shape") else v
            state["diag"].append(d)
        return r

    def patched_set(self, *a, **kw):
        state["set"] += 1
        return orig_set(self, *a, **kw)

    rb.QuantLinear.calc_qparams = patched_calc
    rb.QuantLinear.set_qparams = patched_set
    print("[tiny] patched calc_qparams/set_qparams", flush=True)

llm = rb.RKLLMBase()
ret = llm.load_huggingface(MODEL, None, "cpu", "float32", None, True)
print(f"[tiny] load ret={ret}", flush=True)

kw = {"do_quantization": True, "quantized_dtype": "w4a16_g64",
      "quantized_algorithm": "normal", "optimization_level": 1,
      "target_platform": "rk3576", "num_npu_core": 2, "dataset": DATASET}
if QP:
    kw["extra_qparams"] = QP
    print(f"[tiny] extra_qparams={QP}", flush=True)

ret = llm.build(**kw)
print(f"[tiny] build ret={ret}", flush=True)

if not QP:
    print(f"[tiny] calc_calls={state['calc']} set_calls={state['set']}", flush=True)
    for d in state["diag"]:
        print(f"[tiny] diag={d}", flush=True)
    ret = llm.save_qparams(OUT)
    print(f"[tiny] save_qparams ret={ret} -> {OUT}", flush=True)
print("[tiny] DONE", flush=True)
