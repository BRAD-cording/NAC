"""R1 补充探针：追踪 toolkit 是否消费 extra_qparams，并记录 set_qparams 的 dtype/shape 约定。

做法：用 Schema B（{name: {weight_qscale, weight_qzero}}）写 8 层注入文件，
monkey-patch QuantLinear.set_qparams / calc_qparams 打印调用情况；
只看前几分钟（构建会被外部 timeout 终止），据以判断注入是否生效。
"""
import sys

import torch
from rkllm.api import rkllm_base as rb

FPATH = "/tmp/probe_Bt_hf.qparams"
MODEL = "models/Qwen2.5-1.5B-Instruct-hf"

qp = torch.load("results/nac_qwen1.5b_g64.qparams.pt", map_location="cpu", weights_only=True)
s0 = qp[0]["self_attn.q_proj"]["scale"]
z0 = qp[0]["self_attn.q_proj"]["zp"]

entries = {}
for b in range(2):
    for mod in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
        entries[f"model.layers.{b}.{mod}"] = {"weight_qscale": s0, "weight_qzero": z0}
torch.save(entries, FPATH)
print(f"[trace] wrote {len(entries)} entries (schema B) -> {FPATH}", flush=True)

orig_set = rb.QuantLinear.set_qparams
orig_calc = rb.QuantLinear.calc_qparams
state = {"set": 0, "calc": 0, "set_match_s0": 0, "set_diag": [], "calc_diag": []}


def patched_set(self, *a, **kw):
    state["set"] += 1
    try:
        s = a[0]
        d = {"args": len(a), "kw": list(kw),
             "s_shape": list(s.shape) if hasattr(s, "shape") else None,
             "s_dtype": str(getattr(s, "dtype", None)),
             "s_min": float(s.min()), "s_max": float(s.max())}
        if len(a) > 1 and hasattr(a[1], "shape"):
            d["z_shape"] = list(a[1].shape)
            d["z_dtype"] = str(getattr(a[1], "dtype", None))
            d["z_min"] = float(a[1].min())
            d["z_max"] = float(a[1].max())
        if hasattr(s, "shape") and list(s.shape) == list(s0.shape):
            d["s0_maxdiff"] = float((s.float() - s0.float()).abs().max())
            if d["s0_maxdiff"] < 1e-6:
                state["set_match_s0"] += 1
        if len(state["set_diag"]) < 3:
            state["set_diag"].append(d)
    except Exception as e:
        if len(state["set_diag"]) < 3:
            state["set_diag"].append({"err": str(e), "args": len(a)})
    return orig_set(self, *a, **kw)


def patched_calc(self, *a, **kw):
    state["calc"] += 1
    if len(state["calc_diag"]) < 3:
        d = {"attrs": [k for k in vars(self) if "q" in k.lower()][:12]}
        for k in ("weight_qscale", "weight_qzero"):
            v = getattr(self, k, None)
            if v is not None and hasattr(v, "shape"):
                d[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
        state["calc_diag"].append(d)
    r = orig_calc(self, *a, **kw)
    if state["calc"] in (1, 2, 3):
        print(f"[trace] calc#{state['calc']} after -> " +
              str({k: getattr(self, k, None).shape for k in ("weight_qscale", "weight_qzero")
                   if getattr(self, k, None) is not None}), flush=True)
    return r


rb.QuantLinear.set_qparams = patched_set
rb.QuantLinear.calc_qparams = patched_calc
print("[trace] patched set_qparams/calc_qparams", flush=True)

llm = rb.RKLLMBase()
ret = llm.load_huggingface(MODEL, None, "cpu", "float32", None, False)
print(f"[trace] load ret={ret}", flush=True)

ret = llm.build(do_quantization=True, quantized_dtype="w4a16_g64",
                quantized_algorithm="normal", optimization_level=1,
                target_platform="rk3576", num_npu_core=2,
                dataset="data/quant_dataset.json", extra_qparams=FPATH)
print(f"[trace] build ret={ret} state={ {k: v for k, v in state.items() if k != 'calc_diag'} }", flush=True)
print(f"[trace] set_diag={state['set_diag']}", flush=True)
print(f"[trace] calc_diag={state['calc_diag']}", flush=True)
print("[trace] DONE", flush=True)
