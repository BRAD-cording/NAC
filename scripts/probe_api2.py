"""R1 补充探针 5：构造最小 QuantLinear，调用 calc_qparams/set_qparams 并 save_qparams，
直接观察 toolkit 的 qparams 结构与键名（= extra_qparams 的格式）。"""
import json

import torch

from rkllm.api import rkllm_base as rb

torch.manual_seed(0)


def try_step(desc, fn):
    try:
        r = fn()
        print(f"[q] {desc}: OK {'' if r is None else r}", flush=True)
        return True
    except Exception as e:
        print(f"[q] {desc}: FAIL {type(e).__name__}: {e}", flush=True)
        return False


for dtype_args in (("float32", "float32"), (torch.float32, torch.float32)):
    try:
        q = rb.QuantLinear(128, 64, False, *dtype_args, group_size=64)
        print(f"[q] construct OK dtypes={dtype_args}", flush=True)
        break
    except Exception as e:
        print(f"[q] construct fail dtypes={dtype_args}: {type(e).__name__}: {e}", flush=True)
else:
    raise SystemExit("cannot construct QuantLinear")

q.weight = torch.nn.Parameter(torch.randn(64, 128))
print(f"[q] vars={sorted(vars(q))}", flush=True)

try_step("set_qmode(w4a16_g64)", lambda: q.set_qmode("w4a16_g64"))
try_step("calc_qparams", lambda: q.calc_qparams())
print(f"[q] vars after calc={sorted(vars(q))}", flush=True)
for k, v in vars(q).items():
    if hasattr(v, "shape"):
        print(f"[q]   {k}: shape={list(v.shape)} dtype={v.dtype}", flush=True)
    else:
        print(f"[q]   {k}: {type(v).__name__}={str(v)[:80]}", flush=True)

# 手工 set_qparams：观察 shape 校验与写回键
s = torch.ones(64, 2)
zp = torch.zeros(64, 2)
if try_step("set_qparams(ones,zeros,w)", lambda: q.set_qparams(s, zp, q.weight)):
    for k, v in vars(q).items():
        if hasattr(v, "shape"):
            print(f"[q]   post {k}: shape={list(v.shape)} dtype={v.dtype}", flush=True)

# 伪造已 build 的 base，落盘看结构
llm = rb.RKLLMBase()
holder = torch.nn.Module()
holder.add_module("layers", torch.nn.ModuleList([q]))
llm.model = holder
for flag in ("built", "is_build", "builded"):
    try:
        setattr(llm, flag, True)
    except Exception:
        pass
if try_step("save_qparams(results/api_qparams_probe.qparams)",
            lambda: llm.save_qparams("results/api_qparams_probe.qparams")):
    d = torch.load("results/api_qparams_probe.qparams", map_location="cpu", weights_only=False)
    print(f"[q] saved type={type(d)} n={len(d)}", flush=True)
    items = list(d.items()) if isinstance(d, dict) else list(enumerate(d))
    for k, v in items[:3]:
        if isinstance(v, dict):
            print(f"[q]   key={k!r} value_keys={list(v)} "
                  + str({kk: (list(vv.shape) if hasattr(vv, 'shape') else vv) for kk, vv in v.items()}), flush=True)
        else:
            print(f"[q]   key={k!r} value={type(v)} {str(v)[:120]}", flush=True)
print("[q] DONE", flush=True)
