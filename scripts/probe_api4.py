"""R1 补充探针 7：用真实 RKLLMQTYPE 构造 QuantLinear，跑 calc_qparams，并 save_qparams 反推文件结构。"""
import torch

from rkllm.api import rkllm_base as rb

qs = [(n, getattr(rb, n)) for n in dir(rb)
      if type(getattr(rb, n)).__name__ == "RKLLMQTYPE"]
for n, v in qs:
    print(f"[t] {n}: {v}", flush=True)

src = rb.dst_f16
dsts = [v for n, v in qs if "w4a16" in getattr(v, "name", "").lower()]
dst = dsts[0]
print(f"[t] using src={src} dst={dst}", flush=True)

q = rb.QuantLinear(128, 64, False, src, dst)
q.weight = torch.nn.Parameter(torch.randn(64, 128) * 0.05)
print(f"[t] vars={sorted(vars(q))}", flush=True)
try:
    q.set_qmode(dst)
    print("[t] set_qmode OK", flush=True)
except Exception as e:
    print(f"[t] set_qmode FAIL {type(e).__name__}: {e}", flush=True)

try:
    q.calc_qparams()
    print("[t] calc_qparams OK", flush=True)
except Exception as e:
    print(f"[t] calc_qparams FAIL {type(e).__name__}: {e}", flush=True)

for k, v in vars(q).items():
    if hasattr(v, "shape"):
        print(f"[t]   {k}: shape={list(v.shape)} dtype={v.dtype} "
              f"min={float(v.min()):.4g} max={float(v.max()):.4g}", flush=True)
    elif hasattr(v, "weight") or hasattr(v, "group_size"):
        print(f"[t]   {k}: {type(v).__name__} {v}", flush=True)

llm = rb.RKLLMBase()
holder = torch.nn.Module()
holder.add_module("layers", torch.nn.ModuleList([q]))
llm.model = holder
for flag in ("built", "is_build", "builded", "built_", "quantized"):
    try:
        if hasattr(llm, flag):
            setattr(llm, flag, True)
            print(f"[t] set flag {flag}", flush=True)
    except Exception as e:
        print(f"[t] flag {flag} fail: {e}", flush=True)

try:
    ret = llm.save_qparams("results/api_qparams_probe.qparams")
    print(f"[t] save_qparams ret={ret}", flush=True)
    d = torch.load("results/api_qparams_probe.qparams", map_location="cpu", weights_only=False)
    print(f"[t] file type={type(d)}", flush=True)
    items = list(d.items()) if isinstance(d, dict) else list(enumerate(d))
    print(f"[t] n={len(items)}", flush=True)
    for k, v in items[:3]:
        if isinstance(v, dict):
            print(f"[t]   key={k!r} subkeys={list(v)} "
                  + str({kk: (list(vv.shape) if hasattr(vv, 'shape') else vv) for kk, vv in v.items()}), flush=True)
        else:
            print(f"[t]   key={k!r} {type(v).__name__} {str(v)[:150]}", flush=True)
except Exception as e:
    print(f"[t] save_qparams FAIL {type(e).__name__}: {e}", flush=True)
print("[t] DONE", flush=True)
