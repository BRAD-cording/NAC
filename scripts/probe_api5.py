"""R1 补充探针 8：dump QuantLinear 内 scale/zp/qweight 真实布局，并正确定位 save_qparams 落盘。"""
import glob
import os

import torch

from rkllm.api import rkllm_base as rb

OUT = "/media/pi/DATA/rknn-llm/nac_paper/results/api_qparams_probe.qparams"
for p in glob.glob(OUT):
    os.remove(p)

src, dst = rb.dst_f16, rb.dst_w4a16
q = rb.QuantLinear(128, 64, False, src, dst)
q.weight = torch.nn.Parameter(torch.randn(64, 128) * 0.05)


def show(tag):
    for k in ("qweight", "scale", "zp", "q_group", "min_int", "max_int"):
        v = getattr(q, k, None)
        if hasattr(v, "shape"):
            print(f"[d] {tag} {k}: shape={list(v.shape)} dtype={v.dtype} "
                  f"min={float(v.min()):.4g} max={float(v.max()):.4g}", flush=True)
        else:
            print(f"[d] {tag} {k}: {type(v).__name__} {str(v)[:100]}", flush=True)


show("init")
q.calc_qparams()
show("calc")

try:
    q.set_qparams(q.scale, q.zp, q.weight)
    show("set")
except Exception as e:
    print(f"[d] set_qparams FAIL {type(e).__name__}: {e}", flush=True)

llm = rb.RKLLMBase()
holder = torch.nn.Module()
holder.add_module("layers", torch.nn.ModuleList([q]))
llm.model = holder
ret = llm.save_qparams(OUT)
print(f"[d] save ret={ret} exists={os.path.exists(OUT)}", flush=True)
if os.path.exists(OUT):
    d = torch.load(OUT, map_location="cpu", weights_only=False)
    print(f"[d] file type={type(d).__name__}", flush=True)
    items = list(d.items()) if isinstance(d, dict) else list(enumerate(d))
    print(f"[d] n={len(items)}", flush=True)
    for k, v in items[:4]:
        print(f"[d] key={k!r} -> {type(v).__name__}: "
              + str({kk: (list(vv.shape) if hasattr(vv, 'shape') else vv) for kk, vv in v.items()}
                    if isinstance(v, dict) else str(v)[:150]), flush=True)
print("[d] DONE", flush=True)
