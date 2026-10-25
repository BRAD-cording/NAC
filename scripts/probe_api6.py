"""R1 补充探针 9：定位 QuantLinear 分组参数，使 calc_qparams 真正产出 scale/zp。"""
import torch

from rkllm.api import rkllm_base as rb

src, dst = rb.dst_f16, rb.dst_w4a16
VARIANTS = [
    {"group_size": 64},
    {"q_group": 64},
    {"group_size": 64, "quantized_dtype": "w4a16_g64"},
    {"group_size": "g64"},
    {"quantized_dtype": "w4a16_g64"},
]

for kw in VARIANTS:
    try:
        q = rb.QuantLinear(128, 64, False, src, dst, **kw)
        q.weight = torch.nn.Parameter(torch.randn(64, 128) * 0.05)
        q.calc_qparams()
        info = {}
        for k in ("scale", "zp", "qweight", "q_group"):
            v = getattr(q, k, None)
            info[k] = (list(v.shape), str(v.dtype)) if hasattr(v, "shape") else v
        print(f"[v] kw={kw} -> {info}", flush=True)
        if getattr(q, "scale", None) is not None:
            print(f"[v] HIT kw={kw}", flush=True)
            print(f"[v] scale={q.scale.flatten()[:6].tolist()}", flush=True)
            print(f"[v] zp={q.zp.flatten()[:6].tolist() if hasattr(q.zp, 'flatten') else q.zp}", flush=True)
            break
    except Exception as e:
        print(f"[v] kw={kw} FAIL {type(e).__name__}: {e}", flush=True)
print("[v] DONE", flush=True)
