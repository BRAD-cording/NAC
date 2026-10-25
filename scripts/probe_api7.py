"""R1 补充探针 10：set_qmode(dst, group) 后再 calc_qparams，取 scale/zp 布局。"""
import torch

from rkllm.api import rkllm_base as rb

src, dst = rb.dst_f16, rb.dst_w4a16
for args in ((dst, 64), (dst, 128), ("w4a16_g64", 64), ("w4a16_g64", "g64")):
    try:
        q = rb.QuantLinear(128, 64, False, src, dst)
        q.weight = torch.nn.Parameter(torch.randn(64, 128) * 0.05)
        q.set_qmode(*args)
        q.calc_qparams()
        info = {k: (list(getattr(q, k).shape), str(getattr(q, k).dtype))
                for k in ("scale", "zp", "qweight", "q_group")
                if hasattr(getattr(q, k, None), "shape")}
        print(f"[s] set_qmode{args} -> q_group={q.q_group} {info}", flush=True)
        if "scale" in info:
            print(f"[s] HIT args={args}", flush=True)
            print(f"[s] scale[:4]={q.scale.flatten()[:4].tolist()} zp[:4]={q.zp.flatten()[:4].tolist()}", flush=True)
            break
    except Exception as e:
        print(f"[s] args={args} FAIL {type(e).__name__}: {e}", flush=True)
print("[s] DONE", flush=True)
