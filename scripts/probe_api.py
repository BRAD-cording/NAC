"""R1 补充探针 4：从 API 层反推 qparams 文件结构。

思路：save_qparams 会把模型内的 qparam 落盘；若能构造一个最小 QuantLinear 并
伪造已 build 状态，就能直接看到它写出的键名/结构（= extra_qparams 的读入格式）。
本探针只做 introspection 与构造尝试，不跑 build。
"""
import inspect

import torch

from rkllm.api import rkllm_base as rb

for obj, name in [(rb.QuantLinear.__init__, "QuantLinear.__init__"),
                  (rb.QuantLinear.set_qparams, "QuantLinear.set_qparams"),
                  (rb.QuantLinear.calc_qparams, "QuantLinear.calc_qparams"),
                  (rb.RKLLMBase.save_qparams, "RKLLMBase.save_qparams")]:
    print(f"[api] {name}: sig={inspect.signature(obj)} doc={obj.__doc__!r}", flush=True)

print(f"[api] QuantLinear attrs={sorted(k for k in vars(rb.QuantLinear) if not k.startswith('__'))}", flush=True)

lin = torch.nn.Linear(128, 64, bias=False)
print(f"[api] nn.Linear state_dict={list(lin.state_dict())}", flush=True)

# 构造尝试：按 nn.Linear 签名 + 常见量化字段
for kwargs in ({"in_features": 128, "out_features": 64},
               {"in_features": 128, "out_features": 64, "group_size": 64, "qbit": 4},
               {"128": None}):
    try:
        if "128" in kwargs:
            q = rb.QuantLinear(128, 64)
        else:
            q = rb.QuantLinear(**kwargs)
        print(f"[api] constructed QuantLinear ok kwargs={kwargs} type={type(q)}", flush=True)
        print(f"[api] q state_dict={list(q.state_dict())}", flush=True)
        print(f"[api] q attrs={[a for a in dir(q) if 'q' in a.lower()]}", flush=True)
        break
    except Exception as e:
        print(f"[api] construct fail kwargs={kwargs}: {type(e).__name__}: {e}", flush=True)

print("[api] DONE", flush=True)
