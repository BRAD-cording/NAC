#!/usr/bin/env python3
"""探针：打印 build / quant_model / fakeq_model / save_qparams 的 Python 源码。"""
import inspect

from rkllm.api import rkllm_base as rb

for m in ("build", "quant_model", "fakeq_model", "save_qparams"):
    f = getattr(rb.RKLLMBase, m, None)
    print("=" * 30, m, "=" * 30, flush=True)
    try:
        print(inspect.getsource(f), flush=True)
    except Exception as e:
        print(f"SRC_ERR {type(e).__name__}: {e}", flush=True)
