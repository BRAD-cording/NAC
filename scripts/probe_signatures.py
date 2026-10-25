#!/usr/bin/env python3
"""探针：打印 RKLLMBase 全部方法签名，寻找可用的注入点。"""
import inspect

from rkllm.api import rkllm_base as rb

names = [n for n in dir(rb.RKLLMBase) if not n.startswith("__")]
print("METHODS:", names, flush=True)
for m in names:
    f = getattr(rb.RKLLMBase, m, None)
    try:
        print(f"{m}: {inspect.signature(f)}", flush=True)
    except Exception as e:
        print(f"{m}: SIG_ERR {e}", flush=True)
