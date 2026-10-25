"""R1 补充探针 6：列出 rkllm_base 模块级对象，寻找 QuantLinear 构造所需的 dtype/config 对象。"""
from rkllm.api import rkllm_base as rb

names = [n for n in dir(rb) if not n.startswith("__")]
print(f"[mod] names={names}", flush=True)
for n in names:
    o = getattr(rb, n)
    if isinstance(o, type):
        print(f"[mod] class {n}: sig={getattr(o, '__init__', None).__doc__}", flush=True)
    elif not callable(o):
        print(f"[mod] const {n}={str(o)[:120]}", flush=True)
    else:
        print(f"[mod] func {n}", flush=True)
print("[mod] DONE", flush=True)
