"""R1 补充探针 2：打印 toolkit 载入后的真实模块名路径，用于对齐 extra_qparams 键名。"""
from rkllm.api import rkllm_base as rb

llm = rb.RKLLMBase()
ret = llm.load_huggingface("models/Qwen2.5-1.5B-Instruct-hf", None, "cpu", "float32", None, False)
print(f"[names] load ret={ret}", flush=True)

m = llm.model
print(f"[names] model class={type(m).__name__}", flush=True)
names = [(n, type(mod).__name__) for n, mod in m.named_modules()]
print(f"[names] total={len(names)}", flush=True)
for n, t in names[:15]:
    print(f"[names] {t:24s} {n}", flush=True)
print("[names] --- sample with 'layers.0' ---", flush=True)
for n, t in names:
    if n.startswith("model.layers.0") or n.startswith("layers.0"):
        print(f"[names] {t:24s} {n}", flush=True)
print("[names] --- linear-ish classes ---", flush=True)
seen = {}
for n, t in names:
    if "linear" in t.lower():
        seen.setdefault(t, n)
print(f"[names] {seen}", flush=True)
kinds = sorted({t for _, t in names})
print(f"[names] module classes={kinds}", flush=True)
print("[names] DONE", flush=True)
