"""extra_qparams file format probe: load_weight=False (low mem), parallel-safe with eval.
Usage: python3 probe_fmt.py <schema> <style>
schema: A={name:{scale,zero_point}} B={name:{weight_qscale,weight_qzero}}
        C={name:(scale,zp)} D={name:{scales,qzeros}}
style:  hf=model.layers.{b}.{mod}  blk=blk.{b}.{mod}
Look for: "Using extra qparams:" (accepted) vs validation errors.
"""
import sys
import torch
sys.path.insert(0, ".")
from rkllm.api import rkllm_base as rb

SCHEMA = sys.argv[1] if len(sys.argv) > 1 else "A"
STYLE = sys.argv[2] if len(sys.argv) > 2 else "hf"
FPATH = f"/tmp/probe_{SCHEMA}_{STYLE}.qparams"

qp = torch.load("results/nac_qwen1.5b_g64.qparams.pt", map_location="cpu", weights_only=True)
s0 = qp[0]["self_attn.q_proj"]["scale"]
z0 = qp[0]["self_attn.q_proj"]["zp"]

def make_entry(scale, zp):
    if SCHEMA == "A": return {"scale": scale, "zero_point": zp}
    if SCHEMA == "B": return {"weight_qscale": scale, "weight_qzero": zp}
    if SCHEMA == "C": return (scale, zp)
    if SCHEMA == "D": return {"scales": scale, "qzeros": zp}
    raise SystemExit("bad schema")

entries = {}
for b in range(2):
    for mod in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
        name = f"model.layers.{b}.{mod}" if STYLE == "hf" else f"blk.{b}.{mod}"
        if SCHEMA == "E":  # 扁平参数名 + 后缀
            entries[name + ".weight_qscale"] = s0
            entries[name + ".weight_qzero"] = z0
        else:
            entries[name] = make_entry(s0, z0)

torch.save(entries, FPATH)
print(f"[fmtprobe] schema={SCHEMA} style={STYLE} entries={len(entries)} -> {FPATH}", flush=True)

llm = rb.RKLLMBase()
ret = llm.load_huggingface("models/Qwen2.5-1.5B-Instruct-hf", None, "cpu", "float32", None, False)
print(f"[fmtprobe] load(noweight) ret={ret}", flush=True)

ret = llm.build(do_quantization=True, quantized_dtype="w4a16_g64",
                quantized_algorithm="normal", optimization_level=1,
                target_platform="rk3576", num_npu_core=2,
                dataset="data/quant_dataset.json", extra_qparams=FPATH)
print(f"[fmtprobe] build ret={ret}", flush=True)
print("[fmtprobe] DONE", flush=True)
