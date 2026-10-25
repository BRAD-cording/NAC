"""路径C 注入构建：原始 fp16 模型 × w4a16_g64 × normal + calc_qparams 补丁注入 NAC scale/zp。

原理：toolkit 在 build() 量化阶段对每个 QuantLinear 调 calc_qparams()（min-max 自算）。
补丁流程：先放行 toolkit 自算（确立 weight_qscale/weight_qzero 的 dtype/shape 约定），
随即用 NAC per-group scale/zp 调 set_qparams(s, zp, w) 覆盖 —— 与官方 extra_qparams
文件加载效果等价，绕开未知文件格式。
层→NAC 映射：data_ptr 优先（权重共享时零开销），退化 (shape,|W|和) 指纹；
首层自验证：回读覆盖后的 buffer 与 NAC 值比对（shape/dtype/最大偏差），写
results/pathc_inject_diag.json。
"""
import json
import time

import torch
from rkllm.api import rkllm_base as rb

MODEL = "models/Qwen2.5-1.5B-Instruct-hf"
QPARAMS_PT = "results/nac_qwen1.5b_g64.qparams.pt"
DATASET = "data/quant_dataset.json"
OUT = "models/nac_qwen1.5b_g64_nac.rkllm"
DIAG = "results/pathc_inject_diag.json"

orig_calc = rb.QuantLinear.calc_qparams
state = {"resolved": 0, "fallback": 0, "missed": 0, "n_q": 0, "first_layer_diag": None}


def main():
    qparams = torch.load(QPARAMS_PT, map_location="cpu", weights_only=True)
    print(f"[inject] qparams blocks: {len(qparams)}", flush=True)

    llm = rb.RKLLMBase()
    ret = llm.load_huggingface(MODEL, None, "cpu", "float32", None, True)
    print(f"[inject] load ret={ret}", flush=True)
    if ret != 0:
        raise SystemExit("load failed")

    # ---- QuantLinear → (bi, name) 映射 ----
    ptr_map, fp_map = {}, {}
    with torch.no_grad():
        for bi, block in enumerate(llm.model.model.layers):
            for name, mod in block.named_modules():
                if not isinstance(mod, torch.nn.Linear):
                    continue
                w = mod.weight
                key = (bi, name)
                ptr_map[w.data_ptr()] = key
                fp_map[(tuple(w.shape), round(float(w.float().abs().sum()), 3))] = key
    print(f"[inject] map built: {len(ptr_map)} linears", flush=True)

    def resolve(self):
        key = ptr_map.get(self.weight.data_ptr())
        if key is not None:
            state["resolved"] += 1
            return key
        fk = (tuple(self.weight.shape), round(float(self.weight.float().abs().sum()), 3))
        key = fp_map.get(fk)
        state["fallback" if key is not None else "missed"] += 1
        return key

    def patched_calc(self):
        key = resolve(self)
        if key is None:
            return orig_calc(self)
        bi, name = key
        qp = qparams[bi].get(name)
        if qp is None:
            state["missed"] += 1
            return orig_calc(self)
        r = orig_calc(self)  # toolkit 自算，确立 buffer 约定
        self.set_qparams(qp["scale"], qp["zp"], self.weight)  # NAC 覆盖
        state["n_q"] += 1
        if state["first_layer_diag"] is None:
            d = {"key": key, "nac_scale_shape": list(qp["scale"].shape),
                 "nac_zp_shape": list(qp["zp"].shape)}
            try:
                bs = self.weight_qscale
                d["buf_scale_shape"] = list(bs.shape)
                d["buf_scale_dtype"] = str(bs.dtype)
                d["scale_maxdiff"] = float((bs.float() - qp["scale"].to(bs.dtype).float()).abs().max())
                if getattr(self, "weight_qzero", None) is not None:
                    bz = self.weight_qzero
                    d["buf_zp_shape"] = list(bz.shape)
                    d["buf_zp_dtype"] = str(bz.dtype)
                    d["zp_maxdiff"] = float((bz.float() - qp["zp"].to(bz.dtype).float()).abs().max())
            except Exception as e:
                d["diag_err"] = str(e)
            state["first_layer_diag"] = d
            print(f"[inject] first-layer diag: {json.dumps(d)}", flush=True)
            json.dump(state, open(DIAG, "w"), indent=1, default=str)
        return r

    rb.QuantLinear.calc_qparams = patched_calc
    print("[inject] calc_qparams patched (NAC overwrite mode)", flush=True)

    t0 = time.time()
    ret = llm.build(do_quantization=True, quantized_dtype="w4a16_g64",
                    quantized_algorithm="normal", optimization_level=1,
                    target_platform="rk3576", num_npu_core=2, dataset=DATASET)
    print(f"[inject] build ret={ret} ({time.time() - t0:.0f}s) state={state}", flush=True)
    json.dump(state, open(DIAG, "w"), indent=1, default=str)
    if ret != 0:
        raise SystemExit("build failed")

    ret = llm.export_rkllm(OUT)
    print(f"[inject] export ret={ret}", flush=True)
    print(f"[inject] DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
