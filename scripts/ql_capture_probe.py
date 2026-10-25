"""QuantLinear 运行时捕获探针（路径C 阶段2前置）。

猴子补丁 QuantLinear.calc_qparams，在 build 量化阶段捕获前几个 QuantLinear
实例的全部属性（找 层名/NAC qparams 映射所需信息），随后 os._exit 安全退出，
不进入 Optimizing 阶段、不占内存。
"""
import json
import os

from rkllm.api import rkllm_base as rb

CAPTURE_N = 4
captured = []
orig_calc = rb.QuantLinear.calc_qparams


def patched_calc(self):
    if len(captured) < CAPTURE_N:
        info = {}
        for a in dir(self):
            if a.startswith("__"):
                continue
            try:
                v = getattr(self, a)
                if callable(v):
                    continue
                info[a] = repr(v)[:200]
            except Exception as e:
                info[a] = f"<err {e}>"
        # 权重形状/内容指纹（用于与 NAC qparams 对齐）
        try:
            w = self.weight
            info["_weight_shape"] = str(tuple(w.shape))
            info["_weight_sum"] = f"{float(w.float().abs().sum()):.4e}"
        except Exception as e:
            info["_weight_err"] = str(e)
        try:
            info["_calc_sig"] = str(orig_calc)
        except Exception:
            pass
        captured.append(info)
        with open("/tmp/ql_capture.json", "w") as f:
            json.dump(captured, f, indent=1, ensure_ascii=False)
        print(f"[probe] captured {len(captured)}/{CAPTURE_N}", flush=True)
        if len(captured) >= CAPTURE_N:
            print("[probe] capture complete, aborting build safely", flush=True)
            os._exit(0)
    return orig_calc(self)


rb.QuantLinear.calc_qparams = patched_calc
print("[probe] calc_qparams patched", flush=True)

llm = rb.RKLLMBase()
ret = llm.load_huggingface("models/Qwen2.5-1.5B-Instruct-hf", None, "cpu",
                           "float32", None, True)
print(f"[probe] load ret={ret}", flush=True)
ret = llm.build(do_quantization=True, quantized_dtype="w4a16_g64",
                quantized_algorithm="normal", optimization_level=1,
                target_platform="rk3576", num_npu_core=2,
                dataset="data/quant_dataset.json")
print(f"[probe] build ret={ret} (no capture? calc may not be called)", flush=True)
