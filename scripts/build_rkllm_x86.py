"""阶段2（x86 toolkit）：dequant HF 模型 → .rkllm。

由 src/nac/export_rkllm.py --mode gen-dequant 产出的 dequant 模型（fp16，权重
严格落在 NAC 网格上）经 toolkit normal(min-max) 重量化精确还原 NAC scale/zp。

板上运行方式（qemu-x86 用户态模拟）：
  cd /home/pi/rknn-llm/nac_paper && \
  QEMU_LD_PREFIX=/tmp/rktoolkit/sysroot /tmp/rktoolkit/x86py/bin/python3 \
      scripts/build_rkllm_x86.py --model models/nac_dequant_g64 \
      --dataset data/quant_dataset.json --dtype w4a16_g64 \
      --out models/nac_qwen1.5b_g64_nac.rkllm
"""
import argparse
import time

from rkllm.api import RKLLM


def main():
    ap = argparse.ArgumentParser(description="x86 toolkit build: dequant HF -> .rkllm")
    ap.add_argument("--model", required=True, help="gen-dequant 产出的 HF 模型目录")
    ap.add_argument("--dataset", required=True, help="校准样本 JSON")
    ap.add_argument("--dtype", default="w4a16_g64", help="如 w4a16_g64 / w4a16_g128 / w4a16")
    ap.add_argument("--algorithm", default="normal", help="normal=per-group min-max")
    ap.add_argument("--platform", default="rk3576")
    ap.add_argument("--optim", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    llm = RKLLM()
    t0 = time.time()
    ret = llm.load_huggingface(model=args.model, model_lora=None, device="cpu")
    if ret != 0:
        raise SystemExit(f"load_huggingface failed: {ret}")
    print(f"[build] load_huggingface ok ({time.time() - t0:.0f}s)", flush=True)

    t1 = time.time()
    ret = llm.build(do_quantization=True, quantized_dtype=args.dtype,
                    quantized_algorithm=args.algorithm, optimization_level=args.optim,
                    target_platform=args.platform, dataset=args.dataset,
                    num_npu_core=2)
    if ret != 0:
        raise SystemExit(f"build failed: {ret}")
    print(f"[build] build ok ({time.time() - t1:.0f}s)", flush=True)

    ret = llm.export_rkllm(args.out)
    if ret != 0:
        raise SystemExit(f"export_rkllm failed: {ret}")
    print(f"[build] OK -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
