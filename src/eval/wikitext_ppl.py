"""Wikitext-2 PPL（x86 校准机侧，经 rkllm_toolkit get_logits；方案 D4/D11）。

用途：导出前后快速对比"校准期精度"（GPU/模拟量化 vs NAC），
板上精度由 accuracy.py 的 MMLU/GSM8K 给出。需 rkllm_toolkit（x86 wheel）。

用法（x86）：
  python -m src.eval.wikitext_ppl --model <hf_dir_or_rkllm_input> --dtype w4a16 \
      --dataset data/quant_dataset.json --out results/ppl_x86.json
"""
import argparse
import json
from datetime import datetime
from pathlib import Path


def eval_ppl(modelpath: str, dtype: str, target_platform: str, seqlen: int = 512) -> float:
    import torch
    from torch import nn
    from tqdm import tqdm

    from rkllm.api import RKLLM
    from datasets import load_dataset
    from transformers import AutoTokenizer

    llm = RKLLM()
    if llm.load_huggingface(model=modelpath, model_lora=None, device="cpu") != 0:
        raise SystemExit("load_huggingface failed")
    dataset = str(Path(__file__).resolve().parents[2] / "data" / "quant_dataset.json")
    if llm.build(do_quantization=dtype != "fp16", quantized_dtype=dtype if dtype != "fp16" else "w8a8",
                 quantized_algorithm="normal", target_platform=target_platform,
                 dataset=dataset) != 0:
        raise SystemExit("build failed")

    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    nsamples = testenc.numel() // seqlen
    nlls = []
    for i in tqdm(range(nsamples), desc="wikitext ppl"):
        batch = testenc[:, i * seqlen: (i + 1) * seqlen]
        logits = llm.get_logits({"input_ids": batch})
        if logits is None:
            raise SystemExit("get_logits failed")
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)),
                                     shift_labels.view(-1))
        nlls.append(loss.float() * seqlen)
    return float(torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="w4a16", choices=["w4a16", "w8a8", "fp16"])
    ap.add_argument("--target-platform", default="rk3576")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ppl = eval_ppl(args.model, args.dtype, args.target_platform)
    out = Path(args.out or f"results/ppl_x86_{datetime.now():%Y%m%d_%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "dtype": args.dtype,
                               "wikitext2_ppl": round(ppl, 3)}, indent=2))
    print(f"[ppl] wikitext-2 ppl = {ppl:.3f} -> {out}")


if __name__ == "__main__":
    main()
