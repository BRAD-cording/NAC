"""校准/评测数据工具。

- build_calib_dataset: Wikitext-2 val → toolkit 校准 JSON（{"input","target"}）
- load_calib_samples:  供 calibrate.py 使用，返回 token id tensor 列表
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _get_wikitext(tokenizer):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    return "\n\n".join(ds["text"])


def build_calib_dataset(num_samples: int, seqlen: int, tokenizer, out: str,
                        seed: int = 42) -> list[dict]:
    """Wikitext-2 val 切块 → rkllm toolkit 校准 JSON。"""
    import random

    import torch
    text = _get_wikitext(tokenizer)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_full = ids.numel() // seqlen
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(n_full), min(num_samples * 2, n_full)))
    entries = []
    for i in picks:
        chunk = ids[i * seqlen: (i + 1) * seqlen]
        # 前 3/4 作 input（prompt），后 1/4 作 target（续写），符合 toolkit 样例格式
        split = seqlen * 3 // 4
        inp = tokenizer.decode(chunk[:split])
        tgt = tokenizer.decode(chunk[split:])
        entries.append({"input": inp, "target": tgt})
        if len(entries) >= num_samples:
            break
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(entries, ensure_ascii=False, indent=1))
    print(f"[data] {len(entries)} calib samples -> {out}")
    return entries


def load_calib_samples(calib_json: str, num_samples: int, tokenizer,
                       seqlen: int) -> list[torch.Tensor]:
    """读取校准 JSON 并编码为 (1, seqlen) int64 张量列表。"""
    import torch

    entries = json.loads(Path(calib_json).read_text())
    if not entries:
        raise SystemExit(f"{calib_json} 为空，请先运行 python -m src.utils.data")
    samples = []
    for e in entries[:num_samples]:
        ids = tokenizer(e["input"] + e["target"], return_tensors="pt",
                        truncation=True, max_length=seqlen).input_ids
        if ids.shape[1] == seqlen:
            samples.append(ids)
    if not samples:
        raise SystemExit("校准样本编码后均不足 seqlen，请增大语料或减小 seqlen")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--out", default="data/quant_dataset.json")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    build_calib_dataset(args.num_samples, args.seqlen, tok, args.out)


if __name__ == "__main__":
    main()
