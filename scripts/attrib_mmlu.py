"""D4 归因实验：MMLU 精度异常低的原因定位。

固定 100 题，三种 prompt 配置对比（其余参数完全一致）：
  raw2shot : 现行评测格式（通用 2-shot FEWSHOT + raw completion）
  chat2shot: 同样内容包进 Qwen chat template（<|im_start|>...<|im_end|>）
  raw0shot : 无 fewshot 纯 completion

若 chat2shot 显著更高 → 精度损失主要是评测格式不匹配（Instruct 模型对 raw 格式退化）；
若三者接近且都低 → 真实量化损伤，论文动机成立。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.accuracy import (FEWSHOT, LETTERS, extract_choice,  # noqa: E402
                               load_mmlu, mmlu_prompt)
from src.runtime.llm_client import RKLLMClient  # noqa: E402

CHAT_TPL = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{content}<|im_end|>\n"
            "<|im_start|>assistant\n")


def wrap(kind: str, row: dict) -> str:
    content = mmlu_prompt(row)
    if kind == "raw2shot":
        return content
    if kind == "chat2shot":
        return CHAT_TPL.format(content=content)
    if kind == "raw0shot":
        q = f"{row['question']}\n" + "\n".join(
            f"{l}. {c}" for l, c in zip(LETTERS, row["choices"]))
        return f"{q}\nAnswer:"
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct_RK3576_w4a16.rkllm")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default="results/attrib_mmlu.json")
    args = ap.parse_args()

    rows = load_mmlu(args.num_samples, seed=2026)  # 固定新 seed，避免与主评测重叠
    results = {"model": args.model, "num_samples": len(rows), "configs": {}}
    with RKLLMClient(args.model, max_new_tokens=args.max_new_tokens) as client:
        client.generate("Hello")  # warmup
        for kind in ("raw2shot", "chat2shot", "raw0shot"):
            correct = none_pred = 0
            samples = []
            for i, row in enumerate(rows):
                out = client.generate(wrap(kind, row))
                pred = extract_choice(out["text"])
                gold = LETTERS[row["answer"]]
                ok = pred == gold
                correct += int(ok)
                none_pred += int(pred is None)
                if i < 3:  # 每配置留 3 条样本肉眼核查
                    samples.append({"gold": gold, "pred": pred,
                                    "gen": out["text"][:150]})
            acc = correct / len(rows)
            results["configs"][kind] = {
                "acc": round(acc, 4), "none_pred_rate": round(none_pred / len(rows), 4),
                "samples": samples}
            print(f"[attrib] {kind:9s} acc={acc:.4f} none_pred={none_pred}/{len(rows)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"[attrib] -> {args.out}")


if __name__ == "__main__":
    main()
