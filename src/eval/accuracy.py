"""板上生成式精度评测：MMLU(5-shot, 1k 子采样) / GSM8K(8-shot, 200 子采样)。

方式：经 RKLLMClient 生成答案 → 正则抽取 → 打分。
输出 results/acc_{task}_{model}.json，包含逐题记录（论文附录用）。
"""
import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

from ..runtime.llm_client import RKLLMClient

LETTERS = ["A", "B", "C", "D"]

FEWSHOT = ("The following are multiple choice questions (with answers) about biology.\n\n"
           "Which organelle produces ATP?\nA. Ribosome\nB. Mitochondrion\nC. Nucleus\n"
           "D. Golgi\nAnswer: B\n\n"
           "What base pairs with adenine in DNA?\nA. Guanine\nB. Uracil\nC. Thymine\n"
           "D. Cytosine\nAnswer: C\n\n")

GSM8K_TMPL = """Question: {q}
Answer:"""

GSM8K_FEWSHOT = """Question: Tom has 3 apples and buys 2 more. How many apples does he have?
Answer: 3 + 2 = 5. The answer is 5.

Question: A car travels 60 km per hour for 2 hours. How far does it go?
Answer: 60 * 2 = 120. The answer is 120.

"""


def load_mmlu(num_samples: int, seed: int = 42):
    """MMLU test 集（HF: cais/mmlu, 'all'）分层子采样。"""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subj: dict[str, list] = {}
    for row in ds:
        by_subj.setdefault(row["subject"], []).append(row)
    rng = random.Random(seed)
    per_subj = max(1, num_samples // len(by_subj))
    picked = []
    for subj in sorted(by_subj):
        pool = by_subj[subj]
        rng.shuffle(pool)
        picked += pool[:per_subj]
    rng.shuffle(picked)
    return picked[:num_samples]


def mmlu_prompt(row: dict, fewshot_num: int = 2) -> str:
    q = f"{row['question']}\n" + "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, row["choices"]))
    # zero-shot（fewshot_num=0）时给出字母作答指令，提示直接输出选项字母
    if fewshot_num <= 0:
        return f"{q}\nAnswer with a single letter (A, B, C or D):"
    return FEWSHOT + f"{q}\nAnswer:"


def extract_choice(text: str) -> str | None:
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABCD])\b", text.strip()[:8])
    return m.group(1).upper() if m else None


def load_gsm8k(num_samples: int, seed: int = 42):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [ds[i] for i in idx[:num_samples]]


def extract_number(text: str) -> float | None:
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return float(m[-1]) if m else None


ANSWER_IS_RE = re.compile(r"answer\s+(?:is|:|=)\s*\$?\s*(-?\d[\d,]*\.?\d*)")


def extract_answer_is(text: str) -> float | None:
    """answer-is 优先抽取：首个 "answer is/:" 后数字，无命中回退 extract_number。

    背景（2026-10-29 m1 复核，results/extractor_recheck.json）：非终止生成下
    取末数字会从续写文本错位取数，为全配置一阶混杂；本函数作为 dual-metric
    的第二指标使用，不替换 extract_number 默认路径（保持历史可比性）。
    """
    m = ANSWER_IS_RE.findall(text.replace(",", ""))
    if m:
        return float(m[0])
    return extract_number(text)


def eval_task(client: RKLLMClient, rows: list, kind: str, log_path: Path | None = None,
              fewshot_num: int = 2, max_new_tokens: int = 32):
    correct = 0
    records = []
    for i, row in enumerate(rows):
        if kind == "mmlu":
            prompt = mmlu_prompt(row, fewshot_num)
            gold = LETTERS[row["answer"]]
        else:
            pre = GSM8K_FEWSHOT if fewshot_num > 0 else ""
            prompt = pre + GSM8K_TMPL.format(q=row["question"])
            gold = row["answer"].split("####")[-1].strip().replace(",", "")
        try:
            out = client.generate(prompt)
        except RuntimeError as e:
            print(f"[acc] #{i} run error: {e}")
            continue
        text = out["text"].strip()
        if kind == "mmlu":
            pred = extract_choice(text)
            pred_as, ok_as = pred, (pred == gold)
        else:
            pred = extract_number(text)
            pred_as = extract_answer_is(text)
        ok = (pred == gold) if kind == "mmlu" else (
            pred is not None and abs(pred - float(gold)) < 1e-4)
        if kind != "mmlu":
            ok_as = (pred_as is not None and abs(pred_as - float(gold)) < 1e-4)
        correct += int(ok)
        n_tok = out.get("n_tokens") or 0
        # 归档全文（>=4000 字符可覆盖 512-token 生成）与终止信息：
        # 旧版 gen[:600] 导致事后无法复核抽取规则、无法区分"自然终止/截断"（F2/m1 教训）
        records.append({"i": i, "pred": str(pred), "gold": str(gold),
                        "ok": ok,
                        "pred_ansis": str(pred_as) if kind != "mmlu" else None,
                        "ok_ansis": bool(ok_as) if kind != "mmlu" else None,
                        "gen": text[:4000], "gen_chars": len(text),
                        "n_tokens": n_tok,
                        "hit_cap": n_tok >= max_new_tokens,
                        "decode_tps": out.get("decode_tps"),
                        "ttft_ms": out["ttft_ms"]})
        if (i + 1) % 50 == 0:
            print(f"[acc] {kind} {i + 1}/{len(rows)} acc={correct / (i + 1):.4f}")
    if log_path:
        log_path.write_text(json.dumps(records, indent=1, ensure_ascii=False))
    return correct / max(1, len(rows)), records


def main():
    ap = argparse.ArgumentParser(description="On-device accuracy eval")
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["mmlu", "gsm8k", "both"], default="mmlu")
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--fewshot-num", type=int, default=2,
                    help="few-shot 示例数；0 = zero-shot（3B 预转模型长 few-shot 会复述示例，实测见 BOARD_NOTES）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stem = Path(args.model).stem
    out_path = Path(args.out or f"results/acc_{args.task}_{stem}_{datetime.now():%Y%m%d_%H%M%S}.json")
    results = {"model": args.model, "task": args.task, "num_samples": args.num_samples,
               "fewshot_num": args.fewshot_num, "max_new_tokens": args.max_new_tokens,
               "extractor": ["last_number", "answer_is"], "gen_archive_chars": 4000}

    with RKLLMClient(args.model, max_new_tokens=args.max_new_tokens) as client:
        if args.task in ("mmlu", "both"):
            rows = load_mmlu(args.num_samples)
            acc, recs = eval_task(client, rows, "mmlu",
                                  out_path.with_suffix(".records.json"),
                                  fewshot_num=args.fewshot_num,
                                  max_new_tokens=args.max_new_tokens)
            results["mmlu_acc"] = round(acc, 4)
            print(f"[acc] MMLU acc = {acc:.4f}")
        if args.task in ("gsm8k", "both"):
            rows = load_gsm8k(min(args.num_samples, 200))
            acc, recs = eval_task(client, rows, "gsm8k",
                                  out_path.with_suffix(".records.json"),
                                  fewshot_num=args.fewshot_num,
                                  max_new_tokens=args.max_new_tokens)
            results["gsm8k_acc"] = round(acc, 4)
            results["gsm8k_acc_answeris"] = round(
                sum(1 for r in recs if r.get("ok_ansis")) / max(1, len(recs)), 4)
            print(f"[acc] GSM8K acc = {acc:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[acc] -> {out_path}")


if __name__ == "__main__":
    main()
