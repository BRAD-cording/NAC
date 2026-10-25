"""协议对比实验：首 token 字母匹配 vs 生成式 MMLU 评测（同题目子集）。

背景：rkllm-runtime 1.2.3（RK3576 build）的 RKLLM_INFER_GET_LOGITS 模式不通过
回调返回 logits（实测 mode=2 退化为 generate 且 r.logits 恒空，见
results/protocol_notes.md）。因此采用最强的可用替代协议：

  first-token：greedy 生成 1 个 token，若首 token 属于 {A,B,C,D} 及其
  前导空格变体（token 级匹配，非正则），则预测该字母，否则 pred=None。

与生成式协议（32 token + 正则抽取）相比：单 token、无格式幻觉、协议损耗下限。
输出 results/acc_mmlu_firsttok_{model}.json（与生成式评测同 969 题子集）。
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.accuracy import LETTERS, load_mmlu, mmlu_prompt  # noqa: E402
from src.runtime.llm_client import RKLLMClient  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

TOK_PATH = Path(__file__).resolve().parents[1] / "models" / "tokenizer" / "tokenizer.json"


def build_letter_map() -> dict[int, str]:
    """token_id -> 字母（含 'A' 与 ' A' 变体；多 token id 冲突时保留先到的）。"""
    tok = Tokenizer.from_file(str(TOK_PATH))
    m: dict[int, str] = {}
    for L in LETTERS:
        for tid in tok.encode(L, add_special_tokens=False).ids + \
                tok.encode(" " + L, add_special_tokens=False).ids:
            m.setdefault(tid, L)
    return m


def main():
    ap = argparse.ArgumentParser(description="MMLU first-token protocol eval")
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="调试：只跑前 N 题")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    letter_map = build_letter_map()
    rows = load_mmlu(args.num_samples)
    if args.limit:
        rows = rows[: args.limit]
    stem = Path(args.model).stem
    out_path = Path(args.out or
                    f"results/acc_mmlu_firsttok_{stem}_{datetime.now():%Y%m%d_%H%M%S}.json")

    results = {"model": args.model, "task": "mmlu_firsttoken", "num_samples": len(rows),
               "protocol": ("greedy 1-token letter match (token-id level, "
                            "A/B/C/D with space variants); GET_LOGITS mode "
                            "unavailable in rkllm-runtime 1.2.3 RK3576 build")}
    correct = none_pred = 0
    ms_total = 0.0
    records = []

    with RKLLMClient(args.model, max_new_tokens=1) as client:
        client.generate("Hello")  # warmup
        t_start = time.perf_counter()
        for i, row in enumerate(rows):
            gold = LETTERS[row["answer"]]
            try:
                out = client.generate(mmlu_prompt(row))
            except RuntimeError as e:
                print(f"[firsttok] #{i} error: {e}")
                records.append({"i": i, "gold": gold, "pred": None, "ok": False,
                                "err": str(e)})
                continue
            ms_total += out["total_ms"]
            tids = out.get("token_ids") or []
            pred = letter_map.get(tids[0]) if tids else None
            ok = pred == gold
            correct += int(ok)
            none_pred += int(pred is None)
            records.append({"i": i, "pred": pred, "gold": gold, "ok": ok,
                            "first_token": tids[0] if tids else None,
                            "ms": out["total_ms"]})
            if (i + 1) % 100 == 0:
                print(f"[firsttok] {i + 1}/{len(rows)} acc={correct / (i + 1):.4f}")
        wall_s = time.perf_counter() - t_start

    n = len(rows)
    acc = correct / max(1, n)
    results["mmlu_acc"] = round(acc, 4)
    results["none_pred_rate"] = round(none_pred / max(1, n), 4)
    results["avg_ms_per_q"] = round(ms_total / max(1, n), 1)
    results["wall_s"] = round(wall_s, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    out_path.with_suffix(".records.json").write_text(json.dumps(records, ensure_ascii=False))
    print(f"[firsttok] acc = {acc:.4f} ({correct}/{n}), none_pred={none_pred}, "
          f"avg {results['avg_ms_per_q']} ms/q -> {out_path}")


if __name__ == "__main__":
    main()
