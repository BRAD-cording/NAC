#!/usr/bin/env python3
"""sim 重跑汇总：results/q1_sim_summary.md

对 results/acc_gsm8k_sim_g64_stop_*.json 及其 records 计算：
  主指标（末数字）、answer-is 指标、直接停止率（stop_truncated=False 且长度未顶格）、
  截断率、跨题续写残留率、平均长度。
"""
import glob
import io
import json
import os
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CROSS = re.compile(r"\n\s*Question:")


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []
    for f in sorted(glob.glob(str(RES / "acc_gsm8k_sim_g64_stop_*.json"))):
        d = load(f)
        stem = Path(f).stem
        cfg = stem.replace("acc_gsm8k_sim_g64_stop_", "")
        rec_path = Path(f).with_suffix(f".records.{cfg}.json")
        recs = load(rec_path) if rec_path.exists() else []
        if not recs:
            continue
        n = len(recs)
        acc = sum(1 for r in recs if r.get("ok")) / n
        acc_ai = sum(1 for r in recs if r.get("ok_answeris")) / n
        trunc = sum(1 for r in recs if r.get("stop_truncated")) / n
        lens = [r.get("gen_raw_chars") or len(r.get("gen") or "") for r in recs]
        cross = sum(1 for r in recs if CROSS.search(r.get("gen") or "")) / n
        rows.append((cfg, n, acc, acc_ai, trunc, cross, statistics.median(lens)))

    lines = ["# sim 重跑汇总（停止串/截断后）", "",
             "| config | n | acc (last-num) | acc (answer-is) | truncated | residual cross-q | median chars |",
             "|---|---|---|---|---|---|---|"]
    for cfg, n, acc, acc_ai, trunc, cross, med in rows:
        lines.append(f"| {cfg} | {n} | {acc:.4f} | {acc_ai:.4f} | {trunc:.2f} | {cross:.2f} | {med:.0f} |")
    lines += ["", "判据：`residual cross-q` 应接近 0（截断生效）；"
                  "`acc` 与 `acc(answer-is)` 差距应显著小于旧数据（续写污染已移除）。"]
    out = RES / "q1_sim_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
