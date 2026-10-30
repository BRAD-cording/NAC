#!/usr/bin/env python3
"""截断协议下的 sim 对比：fp16 / GPTQ / NAC-v2（存在几个算几个）。

对每个配置输出 acc(末数字)、acc(answer-is)、截断率、长度中位数；
并以 fp16 为参照做配对 McNemar（按题对齐，先断言 gold 序列一致）。

用法: .venv/bin/python analysis/sim_compare.py [--out results/q1_sim_compare.md]
"""
import argparse
import glob
import io
import json
import math
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def mcnemar(a, b):
    n = min(len(a), len(b))
    if [x["gold"] for x in a][:n] != [x["gold"] for x in b][:n]:
        return None
    oa = [x["ok"] for x in a][:n]
    ob = [x["ok"] for x in b][:n]
    b01 = sum(1 for i in range(n) if oa[i] and not ob[i])
    b10 = sum(1 for i in range(n) if not oa[i] and ob[i])
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10) if (b01 + b10) else 0.0
    p = math.erfc(math.sqrt(chi2 / 2))
    return n, sum(oa) / n, sum(ob) / n, b01, b10, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/q1_sim_compare.md")
    args = ap.parse_args()

    rows, recs = [], {}
    for cfg in ("fp16", "gptq", "nacv2"):
        sp = RES / f"acc_gsm8k_sim_g64_stop_{cfg}.json"
        rp = RES / f"acc_gsm8k_sim_g64_stop_{cfg}.records.{cfg}.json"
        if not sp.exists() or not rp.exists():
            continue
        d, r = load(sp), load(rp)
        n = len(r)
        acc = sum(1 for x in r if x.get("ok")) / n
        ai = sum(1 for x in r if x.get("ok_answeris")) / n
        tr = sum(1 for x in r if x.get("stop_truncated")) / n
        lens = [x.get("gen_raw_chars") or len(x.get("gen") or "") for x in r]
        rows.append((cfg, n, acc, ai, tr, statistics.median(lens)))
        recs[cfg] = r

    lines = ["# sim GSM8K：截断协议下的对比", "",
             "| config | n | acc (last-num) | acc (answer-is) | truncated | median chars |",
             "|---|---|---|---|---|---|"]
    for cfg, n, acc, ai, tr, med in rows:
        lines.append(f"| {cfg} | {n} | {acc:.4f} | {ai:.4f} | {tr:.2f} | {med:.0f} |")

    if "fp16" in recs:
        lines += ["", "**配对 McNemar（相对 fp16）**", "",
                  "| comparison | n | fp16 | other | Δ (pp) | b | c | p |",
                  "|---|---|---|---|---|---|---|---|"]
        for cfg, *_ in rows:
            if cfg == "fp16":
                continue
            m = mcnemar(recs["fp16"], recs[cfg])
            if not m:
                lines.append(f"| fp16 vs {cfg} | — | — | — | — | — | — | gold 序列不一致 |")
                continue
            n, aa, ab, b01, b10, p = m
            lines.append(f"| fp16 vs {cfg} | {n} | {aa:.4f} | {ab:.4f} | {(ab-aa)*100:+.2f} | {b01} | {b10} | {p:.3g} |")

    lines += ["", "旧口径（不截断、末数字）参考：GPTQ 0.305 / min-max 0.075 / NAC-v1 0.070 / NAC-v2 0.150。",
              "若新口径下 fp16 与各量化方法差距 <10pp，则 Finding 1 需改写为「协议放大」而非「校准质量」。"]
    (ROOT / args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
