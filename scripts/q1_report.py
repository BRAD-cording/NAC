#!/usr/bin/env python3
"""板端 512-token 复测结果汇总：生成 results/q1_board512_summary.md。

对 results/acc_gsm8k_*_zs512.json + 同名 .records.json 计算：
  acc（末数字，主指标）、acc_answeris、hit_cap 率、EOS 率、token 中位数、
  字符中位数、跨题续写率（全文中出现 "\\nQuestion:" 的比例）。

用法：.venv/bin/python scripts/q1_report.py
"""
import glob
import hashlib
import io
import json
import os
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CROSS = re.compile(r"\n\s*Question:")


def md5(path, limit=None):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []
    for f in sorted(glob.glob(str(RESULTS / "acc_gsm8k_*_zs512.json"))):
        d = load(f)
        stem = Path(f).stem  # acc_gsm8k_<tag>_zs512
        tag = stem.replace("acc_gsm8k_", "").replace("_zs512", "")
        rec_path = Path(f).with_suffix(".records.json")
        recs = load(rec_path) if rec_path.exists() else []
        n = len(recs) or d.get("gsm8k_acc") and 200
        acc = d.get("gsm8k_acc")
        ans = d.get("gsm8k_acc_answeris")
        if recs:
            has_tok = "n_tokens" in recs[0]
            has_cap = "hit_cap" in recs[0]
            ntok = [r.get("n_tokens") or 0 for r in recs] if has_tok else []
            chars = [r.get("gen_chars") or len(r.get("gen") or "") for r in recs]
            cap = sum(1 for r in recs if r.get("hit_cap")) if has_cap else None
            cross = sum(1 for r in recs if CROSS.search(r.get("gen") or ""))
            n = len(recs)
        else:
            ntok = chars = []
            cap = None
            cross = 0
        model = Path(d.get("model", "")).name
        mp = ROOT / d.get("model", "")
        m = md5(mp)[:12] if mp.exists() else "?"
        rows.append({
            "tag": tag, "model": model, "md5": m, "n": n,
            "acc": acc, "acc_ansis": ans,
            "hit_cap": (cap / n) if (cap is not None and n) else float("nan"),
            "mean_tok": statistics.mean(ntok) if ntok else float("nan"),
            "med_chars": statistics.median(chars) if chars else float("nan"),
            "cross_q": cross / n if n else float("nan"),
        })

    lines = ["# 板端 512-token 复测汇总", "",
             "| tag | model | md5 | n | acc(last-num) | acc(answer-is) | hit_cap | mean tok | med chars | cross-q |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        hc = "n/a" if r["hit_cap"] != r["hit_cap"] else f"{r['hit_cap']:.3f}"
        mt = "n/a" if r["mean_tok"] != r["mean_tok"] else f"{r['mean_tok']:.1f}"
        lines.append(
            f"| {r['tag']} | {r['model']} | `{r['md5']}…` | {r['n']} | "
            f"{r['acc']} | {r['acc_ansis']} | {hc} | {mt} | "
            f"{r['med_chars']:.0f} | {r['cross_q']:.2f} |")
    lines += ["", "判据：`cross-q` 与 `hit_cap` 都低说明模型能在答案后停止；"
                  "`acc` 与 `acc(answer-is)` 接近说明末数字抽取没有被续写污染。"]
    out = RESULTS / "q1_board512_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
