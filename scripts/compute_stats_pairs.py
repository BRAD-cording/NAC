#!/usr/bin/env python3
"""统一重算论文全部配对统计（McNemar χ² 连续性修正 + bootstrap CI95），落盘 stats_pairs.json。

Table 3 数据唯一来源。配对前提：两边 records 顺序一致 —— 用 gold 序列逐位断言。
"""
import json, math, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

# (pair_name, records_a, records_b, axis, protocol_tag)
# records 文件为 list[{i,pred,gold,ok,gen}]；a 为对照组、b 为实验组（diff = b - a）
PAIRS = [
    # sim MMLU g64 (969)
    ("sim_mmlu_g64::nacv2_vs_nac",    "acc_mmlu_sim_g64.records.nac.json",     "acc_mmlu_sim_g64_nacv2.records.nacv2.json", "sim", "mmlu969/32tok"),
    ("sim_mmlu_g64::nacv2_vs_gptq",   "acc_mmlu_sim_g64.records.gptq.json",    "acc_mmlu_sim_g64_nacv2.records.nacv2.json", "sim", "mmlu969/32tok"),
    ("sim_mmlu_g64::nacv2_vs_minmax", "acc_mmlu_sim_g64.records.minmax.json",  "acc_mmlu_sim_g64_nacv2.records.nacv2.json", "sim", "mmlu969/32tok"),
    ("sim_mmlu_g64::nacv2_vs_fp16",   "acc_mmlu_sim_g64.records.fp16.json",    "acc_mmlu_sim_g64_nacv2.records.nacv2.json", "sim", "mmlu969/32tok"),
    ("sim_mmlu_g64::awq_vs_gptq",     "acc_mmlu_sim_g64.records.gptq.json",    "acc_mmlu_sim_g64_awq.records.awq.json",     "sim", "mmlu969/32tok"),
    ("sim_mmlu_g64::awq_vs_nacv2",    "acc_mmlu_sim_g64_nacv2.records.nacv2.json", "acc_mmlu_sim_g64_awq.records.awq.json", "sim", "mmlu969/32tok"),
    ("sim_mmlu_g64::awq_vs_minmax",   "acc_mmlu_sim_g64.records.minmax.json",  "acc_mmlu_sim_g64_awq.records.awq.json",     "sim", "mmlu969/32tok"),
    ("sim_mmlu_g64::awq_vs_fp16",     "acc_mmlu_sim_g64.records.fp16.json",    "acc_mmlu_sim_g64_awq.records.awq.json",     "sim", "mmlu969/32tok"),
    ("sim_mmlu_g64::nac_vs_minmax",   "acc_mmlu_sim_g64.records.minmax.json",  "acc_mmlu_sim_g64.records.nac.json",         "sim", "mmlu969/32tok"),
    # sim GSM8K g64 (200, 2-shot CoT/512)
    ("sim_gsm8k_g64::gptq_vs_minmax", "acc_gsm8k_sim_g64.records.minmax.json", "acc_gsm8k_sim_g64.records.gptq.json",       "sim", "gsm8k200/2shot512"),
    ("sim_gsm8k_g64::gptq_vs_nac",    "acc_gsm8k_sim_g64.records.nac.json",    "acc_gsm8k_sim_g64.records.gptq.json",       "sim", "gsm8k200/2shot512"),
    ("sim_gsm8k_g64::nacv2_vs_nac",   "acc_gsm8k_sim_g64.records.nac.json",    "acc_gsm8k_sim_g64_nacv2.records.nacv2.json","sim", "gsm8k200/2shot512"),
    ("sim_gsm8k_g64::nacv2_vs_gptq",  "acc_gsm8k_sim_g64.records.gptq.json",   "acc_gsm8k_sim_g64_nacv2.records.nacv2.json","sim", "gsm8k200/2shot512"),
    ("sim_gsm8k_g64::nacv2_vs_minmax","acc_gsm8k_sim_g64.records.minmax.json", "acc_gsm8k_sim_g64_nacv2.records.nacv2.json","sim", "gsm8k200/2shot512"),
    ("sim_gsm8k_g64::awq_vs_gptq",    "acc_gsm8k_sim_g64.records.gptq.json",   "acc_gsm8k_sim_g64_awq.records.awq.json",    "sim", "gsm8k200/2shot512"),
    ("sim_gsm8k_g64::awq_vs_nacv2",   "acc_gsm8k_sim_g64_nacv2.records.nacv2.json", "acc_gsm8k_sim_g64_awq.records.awq.json","sim", "gsm8k200/2shot512"),
    ("sim_gsm8k_g64::awq_vs_minmax",  "acc_gsm8k_sim_g64.records.minmax.json", "acc_gsm8k_sim_g64_awq.records.awq.json",    "sim", "gsm8k200/2shot512"),
    # sim MMLU g128 (969)
    ("sim_mmlu_g128::nac_vs_minmax",  "acc_mmlu_sim_g128.records.minmax.json", "acc_mmlu_sim_g128.records.nac.json",        "sim", "g128 mmlu969/32tok"),
    ("sim_mmlu_g128::gptq_vs_minmax", "acc_mmlu_sim_g128.records.minmax.json", "acc_mmlu_sim_g128.records.gptq.json",       "sim", "g128 mmlu969/32tok"),
    ("sim_mmlu_g128::gptq_vs_nac",    "acc_mmlu_sim_g128.records.nac.json",    "acc_mmlu_sim_g128.records.gptq.json",       "sim", "g128 mmlu969/32tok"),
    # 板端（旧注入叙事产物 —— F1 事故后仅存档，不随正文 Table 3 使用，标注 caveat）
    ("board_mmlu::nacinject_vs_base", "acc_mmlu_qwen1.5b_w4a16_zs.records.json", "acc_mmlu_qwen1.5b_g64nac_zs.records.json", "board", "STALE: 注入未生效构建, F1"),
    ("board_gsm8k::nacinject_vs_base","acc_gsm8k_qwen1.5b_w4a16_zs.records.json","acc_gsm8k_qwen1.5b_g64nac_zs.records.json", "board", "STALE: 注入未生效构建 + 基线512tok协议疑不一致, F1/F2"),
]


def load(p: Path):
    d = json.loads(p.read_text())
    if isinstance(d, dict):  # 兼容 {records:[...]} 包装
        d = d.get("records", d)
    return d


def ok_of(r):
    """GSM8K/板端 records 直接有 ok；MMLU sim records 判定 = pred_gen == gold（acc_gen 口径）"""
    if "ok" in r:
        return bool(r["ok"])
    return r.get("pred_gen") == r.get("gold")


def mcnemar_cc(b: int, c: int):
    if b + c == 0:
        return 1.0, 0.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # chi2(1 dof) sf = erfc(sqrt(chi2/2))，等价 scipy.stats.chi2.sf，免 scipy 依赖
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return p, chi2


def boot_ci(ok_a, ok_b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(ok_a, dtype=bool); b = np.asarray(ok_b, dtype=bool)
    d = b.astype(int) - a.astype(int)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = d[idx].mean(axis=1)
    return [round(float(x), 4) for x in np.percentile(diffs, [2.5, 97.5])]


def main():
    out, warnings = {}, []
    for name, fa, fb, axis, proto in PAIRS:
        pa, pb = R / fa, R / fb
        if not pa.exists() or not pb.exists():
            warnings.append(f"{name}: missing {fa if not pa.exists() else fb}")
            continue
        ra, rb = load(pa), load(pb)
        ga = [r.get("gold") for r in ra]; gb = [r.get("gold") for r in rb]
        n = min(len(ra), len(rb))
        if ga[:n] != gb[:n]:
            warnings.append(f"{name}: gold 序列前 {n} 位不一致 —— 位置配对前提存疑，已跳过")
            continue
        oka = [ok_of(r) for r in ra[:n]]
        okb = [ok_of(r) for r in rb[:n]]
        b_ = sum(1 for x, y in zip(oka, okb) if y and not x)   # only-b
        c_ = sum(1 for x, y in zip(oka, okb) if x and not y)   # only-a
        p, chi2 = mcnemar_cc(b_, c_)
        acc_a, acc_b = sum(oka) / n, sum(okb) / n
        out[name] = {
            "axis": axis, "protocol": proto, "n": n,
            "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4),
            "diff_pp": round((acc_b - acc_a) * 100, 2),
            "mcnemar_b_only_b": b_, "mcnemar_c_only_a": c_,
            "chi2_cc": round(chi2, 3), "p": float(f"{p:.3e}"),
            "ci95_diff_pp": boot_ci(oka, okb),
            "records_a": fa, "records_b": fb,
            "order_check": "gold-sequence-identical",
        }
    outp = R / "stats_pairs.json"
    outp.write_text(json.dumps({"generated": "scripts/compute_stats_pairs.py",
                                "pairs": out, "warnings": warnings},
                               indent=2, ensure_ascii=False))
    print(f"-> {outp}  ({len(out)} pairs, {len(warnings)} warnings)")
    for w in warnings:
        print("  WARN:", w)


if __name__ == "__main__":
    sys.exit(main())
