#!/usr/bin/env python3
"""证据账本：论文/评审中每个数字 ↔ 落盘文件 的自动核验。

用法（在 nac_paper 仓库根目录）：
    .venv/bin/python analysis/check_claims.py [--out results/claims_check.md]

设计原则：
- 只读 results/*.json，不调用模型、不重算推理；
- 每条 claim 声明 (来源文件, 取值路径, 期望值, 容差)，任何一条对不上即 FAIL；
- 输出 Markdown 表格，可直接作为论文附录或 artifacts 的一部分。

背景：2026-10-29 的评审发现"文档数字与归档记录不一致""协议字段缺失"两类问题，
本脚本把"数字可追溯"变成可执行检查，防止 R5/重跑后再次漂移。
"""
import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name):
    with io.open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def get(obj, path):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


# (claim_id, 描述, 文件, 路径, 期望值, 容差)
CLAIMS = [
    # --- 板端 32 vs 512 截断伪影 ---
    ("B1", "vendor GSM8K @32tok == 0.0000", "acc_gsm8k_qwen1.5b_w4a16_zs32.json", "gsm8k_acc", 0.0, 1e-9),
    ("B2", "vendor GSM8K @32tok 记录 max_new_tokens == 32", "acc_gsm8k_qwen1.5b_w4a16_zs32.json", "max_new_tokens", 32, 0),
    ("B3", "vendor GSM8K @512tok == 0.5550", "acc_gsm8k_qwen1.5b_w4a16_zs.json", "gsm8k_acc", 0.555, 1e-9),
    ("B4", "NAC 注入 GSM8K @32tok == 0.0050（伪影，非退化）", "acc_gsm8k_qwen1.5b_g64nac_zs.json", "gsm8k_acc", 0.005, 1e-9),
    ("B5", "E3 minmax 注入 GSM8K @32tok == 0.0050", "acc_gsm8k_qwen1.5b_g64mmctrl_zs.json", "gsm8k_acc", 0.005, 1e-9),
    ("B6", "本地构建 GSM8K @512tok == 0.5350（与 @32 的 0.0150 对照）",
     "acc_gsm8k_qwen1.5b_plain_x86_zs512.json", "gsm8k_acc", 0.535, 1e-9),
    ("B7", "本地构建 @512 记录 max_new_tokens == 512",
     "acc_gsm8k_qwen1.5b_plain_x86_zs512.json", "max_new_tokens", 512, 0),
    ("B8", "g64 构建（曾被标为 NAC 崩溃）GSM8K @512tok == 0.5650",
     "acc_gsm8k_q1bd8_zs512.json", "gsm8k_acc", 0.565, 1e-9),
    ("B9", "g64 构建 @512 的 answer-is 指标同为 0.5650（无续写污染）",
     "acc_gsm8k_q1bd8_zs512.json", "gsm8k_acc_answeris", 0.565, 1e-9),
    # --- sim 轴 MMLU ---
    ("S1", "sim fp16 MMLU 0.4634", "acc_mmlu_sim_g64.json", "acc_gen", 0.4634, 1e-9),
    ("S2", "sim minmax MMLU 0.4675", "acc_mmlu_sim_g64.full.json", "configs.minmax.acc_gen", 0.4675, 1e-9),
    ("S3", "sim GPTQ MMLU 0.4520", "acc_mmlu_sim_g64.full.json", "configs.gptq.acc_gen", 0.4520, 1e-9),
    ("S4", "sim NAC-v1 MMLU 0.3932", "acc_mmlu_sim_g64.full.json", "configs.nac.acc_gen", 0.3932, 1e-9),
    ("S5", "sim NAC-v2 MMLU 0.4469", "acc_mmlu_sim_g64_nacv2.full.json", "configs.nacv2.acc_gen", 0.4469, 1e-9),
    ("S6", "sim AWQ MMLU 0.4396", "acc_mmlu_sim_g64_awq.full.json", "configs.awq.acc_gen", 0.4396, 1e-9),
    # --- sim 轴 GSM8K ---
    ("G1", "sim minmax GSM8K 0.075 (n=200, 512tok)", "acc_gsm8k_sim_g64.full.json", "configs.minmax.acc", 0.075, 1e-9),
    ("G2", "sim GPTQ GSM8K 0.305 (n=200)", "acc_gsm8k_sim_g64.full.json", "configs.gptq.acc", 0.305, 1e-9),
    ("G3", "sim NAC-v1 GSM8K 0.070 (n=200)", "acc_gsm8k_sim_g64.full.json", "configs.nac.acc", 0.070, 1e-9),
    ("G4", "sim NAC-v2 GSM8K 0.150 (n=200)", "acc_gsm8k_sim_g64_nacv2.full.json", "configs.nacv2.acc", 0.150, 1e-9),
    ("G5", "sim AWQ GSM8K 0.000 且 n==64", "acc_gsm8k_sim_g64_awq.json", "num_evaluated", 64, 0),
    # --- PPL ---
    ("P1", "g64 PPL fp16 11.0747", "ppl_g64_64seq.json", "ppl.fp16", 11.0747, 1e-4),
    ("P2", "g64 PPL minmax 12.2251", "ppl_g64_64seq.json", "ppl.minmax", 12.2251, 1e-4),
    ("P3", "g64 PPL nac 11.893", "ppl_g64_64seq.json", "ppl.nac", 11.893, 1e-4),
    ("P4", "g64 PPL gptq 12.1679", "ppl_g64_64seq.json", "ppl.gptq", 12.1679, 1e-4),
    ("P5", "g128 PPL nac 12.1644 < gptq 12.5032 < minmax 12.5436",
     "ppl_g128_64seq.json", "ppl.nac", 12.1644, 1e-4),
    # --- 统计 ---
    ("T1", "NAC vs minmax g64 MMLU p=3.285e-9", "stats_pairs.json",
     "pairs.sim_mmlu_g64::nac_vs_minmax.p", 3.285e-9, 1e-12),
    ("T2", "NAC vs minmax g128 MMLU +6.71pp p=7.895e-9", "stats_pairs.json",
     "pairs.sim_mmlu_g128::nac_vs_minmax.p", 7.895e-9, 1e-12),
    ("T3", "NAC-v2 vs NAC MMLU p=2.138e-5", "stats_pairs.json",
     "pairs.sim_mmlu_g64::nacv2_vs_nac.p", 2.138e-5, 1e-15),
    ("T4", "NAC-v2 vs GPTQ MMLU 平手 p=0.7161", "stats_pairs.json",
     "pairs.sim_mmlu_g64::nacv2_vs_gptq.p", 0.7161, 1e-9),
    ("T5", "NAC-v2 vs NAC GSM8K p=0.01242", "stats_pairs.json",
     "pairs.sim_gsm8k_g64::nacv2_vs_nac.p", 0.01242, 1e-12),
    ("T6", "NAC-v2 vs GPTQ GSM8K p=1.225e-4", "stats_pairs.json",
     "pairs.sim_gsm8k_g64::nacv2_vs_gptq.p", 0.0001225, 1e-15),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/claims_check.md")
    args = ap.parse_args()
    rows, fails = [], 0
    for cid, desc, fname, path, want, tol in CLAIMS:
        try:
            got = get(load(fname), path)
            ok = abs(float(got) - float(want)) <= max(tol, abs(float(want)) * 1e-9)
        except Exception as e:  # noqa: BLE001
            got, ok, desc = f"ERR {type(e).__name__}: {e}", False, desc
        fails += (not ok)
        rows.append((cid, desc, fname, str(got), str(want), "PASS" if ok else "FAIL"))

    lines = ["| ID | 断言 | 来源 | 实测 | 期望 | 结果 |", "|---|---|---|---|---|---|"]
    lines += [f"| {c} | {d} | `{f}` | {g} | {w} | {r} |" for c, d, f, g, w, r in rows]
    md = "\n".join(lines) + f"\n\n**{len(rows) - fails}/{len(rows)} PASS**\n"
    (ROOT / args.out).write_text(md, encoding="utf-8")
    print(md)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
