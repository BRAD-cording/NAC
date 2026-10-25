#!/usr/bin/env python3
"""汇总 results/*.json → 论文表格（markdown + latex）。

文件命名约定（实际产出，glob 自动发现，排除 smoke/records/log）：
  acc_mmlu_<tag>.json            生成式 MMLU（mmlu_acc）
  acc_gsm8k_<tag>.json           GSM8K（gsm8k_acc）
  acc_mmlu_firsttok_<tag>.json   首 token 协议 MMLU（mmlu_acc）
  perf_<tag>*.json               性能（ttft/decode/power）
  val_nac_<tag>.json             NAC 模拟量化 PPL 对比
  route_threshold_*.json         prefill 路由阈值扫描
"""
import argparse
import json
from pathlib import Path


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def fmt(v, nd=2):
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def latest(results: Path, pattern: str) -> Path | None:
    cands = sorted(results.glob(pattern), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def discover_tags(results: Path, prefix: str) -> list[str]:
    """acc_mmlu_<tag>.json → tag（排除 smoke/firsttok/records）。"""
    tags = set()
    for p in results.glob(f"{prefix}_*.json"):
        tag = p.stem[len(prefix) + 1:]
        if "smoke" in tag or "firsttok" in tag or "records" in tag:
            continue
        tags.add(tag)
    return tags


def perf_row(p: dict, plen: int) -> dict:
    for pt in p.get("points", []):
        if pt.get("prompt_len_target") == plen:
            return pt
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/tables.md")
    args = ap.parse_args()
    rd = Path(args.results_dir)

    lines = ["# 论文表格汇总（自动生成）", ""]

    # ---- Table 1: Accuracy ----
    lines += ["## Table 1: Accuracy (MMLU / GSM8K, RKLLM W4A16 on NPU)", "",
              "| 配置 | MMLU (生成式) | MMLU (首token) | GSM8K |", "|---|---|---|---|"]
    tags = discover_tags(rd, "acc_mmlu") | discover_tags(rd, "acc_gsm8k")
    for tag in sorted(tags):
        mm = load(latest(rd, f"acc_mmlu_{tag}.json") or Path("/dev/null"))
        mmft = load(latest(rd, f"acc_mmlu_firsttok_{tag}.json") or Path("/dev/null"))
        gs = load(latest(rd, f"acc_gsm8k_{tag}.json") or Path("/dev/null"))
        lines.append(f"| {tag} | {fmt(mm and mm.get('mmlu_acc'))} | "
                     f"{fmt(mmft and mmft.get('mmlu_acc'))} | "
                     f"{fmt(gs and gs.get('gsm8k_acc'))} |")
    lines.append("")

    # ---- Table 2: Performance ----
    lines += ["## Table 2: Performance (median, NPU)", "",
              "| 配置 | TTFT@128 (ms) | decode@128 (tok/s) | TTFT@2048 (ms) | "
              "decode@2048 (tok/s) | Power (W) | Energy/token (J) |",
              "|---|---|---|---|---|---|---|"]
    ptags = set()
    for p in rd.glob("perf_*.json"):
        tag = p.stem[5:]
        while tag and tag[-8:].isdigit() is False and "_" in tag and tag.split("_")[-1].isdigit():
            tag = tag.rsplit("_", 1)[0]
        if "smoke" in tag:
            continue
        # 形如 qwen1.5b_w4a16_full / qwen1.5b_w4a16_power / Qwen2.5-..._20261006_194142
        tag = tag.replace("_full", "").replace("_power", "")
        if tag.count("_") and tag.rsplit("_", 1)[-1].isdigit():
            tag = tag.rsplit("_", 1)[0]
        ptags.add(tag)
    for tag in sorted(ptags):
        cands = [p for p in rd.glob(f"perf_{tag}*.json")
                 if "smoke" not in p.stem]
        if not cands:
            continue
        best = None
        for p in sorted(cands, key=lambda p: p.stat().st_mtime):
            d = load(p)
            if d and d.get("points"):
                best = d  # 取最新有效
        if best is None:
            continue
        p128, p2048 = perf_row(best, 128), perf_row(best, 2048)
        lines.append(
            f"| {tag} | {fmt(p128.get('ttft_ms_median'), 1)} | "
            f"{fmt(p128.get('decode_tps_median'))} | "
            f"{fmt(p2048.get('ttft_ms_median'), 1)} | "
            f"{fmt(p2048.get('decode_tps_median'))} | "
            f"{fmt(p128.get('power_w_mean'))} | {fmt(p128.get('tokens_per_j'))} |")
    lines.append("")

    # ---- Table 3: NAC simulated PPL ----
    lines += ["## Table 3: NAC sim-quant PPL (wikitext-2 test, CPU fp32)", "",
              "| 配置 | fp16 | minmax | NAC | Δminmax | ΔNAC | NAC改善% |",
              "|---|---|---|---|---|---|---|"]
    for p in sorted(rd.glob("val_nac_*.json")):
        d = load(p)
        if not d:
            continue
        ppl = d.get("ppl", {})
        dl = d.get("deltas", {})
        lines.append(f"| {p.stem[8:]} | {fmt(ppl.get('fp16'))} | {fmt(ppl.get('minmax'))} | "
                     f"{fmt(ppl.get('nac'))} | {fmt(dl.get('minmax_vs_fp'))} | "
                     f"{fmt(dl.get('nac_vs_fp'))} | {fmt(dl.get('nac_improvement_pct'))} |")
    lines.append("")

    # ---- Table 4: Routing threshold ----
    lines += ["## Table 4: Prefill routing threshold (CPU vs NPU)", "",
              "| 模型 | L* | 结论 |", "|---|---|---|"]
    for p in sorted(rd.glob("route_threshold_*.json")):
        d = load(p)
        if not d:
            continue
        ls = d.get("L_star")
        concl = ("CPU 交叉点存在" if ls else
                 "RK3576 上 L* 不存在（NPU prefill 全程占优，诚实负结果）")
        lines.append(f"| {Path(d.get('model', p.stem)).name} | {ls if ls else 'None'} | {concl} |")
    lines.append("")

    # ---- LaTeX ----
    lines += ["## LaTeX（Table 1 对应）", "", "```latex", r"\begin{table}[t]",
              r"\centering\caption{Accuracy of W4A16 quantized models on RK3576 NPU.}",
              r"\begin{tabular}{lccc}", r"\toprule",
              r"Config & MMLU & MMLU (first-tok) & GSM8K \\", r"\midrule"]
    for tag in sorted(tags):
        mm = load(latest(rd, f"acc_mmlu_{tag}.json") or Path("/dev/null"))
        mmft = load(latest(rd, f"acc_mmlu_firsttok_{tag}.json") or Path("/dev/null"))
        gs = load(latest(rd, f"acc_gsm8k_{tag}.json") or Path("/dev/null"))
        lines.append(f"{tag.replace('_', '-')} & {fmt(mm and mm.get('mmlu_acc'))} & "
                     f"{fmt(mmft and mmft.get('mmlu_acc'))} & {fmt(gs and gs.get('gsm8k_acc'))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "```", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[collect] -> {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
