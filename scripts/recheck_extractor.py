#!/usr/bin/env python3
"""m1 复核：抽取器伪影检查（离线，不重跑推理）。

背景：accuracy.py extract_number 取生成文本最后一个数字，无 "The answer is"
fallback。AWQ sim GSM8K acc=0.000 的判定依赖该规则；本脚本用三种规则对已落盘
records 重算，回答"0.000 是真实校准行为还是抽取器伪影"。

规则：
  A  现行规则：最后一个数字（extract_number 原样）
  Bf answer-is 优先（first match），无命中回退 A
  Bl answer-is 优先（last match），无命中回退 A

口径告警（写入输出 JSON）：records.gen 为截断存档（sim 300 字符 / 板端 200 字符），
原始评分用的是推理时全文。本复核在截断文本上进行——先报 A 与存档 ok 的一致率作
代表性检验，再比较三规则。结论只用于定性裁定（伪影 vs 真实行为），不改写论文数字。

用法：python scripts/recheck_extractor.py   （输出 results/extractor_recheck.json）
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = {
    "sim_awq": "results/acc_gsm8k_sim_g64_awq.records.awq.json",
    "sim_nac_v1": "results/acc_gsm8k_sim_g64.records.nac.json",
    "sim_minmax": "results/acc_gsm8k_sim_g64.records.minmax.json",
    "sim_gptq": "results/acc_gsm8k_sim_g64.records.gptq.json",
    "sim_nacv2": "results/acc_gsm8k_sim_g64_nacv2.records.nacv2.json",
    "board_vendor_zs": "results/acc_gsm8k_qwen1.5b_w4a16_zs.records.json",
}

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
ANS_RE = re.compile(r"answer\s+(?:is|:|=)\s*\$?\s*(-?\d[\d,]*\.?\d*)")


def rule_a(text: str):
    m = NUM_RE.findall(text.replace(",", ""))
    return float(m[-1]) if m else None


def rule_b(text: str, last: bool):
    ms = ANS_RE.findall(text.replace(",", ""))
    if ms:
        return float(ms[-1] if last else ms[0])
    return rule_a(text)


def hit_gold(pred, gold: str) -> bool:
    try:
        return pred is not None and abs(pred - float(gold.replace(",", ""))) < 1e-4
    except ValueError:
        return False


def main():
    out = {"caveat": "records.gen 是截断存档（sim 300 / board 200 字符）；原始评分为推理时全文。"
                     "本复核在截断文本上进行，A 规则与存档 ok 的一致率代表其代表性；"
                     "结论仅用于 m1 定性裁定（抽取器伪影 vs 真实校准行为），不改写论文数字。",
           "configs": {}}
    for name, rel in CONFIGS.items():
        p = ROOT / rel
        if not p.exists():
            out["configs"][name] = {"error": f"missing {rel}"}
            continue
        recs = json.loads(p.read_text())
        n = len(recs)
        agree = flips = 0
        acc = {"A": 0, "Bf": 0, "Bl": 0}
        gold_in_gen = 0
        changed = []
        for r in recs:
            gen, gold = r["gen"], r["gold"]
            pa, pbf, pbl = rule_a(gen), rule_b(gen, False), rule_b(gen, True)
            oa, obf, obl = hit_gold(pa, gold), hit_gold(pbf, gold), hit_gold(pbl, gold)
            acc["A"] += oa; acc["Bf"] += obf; acc["Bl"] += obl
            agree += int(oa == bool(r["ok"]))
            gold_in_gen += int(gold.replace(",", "") in gen.replace(",", ""))
            if (obf or obl) != oa:
                flips += 1
                changed.append({"i": r["i"], "gold": gold, "predA": str(pa),
                                "predBf": str(pbf), "predBl": str(pbl)})
        out["configs"][name] = {
            "n": n, "source": rel,
            "A_agreement_with_stored_ok": round(agree / n, 4),
            "acc_A_recomputed": round(acc["A"] / n, 4),
            "acc_B_first": round(acc["Bf"] / n, 4),
            "acc_B_last": round(acc["Bl"] / n, 4),
            "records_changed_by_B": flips,
            "gold_string_in_gen": gold_in_gen,
            "changed_samples": changed[:10],
        }
    dst = ROOT / "results/extractor_recheck.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in out["configs"].items():
        if "error" in v:
            print(f"{k:18s} {v['error']}")
            continue
        print(f"{k:18s} n={v['n']:3d}  A={v['acc_A_recomputed']:.4f}  "
              f"Bf={v['acc_B_first']:.4f}  Bl={v['acc_B_last']:.4f}  "
              f"agreeA={v['A_agreement_with_stored_ok']:.3f}  changed={v['records_changed_by_B']}")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
