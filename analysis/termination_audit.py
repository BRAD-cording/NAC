"""终止行为分析（sim GSM8K 存档前缀）：区分"算错"与"不停"。

存档只有 300 字符前缀，因此：
- len < 280 且以句号/答案句结尾 → 生成在归档前停止（大概率 EOS）
- len == 300 → 至少写到 300 字符，是否停止未知（记为 capped）

输出每个配置：早停率、早停中的正确率、capped 中的正确率，
以及"末句是 answer is 且数字==gold"的比例。
"""
import glob
import io
import json
import os
import re

ROOT = "/media/pi/DATA/rknn-llm/nac_paper"
ANS = re.compile(r"answer\s+is\s+\$?\s*(-?\d[\d,]*\.?\d*)", re.I)


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


print(f"{'config':40s} {'n':>4s} {'early':>6s} {'acc_early':>9s} {'acc_cap':>8s} {'ans_tail':>8s} {'ans_tail_ok':>11s}")
for f in sorted(glob.glob(os.path.join(ROOT, "results", "acc_gsm8k_sim_g64*.records*.json"))):
    recs = load(f)
    if not isinstance(recs, list) or not recs:
        continue
    n = len(recs)
    early = [r for r in recs if len((r.get("gen") or "")) < 280]
    capped = [r for r in recs if len((r.get("gen") or "")) >= 280]
    acc_e = sum(1 for r in early if r.get("ok")) / max(1, len(early))
    acc_c = sum(1 for r in capped if r.get("ok")) / max(1, len(capped))
    # 生成末尾 80 字符里是否有 answer is，且数值等于 gold
    tail_ans = tail_ok = 0
    for r in recs:
        g = (r.get("gen") or "")[-80:]
        m = ANS.findall(g)
        if m:
            tail_ans += 1
            try:
                if abs(float(m[0].replace(",", "")) - float(str(r["gold"]).replace(",", ""))) < 1e-4:
                    tail_ok += 1
            except ValueError:
                pass
    name = os.path.basename(f).replace("acc_gsm8k_sim_g64", "").replace(".records", "").replace(".json", "")
    print(f"{name:40s} {n:4d} {len(early)/n:6.3f} {acc_e:9.3f} {acc_c:8.3f} {tail_ans/n:8.3f} {tail_ok/n:11.3f}")
