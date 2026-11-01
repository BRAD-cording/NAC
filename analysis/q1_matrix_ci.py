#!/usr/bin/env python3
"""Q1: paired McNemar + OR + bootstrap 95% CI, truncated-protocol GSM8K.
OR = b01/b10 (A right/B wrong over B right/A wrong). CI over paired delta by
item bootstrap. Honest: every number re-derived from archived JSON."""
import io, json, math, statistics, itertools, random
from pathlib import Path
RES = Path("/media/pi/DATA/rknn-llm/nac_paper/results")

def load(p): return json.load(io.open(p, encoding="utf-8"))

CFGS = ["fp16","minmax","gptq","nac","nacv2"]

def mcnemar(a, b, key="ok", seed=42):
    n = min(len(a), len(b))
    if [x.get("gold") for x in a][:n] != [x.get("gold") for x in b][:n]:
        return None
    oa = [bool(x.get(key)) for x in a][:n]
    ob = [bool(x.get(key)) for x in b][:n]
    b01 = sum(1 for i in range(n) if oa[i] and not ob[i])
    b10 = sum(1 for i in range(n) if not oa[i] and ob[i])
    if b01 + b10 == 0:
        p = 1.0
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        p = math.erfc(math.sqrt(chi2 / 2))
    # OR = b01/b10 (accA right, B wrong) / (B right, A wrong); guard div0
    orr = (b01 / b10) if b10 else float("inf")
    # paired delta accB-accA
    dpp = (sum(ob) - sum(oa)) / n * 100
    # bootstrap CI over items for paired delta (pp)
    rng = random.Random(seed)
    deltas = [int(ob[i]) - int(oa[i]) for i in range(n)]
    bd = []
    for _ in range(2000):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        bd.append(sum(s) / n * 100)
    bd.sort()
    lo, hi = bd[int(0.025*len(bd))], bd[int(0.975*len(bd))]
    return dict(n=n, acc_a=sum(oa)/n, acc_b=sum(ob)/n, b01=b01, b10=b10,
                dpp=dpp, p=p, orr=orr, cilo=lo, cihi=hi)

recs = {}
for c in CFGS:
    r = load(str(RES) + "/acc_gsm8k_sim_g64_stop_%s.records.%s.json" % (c, c))
    recs[c] = r

def fmt_or(x):
    return ("inf" if x == float("inf") else "%.2f" % x)

key = "ok"  # last-number
print("=== paired McNemar (last-num) with OR + bootstrap 95% CI ===\n")
print("%-8s%-8s%4s%7s%7s%7s%6s%6s%4s%4s%10s" % ("A","B","n","accA","accB","dpp","lo","hi","b01","b10","p"))
for a, b in itertools.combinations(CFGS, 2):
    m = mcnemar(recs[a], recs[b], key)
    if m is None:
        print("%-8s%-8s SKIP" % (a, b)); continue
    print("%-8s%-8s%4d%7.4f%7.4f%7.1f%6.1f%6.1f%4d%4d%10.3f" % (
        a, b, m["n"], m["acc_a"], m["acc_b"], m["dpp"], m["cilo"], m["cihi"],
        m["b01"], m["b10"], m["p"]))
    print("     OR=%s" % fmt_or(m["orr"]))

key2 = "ok_answeris"
print("\n=== paired McNemar (answer-is) with OR + bootstrap 95% CI ===\n")
print("%-8s%-8s%4s%7s%7s%7s%6s%6s%10s" % ("A","B","n","aiA","aiB","dpp","lo","hi","p"))
for a, b in itertools.combinations(CFGS, 2):
    m = mcnemar(recs[a], recs[b], key2)
    if m is None:
        print("%-8s%-8s SKIP" % (a, b)); continue
    print("%-8s%-8s%4d%7.4f%7.4f%7.1f%6.1f%6.1f%10.3f" % (
        a, b, m["n"], m["acc_a"], m["acc_b"], m["dpp"], m["cilo"], m["cihi"], m["p"]))
    print("     OR=%s" % fmt_or(m["orr"]))
