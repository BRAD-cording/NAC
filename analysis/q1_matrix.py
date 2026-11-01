#!/usr/bin/env python3
import io, json, math, statistics, itertools
from pathlib import Path
RES = Path("/media/pi/DATA/rknn-llm/nac_paper/results")

def load(p):
    return json.load(io.open(p, encoding="utf-8"))

CFGS = ["fp16", "minmax", "gptq", "nac", "nacv2"]

def mcnemar(a, b):
    n = min(len(a), len(b))
    if [x.get("gold") for x in a][:n] != [x.get("gold") for x in b][:n]:
        return None
    oa = [x.get("ok") for x in a][:n]
    ob = [x.get("ok") for x in b][:n]
    b01 = sum(1 for i in range(n) if oa[i] and not ob[i])
    b10 = sum(1 for i in range(n) if not oa[i] and ob[i])
    if b01 + b10 == 0:
        p = 1.0
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        p = math.erfc(math.sqrt(chi2 / 2))
    return dict(n=n, acc_a=sum(oa)/n, acc_b=sum(ob)/n, b01=b01, b10=b10, p=p)

recs = {}
for c in CFGS:
    r = load(str(RES) + "/acc_gsm8k_sim_g64_stop_%s.records.%s.json" % (c, c))
    recs[c] = r
    n = len(r)
    acc = sum(1 for x in r if x.get("ok")) / n
    ai = sum(1 for x in r if x.get("ok_answeris")) / n
    tr = sum(1 for x in r if x.get("stop_truncated")) / n
    med = statistics.median([x.get("gen_raw_chars") or len(x.get("gen") or "") for x in r])
    print("%s  n=%d  lastnum=%.4f  answeris=%.4f  trunc=%.2f  medchars=%.0f" % (c, n, acc, ai, tr, med))

print("\n=== paired McNemar (last-num), all pairs ===")
print("%-8s%-8s%4s%7s%7s%6s%4s%4s%8s" % ("A","B","n","accA","accB","dpp","b01","b10","p"))
for a, b in itertools.combinations(CFGS, 2):
    m = mcnemar(recs[a], recs[b])
    if m is None:
        print("%-8s%-8s SKIP (gold mismatch)" % (a, b)); continue
    dpp = (m["acc_b"] - m["acc_a"]) * 100
    print("%-8s%-8s%4d%7.4f%7.4f%6.1f%4d%4d%8.3f" % (
        a, b, m["n"], m["acc_a"], m["acc_b"], dpp, m["b01"], m["b10"], m["p"]))

print("\n=== paired McNemar (answer-is), all pairs ===")
print("%-8s%-8s%4s%7s%7s%6s%8s" % ("A","B","n","aiA","aiB","dpp","p"))
for a, b in itertools.combinations(CFGS, 2):
    n = min(len(recs[a]), len(recs[b]))
    aa = [x.get("ok_answeris") for x in recs[a]][:n]
    bb = [x.get("ok_answeris") for x in recs[b]][:n]
    b01 = sum(1 for i in range(n) if aa[i] and not bb[i])
    b10 = sum(1 for i in range(n) if not aa[i] and bb[i])
    if b01 + b10:
        chi2 = (abs(b01-b10)-1)**2/(b01+b10); p = math.erfc(math.sqrt(chi2/2))
    else:
        p = 1.0
    dpp = (sum(bb)-sum(aa))/n*100
    print("%-8s%-8s%4d%7.4f%7.4f%6.1f%8.3f" % (a, b, n, sum(aa)/n, sum(bb)/n, dpp, p))
