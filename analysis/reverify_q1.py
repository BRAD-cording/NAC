#!/usr/bin/env python3
"""Q1 终复查：5 配置截断协议全配对 McNemar + 秩排序稳健性。"""
import io, json, math, statistics, glob, itertools
from pathlib import Path
ROOT = Path("/media/pi/DATA/rknn-llm/nac_paper"); RES = ROOT/"results"
WIN={".":0,"gptq":0,"minmax":1,"nac":2,"nacv2":3,"fp16":4}  # 输出顺序

def load(p):
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except Exception as e:
        print("LOAD_ERR",p,e); return None

def mcnemar(a,b):
    n=min(len(a),len(b))
    ga=[x.get("gold") for x in a][:n]; gb=[x.get("gold") for x in b][:n]
    align = ga==gb
    oa=[x.get("ok") for x in a][:n]; ob=[x.get("ok") for x in b][:n]
    b01=sum(1 for i in range(n) if oa[i] and not ob[i])
    b10=sum(1 for i in range(n) if not oa[i] and ob[i])
    if b01+b10==0: p=1.0
    else:
        if b01<b10: b01,b10=b10,b01
        # exact binomial P(X<=b10 | n=b01+b10, 0.5) two-sided approx
        # use chi-square with Yates (same as sim_compare)
        chi2=(abs(b01-b10)-1)**2/(b01+b10)
        p=math.erfc(math.sqrt(chi2/2))
    return dict(n=n,align=align,acc_a=sum(oa)/n,acc_b=sum(ob)/n,b01=b01,b10=b10,p=p)

def main():
    cfgs=["fp16","minmax","gptq","nac","nacv2"]
    recs={}
    for c in cfgs:
        rp=RES/f"acc_gsm8k_sim_g64_stop_{c}.records.{c}.json"
        r=load(rp)
        recs[c]=r
        n=len(r); acc=sum(1 for x in r if x.get("ok"))/n
        ai=sum(1 for x in r if x.get("ok_answeris"))/n
        tr=sum(1 for x in r if x.get("stop_truncated"))/n
        med=statistics.median([x.get("gen_raw_chars") or len(x.get("gen") or "") for x in r])
        print(f"{c:8s} n={n:3d} lastnum={acc:.4f} answeris={ai:.4f} trunc={tr:.2f} medchars={med:.0f}")
    print("\n=== paired McNemar (last-num), all 15 pairs ===")
    print("A       B         n  accA1   accB     dpp  b01 b10       p align")
    for a,b in itertools.combinations(cfgs,2):
        m=mcnemar(recs[a],recs[b])
        if m is None: print(f"{a:8s}{b:8s} SKIP (no recs)"); continue
        dpp=(m["acc_b"]-m["acc_a"])*100
        print(f"{a:8s}{b:8s}{m[n]:4d}{m[acc_a]:7.4f}{m[acc_b]:7.4f}{dpp:6.1f}{m[b01]:4d}{m[b10]:4d}{m[p]:8.3f}{str(m[align]):>6s}")
    print("\n=== answer-is, all 15 pairs ===")
    for a,b in itertools.combinations(cfgs,2):
        n=min(len(recs[a]),len(recs[b]))
        aa=[x.get("ok_answeris") for x in recs[a]][:n]; bb=[x.get("ok_answeris") for x in recs[b]][:n]
        oa=[x.get("ok") for x in recs[a]][:n]; ob=[x.get("ok") for x in recs[b]][:n]
        na=sum(aa); nb=sum(bb)
        b01=sum(1 for i in range(n) if aa[i] and not bb[i]); b10=sum(1 for i in range(n) if not aa[i] and bb[i])
        if b01+b10: chi2=(abs(b01-b10)-1)**2/(b01+b10); p=math.erfc(math.sqrt(chi2/2))
        else: p=1.0
        ga=[x.get("gold") for x in recs[a]][:n]; gb=[x.get("gold") for x in recs[b]][:n]
        print(f"{a:8s}{b:8s}{n:4d}{na/n:7.4f}{nb/n:7.4f}{(nb-na)/n*100:6.1f}{p:8.3f}{str(ga==gb):>6s}")

if __name__=="__main__":
    main()
