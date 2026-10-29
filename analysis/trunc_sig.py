import json, io, glob, os, re, statistics

ROOT = r"D:\work\nacpaper\nac_paper"
TAIL = re.compile(r"[0-9+\-*/=,.:]")
for f in sorted(glob.glob(os.path.join(ROOT, "results", "acc_gsm8k_*.records*.json"))):
    d = json.load(io.open(f, encoding="utf-8"))
    recs = d if isinstance(d, list) else d.get("records", [])
    if not recs:
        continue
    n = len(recs)
    L = [len(r.get("gen", "")) for r in recs]
    tail_num = sum(1 for r in recs if TAIL.search((r.get("gen") or "").strip()[-1:]))
    ans = sum(1 for r in recs if "answer is" in (r.get("gen") or "").lower())
    name = os.path.basename(f).replace("acc_gsm8k_", "").replace(".records.json", "")
    print(f"{name:40s} n={n:3d} p50={statistics.median(L):5.0f} p90={sorted(L)[int(.9*n)-1]:5d} "
          f"tailmid={tail_num/n:4.2f} hasAns={ans/n:4.2f}")
