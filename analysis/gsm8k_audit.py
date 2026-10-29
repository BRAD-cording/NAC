import json, io, re, glob, os, statistics

ROOT = r"D:\work\nacpaper\nac_paper"
ANS_RE = re.compile(r"answer\s+(?:is|:|=)\s*\$?\s*(-?\d[\d,]*\.?\d*)")
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)

def lastnum(t):
    m = NUM_RE.findall(t.replace(",", ""))
    return float(m[-1]) if m else None

def ansfirst(t):
    m = ANS_RE.findall(t.replace(",", ""))
    return float(m[0]) if m else None

def gold_of(r):
    try:
        return float(str(r["gold"]).replace(",", ""))
    except Exception:
        return None

files = sorted(glob.glob(os.path.join(ROOT, "results", "acc_gsm8k_*.records*.json")))
print(f"{'file':62s} {'n':>3s} {'stored':>6s} {'lastnum':>7s} {'ans1st':>7s} {'hasAns':>6s} {'medlen':>6s} {'p90':>5s} {'<=5tok':>6s} {'tail'}")
for f in files:
    d = load(f)
    if isinstance(d, dict):
        recs = d.get("records") or []
    else:
        recs = d
    if not isinstance(recs, list) or not recs:
        continue
    n = len(recs)
    ok_stored = sum(1 for r in recs if r.get("ok"))
    ln, lo, lf, has = 0, 0, 0, 0
    lens = []
    short = 0
    tails = []
    for r in recs:
        g = r.get("gen") or ""
        gold = gold_of(r)
        lens.append(len(g))
        a = ansfirst(g)
        l = lastnum(g)
        if a is not None: has += 1
        if a is not None and gold is not None and abs(a - gold) < 1e-4: lo += 1
        if l is not None and gold is not None and abs(l - gold) < 1e-4: lf += 1
        if len(g.strip()) < 20: short += 1
        tails.append(g.strip()[-1:] if g.strip() else "")
    med = statistics.median(lens)
    p90 = sorted(lens)[int(0.9 * n) - 1]
    tailtxt = "".join(sorted(set(tails)))[:12]
    name = os.path.basename(f).replace("acc_gsm8k_", "").replace(".records.json", "")
    print(f"{name:62s} {n:3d} {ok_stored/n:6.3f} {lo/n:7.3f} {lf/n:7.3f} {has/n:6.2f} {med:6.0f} {p90:5d} {short:6d} {tailtxt!r}")
