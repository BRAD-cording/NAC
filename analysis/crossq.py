import glob
import io
import json
import os
import re

for f in sorted(glob.glob("/media/pi/DATA/rknn-llm/nac_paper/results/acc_gsm8k_sim_g64*.records*.json")):
    d = json.load(io.open(f, encoding="utf-8"))
    n = len(d)
    cross = sum(1 for r in d if "Question:" in (r.get("gen") or "")[120:])
    ans = sum(1 for r in d if re.search(r"answer is", r.get("gen") or "", re.I))
    print(os.path.basename(f)[:44].ljust(44), "n=%3d" % n,
          "cross_q=%.2f" % (cross / n), "ans=%.2f" % (ans / n))
