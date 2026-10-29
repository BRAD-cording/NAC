#!/usr/bin/env python3
"""生成工件清单 reproduce/MANIFEST.md：每个 .rkllm 的 md5 / 大小 / dtype / 角色 / 对应评测。

md5 优先取 results/env_snapshot.json 的缓存值，缺失的现算。
dtype 从 results/*.log 的 runtime 横幅提取（与模型文件对应）。
"""
import hashlib
import io
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT = ROOT / "reproduce" / "MANIFEST.md"

ROLE = {
    "Qwen2.5-1.5B-Instruct_RK3576_w4a16.rkllm": "vendor reference (1.5B)",
    "Qwen2.5-3B-Instruct_RK3576_w4a16.rkllm": "vendor reference (3B control)",
    "nac_qwen1.5b_g64_nac.rkllm": "NAC-v1 arm — injection never applied (md5 == 3bd8c0ba)",
    "nac_qwen1.5b_g64_mmctrl.rkllm": "E3 control arm — same bytes as NAC arm (3bd8c0ba)",
    "nacv2_qwen1.5b_g64_intzp.rkllm": "NAC-v2 arm — same bytes as NAC arm (3bd8c0ba)",
    "sym_minmax_g64_v2.rkllm": "minmax g64 arm — distinct bytes (d6ff15)",
    "local_w4a16_plain_x86.rkllm": "local rebuild, normal/opt1, W4A16, x86 native",
    "local_w4a16_r4a.rkllm": "local rebuild, normal/opt1, multi-domain calibration",
    "local_w4a16_r4b.rkllm": "local rebuild, normal/opt3",
    "local_w4a16_r4c.rkllm": "local rebuild, normal/opt1, num_npu_core=1",
    "local_w4a16_r4d.rkllm": "local rebuild, gdq, W4A16",
    "asym_minmax_w4a16.rkllm": "asymmetric minmax arm, W4A16",
    "sym_minmax_g64.rkllm": "minmax g64 arm — same bytes as 3bd8c0ba",
}


def md5(p, cache):
    name = p.name
    if name in cache:
        return cache[name]
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    cache[name] = h.hexdigest()
    return cache[name]


def dtypes():
    m = {}
    for f in (ROOT / "results").glob("*.log"):
        txt = f.read_text(encoding="utf-8", errors="ignore")[:40000]
        mm = re.search(r"loading rkllm model from (\S+)", txt)
        dd = re.search(r"model_dtype: (\S+)", txt)
        if mm and dd:
            m.setdefault(os.path.basename(mm.group(1)), set()).add(dd.group(1).rstrip(","))
    return {k: "/".join(sorted(v)) for k, v in m.items()}


def results_map():
    m = {}
    for f in (ROOT / "results").glob("acc_*.json"):
        if ".records" in f.name:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict):
            continue
        name = os.path.basename(d.get("model", ""))
        if not name:
            continue
        bits = []
        if "mmlu_acc" in d:
            bits.append(f"MMLU {d['mmlu_acc']}")
        if "gsm8k_acc" in d:
            bits.append(f"GSM8K {d['gsm8k_acc']}")
        if bits:
            m.setdefault(name, []).append(f.name.replace("acc_", "").replace(".json", "") + ": " + "/".join(bits))
    return m


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    snap = ROOT / "results" / "env_snapshot.json"
    if snap.exists():
        cache.update(json.loads(snap.read_text(encoding="utf-8")).get("artifacts_md5", {}))
    dt = dtypes()
    res = results_map()
    rows = []
    for p in sorted(MODELS.glob("*.rkllm")):
        h = md5(p, cache)
        rows.append((p.name, p.stat().st_size, h, dt.get(p.name, "?"),
                     ROLE.get(p.name, ""), "; ".join(res.get(p.name, []))[:90]))
    lines = [
        "# Artifact manifest",
        "",
        "Every `.rkllm` used in this study, with md5 and the runtime dtype reported by librkllm.",
        "Artifacts with identical md5 are the *same file* despite different names — see the NAC/E3 rows.",
        "",
        "| artifact | size (B) | md5 | dtype | role | evals |",
        "|---|---|---|---|---|---|",
    ]
    for n, s, h, d, r, e in rows:
        lines.append(f"| `{n}` | {s:,} | `{h[:16]}…` | {d} | {r} | {e} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
