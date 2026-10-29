#!/usr/bin/env python3
"""生成论文图（纯 Python SVG，无第三方依赖）。

产物：
  paper/figs/fig1_ppl_vs_gsm8k.svg   精度-困惑度散点（核心图）
  paper/figs/fig2_truncation.svg     32-token 截断签名：生成长度分布
  paper/figs/fig3_group_size.svg     g64 vs g128 MMLU

数据来源全部为 results/*.json（见 analysis/check_claims.py）。
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 640, 420
ML, MR, MT, MB = 70, 30, 30, 60  # margins


def sx(v, lo, hi):
    return ML + (v - lo) / (hi - lo) * (W - ML - MR)


def sy(v, lo, hi):
    return H - MB - (v - lo) / (hi - lo) * (H - MT - MB)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self):
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def line(self, x1, y1, x2, y2, w=1, color="#333", dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{color}" stroke-width="{w}"{d}/>')

    def text(self, x, y, s, size=12, anchor="start", color="#222", weight="normal"):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica,Arial" '
                 f'font-size="{size}" text-anchor="{anchor}" fill="{color}" '
                 f'font-weight="{weight}">{esc(s)}</text>')

    def circle(self, x, y, r=5, fill="#2b6cb0", stroke="none"):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}"/>')

    def rect(self, x, y, w, h, fill="#2b6cb0", stroke="none"):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                 f'fill="{fill}" stroke="{stroke}"/>')

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                f'viewBox="0 0 {W} {H}">\n<rect width="100%" height="100%" fill="white"/>\n')
        Path(path).write_text(head + "\n".join(self.parts) + "\n</svg>\n", encoding="utf-8")


def fig1():
    pts = [  # name, ppl, gsm8k, filled, dx, dy
        ("min-max", 12.2251, 0.075, True, 0, -9),
        ("GPTQ", 12.1679, 0.305, True, -30, -9),
        ("NAC-v1", 11.8930, 0.070, True, 0, -9),
        ("NAC-v2", 11.9620, 0.150, True, 6, -9),
        ("AWQ*", 11.2260, 0.000, False, 34, 4),
    ]
    s = SVG()
    s.text(W / 2, 18, "Perplexity does not predict long-form accuracy", 13, "middle")
    xlo, xhi, ylo, yhi = 11.1, 12.35, -0.02, 0.35
    s.line(ML, H - MB, W - MR, H - MB)
    s.line(ML, MT, ML, H - MB)
    for v in [11.2, 11.4, 11.6, 11.8, 12.0, 12.2]:
        x = sx(v, xlo, xhi)
        s.line(x, H - MB, x, H - MB + 4)
        s.text(x, H - MB + 16, f"{v:.1f}", 10, "middle")
    for v in [0.0, 0.1, 0.2, 0.3]:
        y = sy(v, ylo, yhi)
        s.line(ML - 4, y, ML, y)
        s.text(ML - 8, y + 3, f"{v:.1f}", 10, "end")
    s.text((ML + W - MR) / 2, H - 14, "WikiText-2 perplexity (64 seq x 1024)", 11, "middle")
    s.text(ML, MT - 4, "GSM8K acc.", 9, "start", "#555")
    for name, ppl, acc, filled, dx, dy in pts:
        x, y = sx(ppl, xlo, xhi), sy(acc, ylo, yhi)
        s.circle(x, y, 5, "#2b6cb0" if filled else "none", "#2b6cb0")
        s.text(x + dx, y + dy, name, 10, "middle")
    s.text(W - MR, MT + 8, "* AWQ PPL is a 16-seq probe", 8, "end", "#888")
    s.save(OUT / "fig1_ppl_vs_gsm8k.svg")


def fig2():
    """32-token 截断签名：两配置的生成长度分布（每 10 字符一桶）。"""
    def lens(fn):
        d = json.loads((ROOT / "results" / fn).read_text(encoding="utf-8"))
        return [len(r.get("gen", "")) for r in d]

    a = lens("acc_gsm8k_qwen1.5b_w4a16_zs32.records.json")
    b = lens("acc_gsm8k_qwen1.5b_g64nac_zs.records.json")
    bins = list(range(40, 200, 10))
    hist = lambda L: [sum(1 for x in L if lo <= x < lo + 10) for lo in bins]
    ha, hb = hist(a), hist(b)
    import math
    ymax = math.ceil(max(max(ha), max(hb)) / 10) * 10

    s = SVG()
    s.text(W / 2, 18, "All board GSM8K runs stop at the 32-token budget", 13, "middle")
    s.line(ML, H - MB, W - MR, H - MB)
    s.line(ML, MT, ML, H - MB)
    for k in range(0, 6):
        v = ymax * k / 5
        y = (H - MB) - v / ymax * (H - MT - MB)
        s.line(ML - 4, y, ML, y)
        s.text(ML - 8, y + 3, f"{v:.0f}", 10, "end")
    for lo in [40, 80, 120, 160, 200]:
        x = sx(lo, 40, 200)
        s.line(x, H - MB, x, H - MB + 4)
        s.text(x, H - MB + 16, str(lo), 10, "middle")
    for i, lo in enumerate(bins):
        x0 = sx(lo, 40, 200)
        wbar = (sx(50, 40, 200) - sx(40, 40, 200)) * 0.38
        ya = (H - MB) - ha[i] / ymax * (H - MT - MB)
        yb = (H - MB) - hb[i] / ymax * (H - MT - MB)
        s.rect(x0 - wbar, ya, wbar, H - MB - ya, "#2b6cb0")
        s.rect(x0, yb, wbar, H - MB - yb, "#dd6b20")
    s.text((ML + W - MR) / 2, H - 14, "generated characters (10-char bins)", 11, "middle")
    s.rect(W - 220, MT, 10, 10, "#2b6cb0")
    s.text(W - 205, MT + 9, "vendor @32 (acc 0.0000)", 9)
    s.rect(W - 220, MT + 16, 10, 10, "#dd6b20")
    s.text(W - 205, MT + 25, "NAC build @32 (acc 0.0050)", 9)
    s.save(OUT / "fig2_truncation.svg")


def fig3():
    data = [("min-max", 0.4675, 0.3622), ("NAC-v1", 0.3932, 0.4293), ("GPTQ", 0.4520, 0.4871)]
    s = SVG()
    s.text(W / 2, 18, "Group size reverses the NAC-v1 order", 13, "middle")
    ylo, yhi = 0.30, 0.55
    s.line(ML, H - MB, W - MR, H - MB)
    s.line(ML, MT, ML, H - MB)
    for v in [0.30, 0.35, 0.40, 0.45, 0.50]:
        y = sy(v, ylo, yhi)
        s.line(ML - 4, y, ML, y)
        s.text(ML - 8, y + 3, f"{v:.2f}", 10, "end")
    gw = (W - ML - MR) / len(data)
    for i, (name, g64, g128) in enumerate(data):
        cx = ML + gw * (i + 0.5)
        bw = gw * 0.22
        for j, (val, col) in enumerate([(g64, "#2b6cb0"), (g128, "#dd6b20")]):
            y = sy(val, ylo, yhi)
            s.rect(cx - bw + j * bw, y, bw, H - MB - y, col)
            s.text(cx - bw / 2 + j * bw, y - 5, f"{val:.3f}", 9, "middle")
        s.text(cx, H - MB + 18, name, 11, "middle")
    s.text((ML + W - MR) / 2, H - 12, "MMLU (969 items, 32-token)", 11, "middle")
    s.text(ML, MT - 4, "MMLU acc.", 9, "start", "#555")
    s.rect(W - 180, MT, 10, 10, "#2b6cb0")
    s.text(W - 165, MT + 9, "g64", 9)
    s.rect(W - 180, MT + 16, 10, 10, "#dd6b20")
    s.text(W - 165, MT + 25, "g128", 9)
    s.save(OUT / "fig3_group_size.svg")


def fig4():
    """部署侧：同一模型、同一 dtype，只变构建算法。"""
    data = [("vendor", 0.5583, "#718096"), ("normal\nopt1", 0.4974, "#2b6cb0"),
            ("normal\nopt3", 0.5191, "#2b6cb0"), ("normal\nmultidomain", 0.4974, "#2b6cb0"),
            ("gdq", 0.5387, "#2f855a")]
    s = SVG()
    s.text(W / 2, 18, "A local rebuild does not reproduce vendor quality by default", 12.5, "middle")
    ylo, yhi = 0.45, 0.58
    s.line(ML, H - MB, W - MR, H - MB)
    s.line(ML, MT, ML, H - MB)
    for v in [0.45, 0.48, 0.51, 0.54, 0.57]:
        y = sy(v, ylo, yhi)
        s.line(ML - 4, y, ML, y)
        s.text(ML - 8, y + 3, f"{v:.2f}", 10, "end")
    gw = (W - ML - MR) / len(data)
    for i, (name, val, col) in enumerate(data):
        cx = ML + gw * (i + 0.5)
        bw = gw * 0.46
        y = sy(val, ylo, yhi)
        s.rect(cx - bw / 2, y, bw, H - MB - y, col)
        s.text(cx, y - 5, f"{val:.4f}", 9, "middle")
        for k, part in enumerate(name.split("\n")):
            s.text(cx, H - MB + 16 + k * 12, part, 9, "middle")
    yv = sy(0.5583, ylo, yhi)
    s.line(ML, yv, W - MR, yv, 1, "#c53030", "5,3")
    s.text(W - MR, yv - 5, "vendor 0.5583", 9, "end", "#c53030")
    s.text((ML + W - MR) / 2, H - 12, "MMLU (969 items, 32-token), all W4A16", 10, "middle")
    s.text(ML, MT - 4, "MMLU acc.", 9, "start", "#555")
    s.save(OUT / "fig4_build_path.svg")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("wrote", *[p.name for p in sorted(OUT.glob("*.svg"))])
