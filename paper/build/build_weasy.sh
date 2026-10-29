#!/bin/bash
# WeasyPrint 路径：Markdown -> HTML(pandoc, 自包含) -> PDF(weasyprint)
set -eu
cd "$(dirname "$0")/.."          # -> paper/
OUT="$PWD/build"

python3 - "$PWD/draft_v0.1.md" "$OUT/_wp.md" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8").read().splitlines()
out, skip = [], False
for i, ln in enumerate(lines):
    if i == 0 and ln.startswith("# "):
        continue
    if ln.startswith("> **状态说明"):
        skip = True
        continue
    if skip and ln.strip() == "":
        skip = False
        continue
    if not skip:
        out.append(ln)
open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY

cat > "$OUT/_meta.yaml" <<'YAML'
title: "Calibration Objectives and Evaluation Protocols in 4-bit Edge LLM Deployment"
subtitle: "A controlled measurement study on Rockchip RK3576 — draft v0.1"
author: "NAC project · internal draft for review"
date: "2026-10-30"
YAML

pandoc "$OUT/_wp.md" \
  --metadata-file="$OUT/_meta.yaml" \
  --from markdown+pipe_tables-implicit_figures --to html5 --standalone --embed-resources \
  --css "build/style.css" \
  --output "$OUT/draft_wp.html"

rm -f "$OUT/draft_v0.1_weasy.pdf"
python3 -m weasyprint "$OUT/draft_wp.html" "$OUT/draft_v0.1_weasy.pdf" --base-url "$PWD" 2>&1 | tail -3
ls -la "$OUT/draft_v0.1_weasy.pdf"
