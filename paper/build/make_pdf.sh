#!/bin/bash
# 构建论文 PDF：Markdown → HTML(pandoc) → PDF(WeasyPrint)
# 用法（在 WSL 中）： bash /mnt/d/work/nacpaper/nac_paper/paper/build/make_pdf.sh
set -eu
cd "$(dirname "$0")/.."          # -> paper/
OUT="$PWD/build"
mkdir -p "$OUT"

TMP="$OUT/_body.md"
SRC="draft_v0.1.md"

# 去掉开头的"状态说明"引用块（不属于论文正文），保留其余内容
tr -d '' < "$SRC" | awk 'BEGIN{skip=0} /^> \*\*状态说明/{skip=1} skip==1 && /^$/{skip=0; next} skip==0{print}' > "$TMP"

cat > "$OUT/_meta.yaml" <<'YAML'
title: "Calibration Objectives and Evaluation Protocols in 4-bit Edge LLM Deployment"
subtitle: "A controlled measurement study on Rockchip RK3576 (draft v0.1, internal)"
author: "NAC project — draft for internal review"
date: "2026-11-01"
lang: en
YAML

pandoc "$TMP" \
  --metadata-file="$OUT/_meta.yaml" \
  --from markdown+pipe_tables+tex_math_dollars \
  --to html5 --standalone --self-contained \
  --css "build/style.css" \
  --output "$OUT/draft.html"

python3 -m weasyprint "$OUT/draft.html" "$OUT/draft_v0.1.pdf" \
  --base-url "$PWD" 2>&1 | tail -3

ls -la "$OUT/draft_v0.1.pdf"
