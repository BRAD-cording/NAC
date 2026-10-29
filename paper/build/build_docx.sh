#!/bin/bash
# 构建 docx（含高清 PNG 图），再由 LibreOffice Kit 转 PDF
set -eu
cd "$(dirname "$0")/.."          # -> paper/
OUT="$PWD/build"
mkdir -p "$OUT/figs"

for f in figs/*.svg; do
  b=$(basename "$f" .svg)
  python3 build/svg2png.py "$f" "$OUT/figs/$b.png" 2.5 >/dev/null
done
echo "figures:"; ls "$OUT/figs"

# 图片路径改写为 PNG（pandoc 无 rsvg 时不能嵌 SVG）
sed 's#(figs/\([a-z0-9_]*\)\.svg)#(build/figs/\1.png)#g' draft_v0.1.md > "$OUT/_draft.md"

# 去掉开头的状态说明块
python3 - "$OUT/_draft.md" "$OUT/_body.md" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8").read().splitlines()
out, skip = [], False
for i, ln in enumerate(lines):
    if i == 0 and ln.startswith("# "):
        continue          # 标题由 metadata 提供，避免重复
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

pandoc "$OUT/_body.md" \
  --metadata title="Calibration Objectives and Evaluation Protocols in 4-bit Edge LLM Deployment" \
  --metadata subtitle="A controlled measurement study on Rockchip RK3576 — draft v0.1" \
  --metadata date="2026-10-30" \
  --from markdown+pipe_tables --to docx \
  --resource-path="$PWD" \
  -o "$OUT/draft_v0.1.docx"

ls -la "$OUT/draft_v0.1.docx"
