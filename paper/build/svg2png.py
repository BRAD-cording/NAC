"""SVG -> PNG（高分辨率），仅支持本项目 figs.py 生成的元素。用于 docx/PDF 回退路径。"""
import sys
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

NS = "{http://www.w3.org/2000/svg}"
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "C:/Windows/Fonts/arial.ttf"]


def font(size):
    for p in FONTS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def num(s, default=0.0):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def main(src, dst, scale=2.5):
    root = ET.parse(src).getroot()
    w, h = float(root.get("width")), float(root.get("height"))
    W, H = int(w * scale), int(h * scale)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    def S(v):
        return v * scale

    for el in root.iter():
        tag = el.tag.replace(NS, "")
        a = el.attrib
        if tag == "line":
            d.line([S(num(a["x1"])), S(num(a["y1"])), S(num(a["x2"])), S(num(a["y2"]))],
                   fill=a.get("stroke", "black"), width=max(1, int(S(num(a.get("stroke-width"), 1)))))
        elif tag == "rect":
            if "x" not in a and "width" not in a:
                continue
            x, y = num(a.get("x", 0)), num(a.get("y", 0))
            rw = w - x if a.get("width", "").endswith("%") else num(a.get("width", 0))
            rh = h - y if a.get("height", "").endswith("%") else num(a.get("height", 0))
            fill = a.get("fill", "black")
            d.rectangle([S(x), S(y), S(x + rw), S(y + rh)],
                        fill=None if fill == "none" else fill,
                        outline=a.get("stroke") if a.get("stroke") not in (None, "none") else None)
        elif tag == "circle":
            x, y, r = num(a["cx"]), num(a["cy"]), num(a["r"])
            fill = a.get("fill", "black")
            d.ellipse([S(x - r), S(y - r), S(x + r), S(y + r)],
                      fill=None if fill == "none" else fill,
                      outline=a.get("stroke") if a.get("stroke") not in (None, "none") else None)
        elif tag == "text":
            txt = "".join(el.itertext())
            size = int(S(num(a.get("font-size"), 12)))
            anchor = {"middle": "mm", "end": "rm"}.get(a.get("text-anchor"), "lm")
            d.text((S(num(a["x"])), S(num(a["y"]))), txt, fill=a.get("fill", "black"),
                   font=font(size), anchor=anchor)
    img = img.resize((int(w * 1.6), int(h * 1.6)), Image.LANCZOS)
    img.save(dst, dpi=(300, 300))
    print("wrote", dst, img.size)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.5)
