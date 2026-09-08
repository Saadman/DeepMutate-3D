#!/usr/bin/env python
"""Compose Figure 1 of the preprint from three interface screenshots.

Panels are captured separately at readable zoom rather than as one shrunken
full-page shot, then assembled here. Each is placed at its native aspect ratio;
nothing is stretched.

    python paper/make_figure1.py
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
PANELS = HERE / "panels"
OUT = HERE / "figures"

WIDTH = 2400          # 8 inches at 300 dpi
GAP = 34
MARGIN = 18
LABEL_PT = 52
BORDER = (208, 208, 205)
INK = (11, 11, 11)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Matplotlib ships DejaVu, so a bold face is available without extra deps."""
    import matplotlib
    path = pathlib.Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(str(path), size)


def fit(img: Image.Image, width: int = None, height: int = None) -> Image.Image:
    if width:
        return img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    return img.resize((round(img.width * height / img.height), height), Image.LANCZOS)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a = Image.open(PANELS / "panelA_interface.png").convert("RGB")
    b = Image.open(PANELS / "panelB_structure.png").convert("RGB")
    c = Image.open(PANELS / "panelC_table.png").convert("RGB")

    font = load_font(LABEL_PT)
    label_h = LABEL_PT + 14

    # Row 1: the full interface, across the figure.
    a = fit(a, width=WIDTH)

    # Row 2: structure and table share a height so their tops and bottoms line up.
    avail = WIDTH - GAP
    row2_h = round(avail / (b.width / b.height + c.width / c.height))
    b = fit(b, height=row2_h)
    c = fit(c, height=row2_h)
    # Absorb any rounding drift into the wider panel so the row is exactly WIDTH.
    if b.width + GAP + c.width != WIDTH:
        b = b.resize((WIDTH - GAP - c.width, b.height), Image.LANCZOS)

    total_h = label_h + a.height + GAP + label_h + row2_h
    sheet = Image.new("RGB", (WIDTH, total_h), "white")
    draw = ImageDraw.Draw(sheet)

    def place(img, x, y, letter):
        draw.text((x, y - label_h + 4), letter, font=font, fill=INK)
        sheet.paste(img, (x, y))
        draw.rectangle([x, y, x + img.width - 1, y + img.height - 1], outline=BORDER, width=2)

    y = label_h
    place(a, 0, y, "A")
    y += a.height + GAP + label_h
    place(b, 0, y, "B")
    place(c, b.width + GAP, y, "C")

    for ext in ("png", "pdf"):
        sheet.save(OUT / f"figure1_interface.{ext}", dpi=(300, 300))
    print(f"figure1_interface: {sheet.size} "
          f"({sheet.width/300:.1f} x {sheet.height/300:.1f} inches at 300 dpi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
