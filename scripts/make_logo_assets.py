#!/usr/bin/env python
"""Derive the published logo assets from the two source artworks.

Sources (not published, kept for regeneration):
    images/DeepMutate-3D/1.png        dark artwork, light art on near-black
    images/DeepMutate-3D_light/1.png  light artwork, dark art on light grey

Both are 2000x2000 with the artwork occupying about 3.5% of the canvas, so the
first job is always to crop to content.

The light artwork is keyed to transparency, which works cleanly in this
direction: the art is dark on a near-uniform light backdrop, so alpha follows
the distance from that backdrop and the original colours are recovered by
undoing the composite. Keying the DARK artwork the same way fails, because its
wordmark is a solid mid-blue that becomes translucent and drops to 1.72:1
contrast on a light page.

    python scripts/make_logo_assets.py
"""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"

# Display widths in CSS pixels; assets are written at 2x for Retina screens.
HEADER_CSS_WIDTH = 340
BANNER_CSS_WIDTH = 620


def content_box(rgb: np.ndarray, background: np.ndarray, pad: int = 0) -> tuple:
    """Bounding box of everything meaningfully different from the backdrop.

    `pad` defaults to 0 deliberately. Padding baked into a transparent asset is
    invisible but still occupies layout width, which indents the artwork
    relative to the text beneath it.
    """
    diff = np.abs(rgb.astype(int) - background.astype(int)).sum(axis=2)
    ys, xs = np.where(diff > 30)
    h, w, _ = rgb.shape
    return (max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0),
            min(int(xs.max()) + pad, w), min(int(ys.max()) + pad, h))


def key_out_background(img: Image.Image) -> Image.Image:
    """Make a near-uniform light backdrop transparent, preserving edge quality.

    The visible pixel is `art * alpha + background * (1 - alpha)`. Estimating
    alpha from the distance to the backdrop and inverting that relation
    recovers the true art colour, so antialiased edges stay clean instead of
    picking up a halo of the old background.
    """
    a = np.array(img.convert("RGB")).astype(float)
    background = a[3, 3].copy()

    distance = np.abs(a - background).max(axis=2)
    alpha = np.clip(distance / 60.0, 0.0, 1.0)          # fully opaque by 60/255

    safe = np.maximum(alpha[..., None], 1e-6)
    art = (a - background * (1.0 - alpha[..., None])) / safe
    art = np.clip(art, 0, 255)

    return Image.fromarray(
        np.dstack([art, alpha * 255]).astype(np.uint8), "RGBA")


def resize(img: Image.Image, width: int) -> Image.Image:
    return img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    light_src = Image.open(ROOT / "images/DeepMutate-3D_light/1.png").convert("RGB")
    arr = np.array(light_src)
    light = key_out_background(light_src.crop(content_box(arr, arr[3, 3])))

    dark_src = Image.open(ROOT / "images/DeepMutate-3D/1.png").convert("RGB")
    # The dark artwork sits on near-black, so brightness alone finds the content.
    lum = np.array(dark_src).sum(axis=2)
    ys, xs = np.where(lum > 120)
    # The dark variant keeps a small margin: its backdrop is opaque, so the
    # padding reads as part of the panel rather than as dead space.
    dark = dark_src.crop((max(int(xs.min()) - 30, 0), max(int(ys.min()) - 30, 0),
                          min(int(xs.max()) + 30, dark_src.width),
                          min(int(ys.max()) + 30, dark_src.height)))

    written = []
    for name, img, width in [
        ("logo-header.png", light, HEADER_CSS_WIDTH * 2),   # in-app, transparent
        ("logo-banner.png", light, BANNER_CSS_WIDTH * 2),   # README, transparent
        ("logo-dark.png", dark, BANNER_CSS_WIDTH * 2),      # for dark surfaces
    ]:
        out = resize(img, width)
        out.save(OUT / name, optimize=True)
        written.append(f"  {name:18s} {out.size}  {(OUT / name).stat().st_size // 1024} KB")

    icon = light.crop((0, 0, min(light.height, light.width), light.height))
    icon = resize(icon, 512)
    icon.save(OUT / "icon.png", optimize=True)
    written.append(f"  {'icon.png':18s} {icon.size}  {(OUT / 'icon.png').stat().st_size // 1024} KB")

    print("\n".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
