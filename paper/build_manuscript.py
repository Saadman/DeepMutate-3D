#!/usr/bin/env python
"""Build submittable manuscript files from the markdown source.

Produces:
    DeepMutate-3D_preprint.docx   bioRxiv accepts Word directly
    DeepMutate-3D_preprint.html   standalone, figures embedded; print to PDF

Pandoc can emit PDF directly, but only through a LaTeX engine, and none is
installed here. Rather than require a TeX distribution, this script produces a
Word file (which bioRxiv accepts as-is) and a self-contained HTML file styled
for printing, which any browser turns into a clean PDF with Cmd+P.

    python paper/build_manuscript.py
"""

from __future__ import annotations

import base64
import pathlib
import re

import pypandoc

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "DeepMutate-3D_preprint.md"
FIGURES = HERE / "figures"

# Inserted at each figure's caption so the images appear in the output.
FIGURE_FILES = {
    "Figure 1.": "figure1_interface.png",
    "Figure 2.": "figure2_model_size.png",
    "Figure 3.": "figure3_lysozyme.png",
}

PRINT_CSS = """
@page { size: A4; margin: 22mm 20mm; }
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 10.5pt; line-height: 1.5; color: #111;
  max-width: 46em; margin: 0 auto;
}
h1 { font-size: 18pt; line-height: 1.25; margin: 0 0 .3em; }
h2 { font-size: 13pt; margin-top: 1.6em; border-bottom: 1px solid #ddd;
     padding-bottom: .2em; }
h3 { font-size: 11.5pt; margin-top: 1.2em; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt;
        margin: 1em 0; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; }
th { background: #f4f4f2; }
code, pre { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt; }
pre { background: #f7f7f5; padding: 10px 12px; border-radius: 4px;
      overflow-x: auto; page-break-inside: avoid; }
img { max-width: 100%; height: auto; display: block; margin: 1em auto; }
p { text-align: justify; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
blockquote { margin: 1em 0; padding-left: 1em; border-left: 3px solid #ddd;
             color: #444; }
h2, h3 { page-break-after: avoid; }
"""


def with_figures(markdown: str) -> str:
    """Place each figure image immediately above its caption."""
    for marker, filename in FIGURE_FILES.items():
        path = FIGURES / filename
        if not path.exists():
            print(f"  WARNING: {filename} missing, caption left without an image")
            continue
        pattern = re.compile(r"^(\*\*" + re.escape(marker) + r")", re.M)
        markdown = pattern.sub(f"![]({path.as_posix()})\n\n\\1", markdown, count=1)
    return markdown


def embed_images(html: str) -> str:
    """Inline every <img> as a data URI so the HTML is a single portable file."""
    def repl(match):
        src = match.group(1)
        path = pathlib.Path(src)
        if not path.is_absolute():
            path = HERE / src
        if not path.exists():
            return match.group(0)
        data = base64.b64encode(path.read_bytes()).decode()
        return match.group(0).replace(src, f"data:image/png;base64,{data}")
    return re.sub(r'<img[^>]+src="([^"]+)"', repl, html)


def main() -> int:
    markdown = with_figures(SRC.read_text())

    docx = HERE / "DeepMutate-3D_preprint.docx"
    # bioRxiv recommends Times/Courier/Helvetica/Arial for reliable PDF
    # conversion. Word's default theme uses Aptos and Consolas, so the build
    # supplies a reference document with the fonts swapped.
    extra = ["--standalone"]
    reference = HERE / "reference.docx"
    if reference.exists():
        extra += ["--reference-doc", str(reference)]
    pypandoc.convert_text(markdown, "docx", format="markdown",
                          outputfile=str(docx), extra_args=extra)
    print(f"  {docx.name:34s} {docx.stat().st_size // 1024} KB")

    html = pypandoc.convert_text(
        markdown, "html5", format="markdown",
        extra_args=["--standalone", "--metadata", "pagetitle=DeepMutate-3D",
                    "--css=inline"])
    html = html.replace('<link rel="stylesheet" href="inline" />',
                        f"<style>{PRINT_CSS}</style>")
    html = embed_images(html)
    out_html = HERE / "DeepMutate-3D_preprint.html"
    out_html.write_text(html)
    print(f"  {out_html.name:34s} {out_html.stat().st_size // 1024} KB (standalone)")

    print("\nTo produce the PDF:")
    print(f"  open {out_html}")
    print("  then Cmd+P -> Save as PDF (enable 'Print backgrounds' for the tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
