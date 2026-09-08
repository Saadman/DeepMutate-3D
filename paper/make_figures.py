#!/usr/bin/env python
"""Generate Figures 2 and 3 for the DeepMutate-3D preprint.

Figure 1 is a screenshot of the running application and is assembled separately.

Colour choices follow a single validated system. Model size is an ORDINAL
variable, so the three models use one blue hue stepped light to dark rather than
arbitrary categorical colours; the ordering is then legible even in greyscale
and to colourblind readers. Annotated residues use the reserved red, and are
also directly labelled, so identity never rests on colour alone.
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = pathlib.Path(__file__).parent / "figures"

# Ordinal ramp, light to dark. Lightest step clears the 2:1 contrast floor.
ORDINAL = {"ESM-2 35M": "#86b6ef", "ESM-2 150M": "#2a78d6", "ESM-2 650M": "#104281"}
SERIES = "#2a78d6"
ACCENT = "#e34948"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2df"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def load(tag):
    p = ROOT / f"validation/results/per_assay_wt_{tag}.jsonl"
    return pd.read_json(p, lines=True)[["DMS_id", "spearman", "seq_len"]]


def figure2():
    """Distribution of per-assay accuracy across model sizes, and the paired gain."""
    models = [("ESM-2 35M", "esm2_t12_35M_UR50D"),
              ("ESM-2 150M", "esm2_t30_150M_UR50D"),
              ("ESM-2 650M", "esm2_t33_650M_UR50D")]
    frames = {label: load(tag) for label, tag in models}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # Panel A: empirical CDF. Shows the whole distribution, not just a summary,
    # and a uniform rightward shift is immediately visible.
    for label, frame in frames.items():
        vals = np.sort(frame["spearman"].dropna().to_numpy())
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax1.step(vals, y, where="post", color=ORDINAL[label], linewidth=2,
                 label=f"{label}  (mean {vals.mean():.3f})")
    ax1.set_xlabel("Per-assay Spearman rho")
    ax1.set_ylabel("Cumulative fraction of assays")
    ax1.set_title("A   Accuracy distribution, 217 DMS assays")
    ax1.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax1.set_axisbelow(True)
    ax1.axvline(0, color=INK_2, linewidth=0.8, linestyle=":")
    ax1.legend(frameon=False, loc="lower right", fontsize=7)

    # Panel B: paired differences. Pairing removes between-assay variance, so
    # this is the honest view of whether the larger model actually helps.
    merged = frames["ESM-2 150M"].merge(
        frames["ESM-2 650M"], on="DMS_id", suffixes=("_150", "_650"))
    diff = (merged["spearman_650"] - merged["spearman_150"]).dropna()
    ax2.hist(diff, bins=36, color=SERIES, edgecolor="white", linewidth=0.5)
    ax2.axvline(0, color=INK_2, linewidth=1.0)
    ax2.axvline(diff.mean(), color=ACCENT, linewidth=2)
    ax2.annotate(f"mean +{diff.mean():.3f}\n{100*(diff>0).mean():.0f}% of assays improve",
                 xy=(diff.mean(), ax2.get_ylim()[1] * 0.72),
                 xytext=(0.52, 0.80), textcoords="axes fraction",
                 color=INK, fontsize=7.5,
                 arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=1.2))
    ax2.set_xlabel("Change in rho, 650M minus 150M")
    ax2.set_ylabel("Assays")
    ax2.set_title("B   Paired effect of model size")
    ax2.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.9)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure2_model_size.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 2: {len(diff)} paired assays, mean gain {diff.mean():+.4f}")


def figure3():
    """Sensitivity along the lysozyme sequence, with disulfide cysteines marked."""
    import app
    seq, _ = app.fetch_uniprot_sequence("P00698")
    scores, _ = app.compute_sensitivity(
        seq, "Masked marginals (accurate)", "ESM-2 650M (most accurate)")
    scores = np.asarray(scores)
    pos = np.arange(1, len(seq) + 1)
    bridges = [(24, 145), (48, 133), (82, 98), (94, 112)]
    cys = sorted({p for pair in bridges for p in pair})

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.fill_between(pos, scores, 0, color=SERIES, alpha=0.16, linewidth=0)
    ax.plot(pos, scores, color=SERIES, linewidth=1.4)

    median = float(np.median(scores))
    ax.axhline(median, color=INK_2, linewidth=0.9, linestyle="--")
    # Park the median label in the left margin, clear of the line itself.
    ax.annotate(f"protein median {median:.2f}", xy=(2, median),
                xytext=(2, 8), textcoords="offset points",
                fontsize=7, color=INK_2, ha="left", va="bottom")

    ax.scatter(cys, scores[[c - 1 for c in cys]], s=34, color=ACCENT,
               zorder=5, edgecolor="white", linewidth=0.8,
               label="cysteine in a disulfide bridge")
    # Stagger labels so neighbouring pairs (94/98, 112/133) do not collide.
    for idx, c in enumerate(cys):
        ax.annotate(f"C{c}", xy=(c, scores[c - 1]),
                    xytext=(0, -13 if idx % 2 == 0 else -21),
                    textcoords="offset points", ha="center", fontsize=6.5,
                    color=ACCENT)

    ax.set_xlabel("Residue position")
    ax.set_ylabel("Sensitivity (mean LLR)")
    ax.set_title("Lysozyme C: all eight disulfide cysteines rank among the "
                 "most constrained positions")
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    # Upper centre: the trace sits low on the right, so the legend cannot
    # collide with the C145 label there.
    ax.legend(frameon=False, loc="upper center", fontsize=7,
              bbox_to_anchor=(0.5, 1.02))
    ax.set_xlim(-4, len(seq) + 6)          # headroom so C145 is not clipped
    ax.set_ylim(min(scores) - 2.6, max(scores) + 0.6)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure3_lysozyme.{ext}", bbox_inches="tight")
    plt.close(fig)
    ranks = pd.Series(scores).rank()
    print(f"  Figure 3: {len(seq)} residues; disulfide cysteine ranks "
          f"{sorted(int(ranks[c-1]) for c in cys)} of {len(seq)}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure2()
    figure3()
    print(f"\nwritten to {OUT}/")
