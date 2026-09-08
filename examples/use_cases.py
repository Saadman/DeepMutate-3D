#!/usr/bin/env python
"""Worked case studies showing what DeepMutate-3D is useful for.

Each case tests the tool against ground truth that was decided independently of
this software, so the results are checkable rather than anecdotal.

  1. Cancer hotspot recovery   does the tool rank known TP53 mutation hotspots
                               among the most constrained positions?
  2. Active site recovery      across a panel of enzymes, do UniProt-annotated
                               catalytic and binding residues score as critical?
  3. Engineering candidates    which positions tolerate substitution, for
                               library design or stabilisation work?

    python examples/use_cases.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import math

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import app  # noqa: E402

MODEL = "ESM-2 650M (most accurate)"
MODE = "Masked marginals (accurate)"
OUT = pathlib.Path(__file__).parent / "results"

# TP53 mutation hotspots from IARC/COSMIC tumour sequencing. These six account
# for a large share of all TP53 missense mutations found in human cancers.
TP53_HOTSPOTS = {175: "R", 245: "G", 248: "R", 249: "R", 273: "R", 282: "R"}

# Enzymes with curated UniProt active site annotations.
ENZYME_PANEL = [
    ("P62593", "TEM-1 beta-lactamase"),
    ("P00760", "Cationic trypsin"),
    ("P00698", "Lysozyme C"),
    ("P00918", "Carbonic anhydrase 2"),
    ("P04406", "GAPDH"),
    ("P00558", "Phosphoglycerate kinase 1"),
]

FUNCTIONAL_FEATURES = {"Active site", "Binding site", "Site"}


def permutation_p(scores, positions, n_perm=20000, seed=0):
    """How often would a random set of the same size look this constrained?

    Compares the observed median sensitivity of `positions` against the median
    of random position sets drawn from the same protein. This is the null model
    the percentile numbers on their own do not provide.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores, dtype=float)
    observed = float(np.median(arr[[p - 1 for p in positions]]))
    k = len(positions)
    draws = np.array([np.median(rng.choice(arr, size=k, replace=False))
                      for _ in range(n_perm)])
    # One-sided: constrained means MORE negative than chance.
    hits = int((draws <= observed).sum())
    return observed, (hits + 1) / (n_perm + 1)


def hypergeometric_p(n_total, n_marked, n_drawn, n_hits):
    """P(at least n_hits marked items in a draw of n_drawn), exactly."""
    total = 0.0
    for i in range(n_hits, min(n_marked, n_drawn) + 1):
        total += (math.comb(n_marked, i) * math.comb(n_total - n_marked, n_drawn - i)
                  / math.comb(n_total, n_drawn))
    return total


def percentile_rank(scores, positions):
    """What fraction of residues are LESS constrained than these positions?

    100 means the most constrained residue in the protein. Uses the sensitivity
    score directly, where more negative means less tolerant of mutation.
    """
    order = pd.Series(scores).rank()          # 1 = most negative = most fragile
    n = len(scores)
    return {p: round(100.0 * (n - order[p - 1]) / (n - 1), 1) for p in positions}


def uniprot_functional_sites(accession):
    """Curated catalytic and binding residues from UniProt, as ground truth."""
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{accession}.json", timeout=30)
    r.raise_for_status()
    data = r.json()
    sites = {}
    for feat in data.get("features", []):
        if feat.get("type") not in FUNCTIONAL_FEATURES:
            continue
        loc = feat.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        if start and end and start == end:
            sites[start] = feat.get("description") or feat.get("type")
    return sites, data.get("sequence", {}).get("value", "")


def case1_tp53():
    print("=" * 72)
    print("CASE 1  TP53 cancer hotspots")
    print("=" * 72)
    seq, _ = app.fetch_uniprot_sequence("P04637")
    scores, llr = app.compute_sensitivity(seq, MODE, MODEL)
    details = app.residue_details(seq, scores, llr)
    pct = percentile_rank(scores, TP53_HOTSPOTS)

    rows = []
    for pos, wt in TP53_HOTSPOTS.items():
        assert seq[pos - 1] == wt, f"position {pos} is {seq[pos-1]}, expected {wt}"
        d = details[pos]
        rows.append({"hotspot": f"{wt}{pos}", "sensitivity": round(scores[pos - 1], 2),
                     "verdict": d["v"], "constraint_percentile": pct[pos],
                     "worst_substitution": f"{wt}{pos}{d['worst'][0]}",
                     "worst_llr": d["worst"][1]})
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))

    obs, pval = permutation_p(scores, list(TP53_HOTSPOTS))
    med = float(np.median(list(pct.values())))
    top10 = sum(1 for v in pct.values() if v >= 90)
    print(f"\n  permutation test vs random position sets of the same size:")
    print(f"    observed median sensitivity {obs:.2f} "
          f"vs protein median {np.median(scores):.2f}, p = {pval:.5f}")
    print(f"\n  median constraint percentile of the six hotspots: {med:.1f}")
    print(f"  hotspots in the most constrained 10% of the protein: {top10}/6")
    print(f"  protein median sensitivity: {np.median(scores):.2f}")
    return frame, {"median_percentile": med, "in_top_decile": top10,
                   "protein_median_sensitivity": round(float(np.median(scores)), 2),
                   "permutation_p": pval,
                   "observed_median_sensitivity": round(obs, 2)}


def case2_active_sites():
    print("\n" + "=" * 72)
    print("CASE 2  Active site recovery across an enzyme panel")
    print("=" * 72)
    rows = []
    for acc, name in ENZYME_PANEL:
        try:
            sites, seq = uniprot_functional_sites(acc)
        except Exception as exc:
            print(f"  {name}: UniProt lookup failed ({exc})")
            continue
        if not sites or not seq or len(seq) > app.WINDOW_RESIDUES:
            print(f"  {name}: skipped (sites={len(sites)}, length={len(seq)})")
            continue
        scores, _ = app.compute_sensitivity(seq, MODE, MODEL)
        pct = percentile_rank(scores, sites)
        vals = list(pct.values())
        _obs, pval = permutation_p(scores, list(sites))
        rows.append({"protein": name, "accession": acc, "length": len(seq),
                     "n_annotated_sites": len(sites),
                     "median_percentile": round(float(np.median(vals)), 1),
                     "frac_in_top_decile": round(sum(v >= 90 for v in vals) / len(vals), 2),
                     "permutation_p": round(pval, 6)})
        print(f"  {name:28s} n={len(sites):2d}  median percentile "
              f"{np.median(vals):5.1f}  in top 10%: "
              f"{sum(v>=90 for v in vals)}/{len(vals)}  p={pval:.5f}")
    frame = pd.DataFrame(rows)
    if len(frame):
        print(f"\n  panel median: {frame['median_percentile'].median():.1f} percentile")
    return frame


def case3_engineering():
    print("\n" + "=" * 72)
    print("CASE 3  Engineering candidates in lysozyme C")
    print("=" * 72)
    seq, _ = app.fetch_uniprot_sequence("P00698")
    scores, llr = app.compute_sensitivity(seq, MODE, MODEL)
    details = app.residue_details(seq, scores, llr)
    frame = pd.DataFrame({
        "pos": range(1, len(seq) + 1), "wt": list(seq),
        "sensitivity": [round(s, 2) for s in scores],
        "best_substitution": [f"{details[i+1]['wt']}{i+1}{details[i+1]['best'][0]}"
                              for i in range(len(seq))],
        "best_llr": [details[i + 1]["best"][1] for i in range(len(seq))],
    })
    tolerant = frame.nlargest(10, "sensitivity")
    print("  Ten most substitution-tolerant positions (library design targets):")
    print(tolerant.to_string(index=False))
    constrained = frame.nsmallest(10, "sensitivity")
    print("\n  Ten most constrained positions (do not touch):")
    print(constrained.to_string(index=False))

    # UniProt annotates four disulfide bridges in this protein. How surprising
    # is it that the eight participating cysteines all land in the top ten?
    ss_positions = {24, 145, 48, 133, 82, 98, 94, 112}
    hits = len(ss_positions & set(constrained["pos"]))
    pval = hypergeometric_p(len(seq), len(ss_positions), 10, hits)
    print(f"\n  disulfide cysteines in the top 10: {hits}/{len(ss_positions)}")
    print(f"  exact hypergeometric p = {pval:.3e} "
          f"(drawing 10 of {len(seq)} positions at random)")
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"model {MODEL} | mode {MODE} | device {app.resolve_device()}\n")
    tp53, tp53_summary = case1_tp53()
    tp53.to_csv(OUT / "case1_tp53_hotspots.csv", index=False)
    panel = case2_active_sites()
    panel.to_csv(OUT / "case2_active_sites.csv", index=False)
    lyso = case3_engineering()
    lyso.to_csv(OUT / "case3_lysozyme_positions.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "model": app.resolve_model(MODEL)[0], "mode": MODE,
        "case1_tp53": tp53_summary,
        "case2_panel_median_percentile":
            float(panel["median_percentile"].median()) if len(panel) else None,
    }, indent=2))
    print(f"\nwritten to {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
