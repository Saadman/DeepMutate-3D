#!/usr/bin/env python
"""Can DeepMutate-3D's scores separate pathogenic from benign clinical variants?

Uses ProteinGym's curated ClinVar substitution set (Pathogenic vs Benign labels)
and asks a single question: if you rank a protein's variants by our LLR, do the
pathogenic ones fall at the damaging end?

Metric is AUROC. 0.5 is a coin flip; 1.0 is perfect separation. This is
zero-shot - no clinical label is ever shown to the model.

    python validation/run_clinvar.py --model "ESM-2 650M (most accurate)"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import app  # noqa: E402

DATA = pathlib.Path(__file__).parent / "data" / "clinical_substitutions.parquet"
RESULTS = pathlib.Path(__file__).parent / "results"
S3_URL = "https://proteingym.s3.amazonaws.com/clinical_substitutions.parquet"


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U identity. No sklearn dependency.

    `labels` is 1 for the positive class (pathogenic), 0 otherwise. Ties get
    average ranks, which is the standard treatment.
    """
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=DATA)
    parser.add_argument("--model", default=app.DEFAULT_MODEL_LABEL,
                        choices=list(app.MODEL_CHOICES))
    parser.add_argument("--mode", default="Wildtype marginals (fast)")
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--min-variants", type=int, default=10,
                        help="minimum variants for a per-protein AUROC")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=pathlib.Path, default=RESULTS)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Missing {args.data}. Download:\n  curl -L -o {args.data} {S3_URL}")
        return 1

    frame = pd.read_parquet(
        args.data, columns=["protein_id", "target_seq", "mutant", "annotation"]
    )
    frame = frame[frame["target_seq"].str.len() <= args.max_length]
    proteins = list(frame.groupby("protein_id"))
    if args.limit:
        proteins = proteins[: args.limit]

    print(f"Device {app.resolve_device()} | model {args.model} | mode {args.mode}")
    print(f"{len(proteins)} proteins, {len(frame):,} variants\n")

    aa_index = {aa: i for i, aa in enumerate(app.AMINO_ACIDS)}
    per_protein, pooled = [], []
    started = time.time()

    for i, (protein_id, group) in enumerate(proteins, 1):
        seq = group["target_seq"].iloc[0]
        try:
            _scores, llr = app.compute_sensitivity(seq, args.mode, args.model)
        except Exception as exc:  # noqa: BLE001
            print(f"  {protein_id}: scoring failed ({exc})")
            continue

        preds, labels = [], []
        for mutant, annotation in zip(group["mutant"], group["annotation"]):
            wt_aa, mut_aa = mutant[0], mutant[-1]
            try:
                pos = int(mutant[1:-1])
            except ValueError:
                continue
            if not (1 <= pos <= len(seq)) or seq[pos - 1] != wt_aa:
                continue
            if mut_aa not in aa_index:
                continue
            # Lower LLR = more damaging, so negate to make "high = pathogenic".
            preds.append(-llr[pos - 1][aa_index[mut_aa]])
            labels.append(1 if annotation == "Pathogenic" else 0)

        if not preds:
            continue
        preds_a, labels_a = np.array(preds), np.array(labels)
        pooled.extend(
            {"protein_id": protein_id, "score": p, "label": l}
            for p, l in zip(preds, labels)
        )
        if len(preds) >= args.min_variants and 0 < labels_a.sum() < len(labels_a):
            per_protein.append({
                "protein_id": protein_id, "seq_len": len(seq),
                "n_variants": len(preds), "n_pathogenic": int(labels_a.sum()),
                "auroc": auroc(preds_a, labels_a),
            })
        if i % 200 == 0:
            print(f"  [{i}/{len(proteins)}] {time.time()-started:.0f}s elapsed")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = app.resolve_model(args.model)[0].split("/")[-1]

    pooled_frame = pd.DataFrame(pooled)
    # Per-protein z-scoring before pooling: raw LLRs are not comparable across
    # proteins, so pooling them unnormalised would measure between-protein
    # offsets rather than within-protein discrimination.
    pooled_frame["z"] = pooled_frame.groupby("protein_id")["score"].transform(
        lambda s: (s - s.mean()) / (s.std() if s.std() else 1)
    )
    pooled_auroc = auroc(pooled_frame["z"].to_numpy(), pooled_frame["label"].to_numpy())
    raw_auroc = auroc(pooled_frame["score"].to_numpy(), pooled_frame["label"].to_numpy())

    pp = pd.DataFrame(per_protein)
    pp.to_csv(args.out / f"clinvar_per_protein_{tag}.csv", index=False)
    summary = {
        "model": app.resolve_model(args.model)[0],
        "mode": args.mode,
        "n_proteins_scored": int(pooled_frame["protein_id"].nunique()),
        "n_variants": int(len(pooled_frame)),
        "n_pathogenic": int(pooled_frame["label"].sum()),
        "pooled_auroc_zscored": round(pooled_auroc, 4),
        "pooled_auroc_raw": round(raw_auroc, 4),
        "n_proteins_with_auroc": int(len(pp)),
        "mean_per_protein_auroc": round(float(pp["auroc"].mean()), 4) if len(pp) else None,
        "median_per_protein_auroc": round(float(pp["auroc"].median()), 4) if len(pp) else None,
        "frac_proteins_auroc_above_0.7": round(float((pp["auroc"] > 0.7).mean()), 4) if len(pp) else None,
        "wall_clock_seconds": round(time.time() - started, 1),
    }
    (args.out / f"clinvar_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    for k, v in summary.items():
        print(f"  {k:32s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
