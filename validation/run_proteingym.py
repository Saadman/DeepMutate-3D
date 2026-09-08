#!/usr/bin/env python
"""Benchmark DeepMutate-3D's LLR scoring against ProteinGym deep mutational scans.

For each DMS assay we score the wildtype sequence once with ESM-2, read off the
log-likelihood ratio for every measured single substitution, and correlate those
predictions with the experimental fitness scores (Spearman rho, the metric
ProteinGym reports).

This is a zero-shot evaluation: no DMS data is ever shown to the model.

Usage
-----
    pip install -r validation/requirements.txt
    python validation/run_proteingym.py --limit 20            # quick pass
    python validation/run_proteingym.py --mode wt             # fast scoring
    python validation/run_proteingym.py                       # everything

Results stream to validation/results/per_assay.jsonl and the run is resumable:
re-running skips assays already present.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import app  # noqa: E402  - needs the sys.path entry above

DATA = pathlib.Path(__file__).parent / "data" / "DMS_substitutions.parquet"
RESULTS = pathlib.Path(__file__).parent / "results"
S3_URL = "https://proteingym.s3.amazonaws.com/DMS_substitutions.parquet"

MODES = {
    "masked": "Masked marginals (accurate)",
    "wt": "Wildtype marginals (fast)",
}


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman rho via ranks + Pearson. Avoids a scipy dependency.

    `.rank()` assigns average ranks to ties, which is what Spearman requires.
    """
    xr, yr = x.rank().to_numpy(), y.rank().to_numpy()
    if len(xr) < 3 or xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def load_assays(
    path: pathlib.Path, max_length: int, min_variants: int
) -> Dict[str, pd.DataFrame]:
    """Group the benchmark into single-substitution assays we can score."""
    frame = pd.read_parquet(path, columns=["DMS_id", "target_seq", "mutant", "DMS_score"])
    before = len(frame)

    # Multi-mutants (colon-separated) are out of scope: this tool scores single
    # substitutions, and an additive LLR over several sites is a different claim.
    frame = frame[~frame["mutant"].str.contains(":", regex=False)]
    print(f"  {before:,} rows -> {len(frame):,} single substitutions")

    assays: Dict[str, pd.DataFrame] = {}
    skipped_long = skipped_small = 0
    for dms_id, group in frame.groupby("DMS_id", sort=True):
        seq = group["target_seq"].iloc[0]
        if len(seq) > max_length:
            skipped_long += 1
            continue
        if len(group) < min_variants:
            skipped_small += 1
            continue
        assays[dms_id] = group
    print(
        f"  {len(assays)} assays usable "
        f"({skipped_long} over {max_length} aa, {skipped_small} under "
        f"{min_variants} variants)"
    )
    return assays


def score_assay(
    dms_id: str, group: pd.DataFrame, mode: str, model_label: str
) -> Optional[dict]:
    """Score one assay. Returns a record, or None if it could not be evaluated."""
    seq = group["target_seq"].iloc[0]
    aa_index = {aa: i for i, aa in enumerate(app.AMINO_ACIDS)}

    started = time.time()
    _scores, llr = app.compute_sensitivity(seq, mode, model_label)
    elapsed = time.time() - started

    preds: List[float] = []
    measured: List[float] = []
    mismatched = unparsed = 0

    for mutant, dms_score in zip(group["mutant"], group["DMS_score"]):
        wt_aa, mut_aa = mutant[0], mutant[-1]
        try:
            pos = int(mutant[1:-1])
        except ValueError:
            unparsed += 1
            continue
        # ProteinGym positions are 1-based against target_seq. Verify rather than
        # trust: a silent off-by-one would quietly destroy the correlation.
        if not (1 <= pos <= len(seq)) or seq[pos - 1] != wt_aa:
            mismatched += 1
            continue
        if mut_aa not in aa_index:
            unparsed += 1
            continue
        preds.append(llr[pos - 1][aa_index[mut_aa]])
        measured.append(dms_score)

    if len(preds) < 3:
        return None

    rho = spearman(pd.Series(preds), pd.Series(measured))
    return {
        "DMS_id": dms_id,
        "seq_len": len(seq),
        "n_variants": len(preds),
        "n_mismatched": mismatched,
        "n_unparsed": unparsed,
        "spearman": rho,
        "seconds": round(elapsed, 2),
        "mode": mode,
        "model": app.resolve_model(model_label)[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, default=DATA)
    parser.add_argument("--mode", choices=sorted(MODES), default="masked")
    parser.add_argument(
        "--model",
        choices=list(app.MODEL_CHOICES),
        default=app.DEFAULT_MODEL_LABEL,
        help="which ESM-2 size to benchmark",
    )
    parser.add_argument("--limit", type=int, default=None, help="max assays to run")
    parser.add_argument("--max-length", type=int, default=app.WINDOW_RESIDUES)
    parser.add_argument("--min-variants", type=int, default=100)
    parser.add_argument("--out", type=pathlib.Path, default=RESULTS)
    parser.add_argument("--fresh", action="store_true", help="ignore prior results")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Benchmark not found at {args.data}\nDownload it with:\n"
              f"  curl -L -o {args.data} {S3_URL}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.mode}_{app.resolve_model(args.model)[0].split('/')[-1]}"
    jsonl = args.out / f"per_assay_{tag}.jsonl"
    if args.fresh and jsonl.exists():
        jsonl.unlink()

    done = set()
    if jsonl.exists():
        with jsonl.open() as handle:
            done = {json.loads(line)["DMS_id"] for line in handle if line.strip()}
        print(f"Resuming: {len(done)} assays already scored")

    mode = MODES[args.mode]
    print(f"Device: {app.resolve_device()} | mode: {mode} | model: {args.model}")
    print(f"Loading {args.data} ...")
    assays = load_assays(args.data, args.max_length, args.min_variants)

    pending = [k for k in assays if k not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"Scoring {len(pending)} assays\n")

    started = time.time()
    for i, dms_id in enumerate(pending, 1):
        record = score_assay(dms_id, assays[dms_id], mode, args.model)
        if record is None:
            print(f"[{i}/{len(pending)}] {dms_id}: skipped (no usable variants)")
            continue
        with jsonl.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        flag = " MISMATCH" if record["n_mismatched"] else ""
        print(
            f"[{i}/{len(pending)}] {dms_id[:44]:44s} "
            f"L={record['seq_len']:4d} n={record['n_variants']:6d} "
            f"rho={record['spearman']:+.3f} ({record['seconds']:.1f}s){flag}"
        )

    if not jsonl.exists():
        print("No results written.")
        return 1

    frame = pd.read_json(jsonl, lines=True)
    frame = frame.sort_values("spearman", ascending=False)
    frame.to_csv(args.out / f"per_assay_{tag}.csv", index=False)

    rho = frame["spearman"].dropna()
    summary = {
        "mode": mode,
        "model": app.resolve_model(args.model)[0],
        "n_assays": int(len(rho)),
        "n_variants_total": int(frame["n_variants"].sum()),
        "mean_spearman": round(float(rho.mean()), 4),
        "median_spearman": round(float(rho.median()), 4),
        "std_spearman": round(float(rho.std()), 4),
        "frac_above_0.3": round(float((rho > 0.3).mean()), 4),
        "frac_above_0.5": round(float((rho > 0.5).mean()), 4),
        "wall_clock_seconds": round(time.time() - started, 1),
    }
    (args.out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 66)
    print(f"ProteinGym zero-shot results — {app.resolve_model(args.model)[0]}")
    print("=" * 66)
    for key, value in summary.items():
        print(f"  {key:22s} {value}")
    print("\nBest 5 assays:")
    print(frame.head(5)[["DMS_id", "seq_len", "n_variants", "spearman"]].to_string(index=False))
    print("\nWorst 5 assays:")
    print(frame.tail(5)[["DMS_id", "seq_len", "n_variants", "spearman"]].to_string(index=False))
    print(f"\nWritten to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
