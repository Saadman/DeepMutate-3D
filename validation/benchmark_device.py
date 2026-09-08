#!/usr/bin/env python
"""Time an identical DeepMutate-3D workload on whatever device is available.

Run this on your laptop, then on a Hugging Face ZeroGPU Space, and compare the
JSON files. The workload is fixed so the two runs are directly comparable.

    python validation/benchmark_device.py --out validation/results/device_mac.json

Fair-comparison notes
---------------------
* Model load is excluded from the timings (it is dominated by disk/network).
* Every configuration is run twice and the *second* time is reported, so the
  first-call compilation and allocation costs do not distort the result.
* On ZeroGPU each decorated call pays a GPU allocation and queueing overhead of
  roughly a second. Small jobs can therefore look SLOWER on ZeroGPU than on a
  laptop. The honest comparison is the large masked-marginal jobs, where actual
  compute dominates that fixed overhead - which is why this script sweeps a
  range of lengths instead of timing one protein.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import app  # noqa: E402

# UniProt accessions spanning the length range, so the fixed per-call overhead
# can be separated from compute that scales with the protein.
TARGETS = [
    ("P01308", "Insulin"),
    ("P69905", "Haemoglobin alpha"),
    ("P04637", "p53"),
    ("P00533", "EGFR (truncated)"),
    ("P0DTC2", "SARS-CoV-2 spike"),
]


def device_info() -> dict:
    device = app.resolve_device()
    info = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "on_hf_space": bool(
            __import__("os").environ.get("SPACE_ID")
            or __import__("os").environ.get("SPACES_ZERO_GPU")
        ),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
    return info


def timed(sequence: str, mode: str, model_label: str) -> float:
    """Second of two runs, so warm-up cost is excluded."""
    app.compute_sensitivity(sequence[:32], mode, model_label)  # warm the path
    start = time.perf_counter()
    app.compute_sensitivity(sequence, mode, model_label)
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--model", default=app.DEFAULT_MODEL_LABEL,
                        choices=list(app.MODEL_CHOICES))
    parser.add_argument("--max-length", type=int, default=1022,
                        help="truncate targets to this length")
    parser.add_argument(
        "--masked-max-length",
        type=int,
        default=250,
        help=(
            "only run masked marginals on proteins up to this length. Masked "
            "costs one forward pass PER RESIDUE, so at 650M a 1000-residue "
            "protein takes ~10 minutes on a laptop. The long targets still run "
            "in wildtype mode, which is what makes the length sweep useful."
        ),
    )
    parser.add_argument("--modes", default="both",
                        choices=["both", "wt", "masked"])
    args = parser.parse_args()

    info = device_info()
    print(json.dumps(info, indent=2))
    print(f"\nModel: {args.model}\n")

    modes = {
        "wt": ["Wildtype marginals (fast)"],
        "masked": ["Masked marginals (accurate)"],
        "both": ["Wildtype marginals (fast)", "Masked marginals (accurate)"],
    }[args.modes]

    rows = []
    print(f"{'protein':22s} {'len':>5s}  {'mode':9s} {'seconds':>9s} {'ms/residue':>11s}")
    print("-" * 62)
    for accession, name in TARGETS:
        sequence, note = app.fetch_uniprot_sequence(accession)
        if not sequence:
            print(f"{name:22s}  skipped ({note})")
            continue
        sequence = sequence[: args.max_length]
        for mode in modes:
            if (
                not mode.startswith("Wildtype")
                and len(sequence) > args.masked_max_length
            ):
                print(f"{name:22s} {len(sequence):5d}  {'masked':9s} "
                      f"{'skipped':>9s}  (over --masked-max-length)")
                continue
            seconds = timed(sequence, mode, args.model)
            per_res = 1000 * seconds / len(sequence)
            rows.append({
                "accession": accession, "name": name, "length": len(sequence),
                "mode": mode, "seconds": round(seconds, 3),
                "ms_per_residue": round(per_res, 3),
            })
            short = "wt" if mode.startswith("Wildtype") else "masked"
            print(f"{name:22s} {len(sequence):5d}  {short:9s} {seconds:9.2f} {per_res:11.2f}")

    payload = {"device_info": info, "model": app.resolve_model(args.model)[0],
               "results": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
