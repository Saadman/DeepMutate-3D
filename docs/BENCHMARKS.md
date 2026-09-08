# Benchmarks

All results are zero-shot. No experimental measurement is ever shown to the
model. Every number below was produced by the scripts in `validation/`.

## ProteinGym: mutation effect prediction

[ProteinGym](https://proteingym.org) collects deep mutational scanning
experiments, where thousands of variants of a protein are measured in a
single functional assay. Metric is Spearman rho between predicted and
measured effect, computed per assay.

Scope: 197 assays, 644,231 single substitutions.

| Model | Scoring mode | Mean rho | Median rho | rho > 0.3 | Compute |
| --- | --- | --- | --- | --- | --- |
| esm2_t12_35M | wildtype | 0.302 | 0.329 | 52% | 14 s |
| esm2_t30_150M | wildtype | 0.392 | 0.441 | 71% | 11 s |
| esm2_t30_150M | masked | 0.399 | 0.453 | 72% | 4903 s |
| esm2_t33_650M | wildtype | 0.430 | 0.479 | 78% | 37 s |

Two results worth acting on:

1. **Masked marginals is not worth its cost at this scale.** It beats
   wildtype marginals by +0.007 mean rho
   for 434x the compute.
2. **Model size buys more than scoring protocol.** ESM-2 650M in the fast
   mode (rho 0.430, 37 s)
   beats ESM-2 150M in the slow mode (rho 0.399,
   4903 s), at
   133x less compute.

## ClinVar: pathogenic versus benign classification

A different task on independent data. Given a protein's clinically annotated
variants, can the score rank pathogenic ones as more damaging? Metric is
AUROC, where 0.5 is chance.

Scope: 2,011 proteins, 38,901 variants (22,669 pathogenic). Model esm2_t33_650M_UR50D.

| Metric | Value |
| --- | --- |
| Mean per-protein AUROC | **0.881** |
| Median per-protein AUROC | 0.931 |
| Proteins with AUROC > 0.7 | 89% |
| Pooled AUROC, per-protein z-scored | 0.771 |
| Pooled AUROC, raw scores | 0.868 |

Per-protein AUROC is the primary metric: it asks whether variants can be
ranked *within* a protein, which is the actual use. Raw LLR values are not
comparable across proteins, so pooling requires per-protein z-scoring first.

## Runtime: CPU, Apple Silicon, NVIDIA

Same model (ESM-2 650M), same proteins, measured with
`validation/benchmark_device.py`. CPU and Apple Silicon figures are from an
Apple M3 (6 threads for the CPU run) and will not transfer to other
hardware; the ratios are more portable than the absolute times. NVIDIA
figures are from Hugging Face ZeroGPU, taking the minimum of repeated calls
so that GPU allocation overhead is excluded.

| Protein | Length | Mode | M3 CPU | M3 mps | NVIDIA | NVIDIA vs CPU |
| --- | --- | --- | --- | --- | --- | --- |
| Haemoglobin alpha | 142 | wildtype | 0.41 s | 0.09 s | 0.18 s | 2.4x |
| Haemoglobin alpha | 142 | masked | 30.98 s | 7.82 s | 1.45 s | 21.4x |
| p53 | 393 | wildtype | 0.78 s | 0.18 s | 0.18 s | 3.9x |
| p53 | 393 | masked | 236.19 s | 62.27 s | 10.99 s | 21.5x |
| SARS-CoV-2 spike | 1273 | wildtype | 3.50 s | 0.43 s | 0.31 s | 13.7x |
| SARS-CoV-2 spike | 1273 | masked | not measured | not measured | 152.33 s | - |

The pattern that matters: in wildtype mode everything is fast everywhere, and
on small proteins NVIDIA is actually *slower* than a laptop because ZeroGPU
pays a fixed allocation cost of roughly a second. The GPU advantage appears
in masked mode and grows with length, because attention cost is quadratic.
Masked scoring of the 1273 residue spike protein takes 152 s on NVIDIA; the
same job was not run to completion on CPU, where the measured 601 ms per
residue at 393 residues already implies well over half an hour.

## Where the method fails

- Designed proteins with no evolutionary history, such as engineered
  fluorescent proteins, score near zero at every model size. This looks
  fundamental: the model estimates evolutionary constraint, and there is none.
- Fast-evolving viral surface proteins scored near zero at 150M, but influenza
  haemagglutinin recovers to rho 0.46 at 650M. That limitation was model
  capacity, not the premise.
- AlphaFold `-F1-` models are single chains, so residues that only matter in a
  complex are under-weighted.
- For proteins longer than 1022 residues, a score depends slightly on where
  the sliding window boundaries fall.

## Reproducing

```bash
pip install -r validation/requirements.txt
mkdir -p validation/data
curl -L -o validation/data/DMS_substitutions.parquet \
  https://proteingym.s3.amazonaws.com/DMS_substitutions.parquet
curl -L -o validation/data/clinical_substitutions.parquet \
  https://proteingym.s3.amazonaws.com/clinical_substitutions.parquet

python validation/run_proteingym.py --mode wt --model "ESM-2 650M (most accurate)"
python validation/run_clinvar.py
python validation/benchmark_device.py --out validation/results/device_local.json
```

Per-assay and per-protein outputs are in `validation/results/`.
