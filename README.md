---
title: DeepMutate-3D
emoji: 🧬
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.26.0
python_version: 3.12.12
app_file: app.py
pinned: false
license: apache-2.0
short_description: ESM-2 mutation scanning painted onto the AlphaFold 3D fold
models:
  - facebook/esm2_t30_150M_UR50D
preload_from_hub:
  - facebook/esm2_t30_150M_UR50D
  - facebook/esm2_t33_650M_UR50D
  - facebook/esm2_t12_35M_UR50D
---

# 🧬 DeepMutate-3D

**Which residues in your protein can't tolerate mutation — and where are they in the fold?**

DeepMutate-3D scores every possible single amino-acid substitution with the ESM-2
protein language model and paints the result onto the protein's AlphaFold
structure as an interactive 3D heatmap.

**Red = evolutionarily constrained.** Positions the model refuses to change are
usually active sites, buried cores, or binding interfaces. **Blue = tolerant.**

### ▶️ [Try it in your browser — no install](https://huggingface.co/spaces/ras1992/DeepMutate-3D)

---

## What you get

Enter a UniProt ID (or paste a sequence) and press scan:

- **A 3D structure** coloured by mutation sensitivity, which you can rotate and zoom.
  Hover any point on the ribbon to read that residue's scores in the panel above it.
- **A per-residue table** giving each position's sensitivity score, a verdict
  (Critical / Sensitive / Tolerant), and its most damaging and most tolerated
  substitution.

- **Two downloads**: the sensitivity table as CSV, and the scored structure as a
  PDB file whose B-factor column holds the sensitivity score instead of pLDDT —
  so it opens straight in PyMOL or ChimeraX (`spectrum b, red_white_blue`).

Leave the sequence box empty and DeepMutate-3D fetches the canonical sequence from
UniProt for you — which also guarantees the scores line up with the structure.

## Run it locally

Requires Python 3.10+. Works on Apple Silicon (`mps`), NVIDIA (`cuda`) or CPU —
the device is chosen automatically.

```bash
git clone https://github.com/Saadman/DeepMutate-3D.git
cd DeepMutate-3D
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py                      # http://127.0.0.1:7860
```

First launch downloads ~600 MB of ESM-2 weights, then caches them.

**Try `P69905`** (haemoglobin subunit alpha) with the sequence box empty. The top
hotspot is **H88** — the proximal histidine that coordinates the haem iron. The
model is given no structure and no annotations; it recovers the functional core
from sequence alone.

## How it works

| Stage | |
|---|---|
| **Score** | For every position, ESM-2 computes `log P(mutant) − log P(wildtype)` for all 19 substitutions. Averaging them gives one *Residue Sensitivity Score* per position. |
| **Fetch** | The predicted fold is retrieved from the AlphaFold DB by UniProt accession. |
| **Paint** | Scores are written into the PDB B-factor column and rendered by py3Dmol with a red-white-blue gradient centred on zero. |

Three ESM-2 sizes are selectable (35M / 150M / 650M) and two scoring modes.
**Masked marginals** masks each position in turn (one forward pass per residue;
the Meier et al. protocol). **Wildtype marginals** uses a single pass per window.
See the benchmark below before assuming you need the slow one.

Proteins longer than ESM-2's 1024-token context are covered by overlapping
1022-residue windows, so the full 1273-residue SARS-CoV-2 spike scans fine.

## Benchmark

Evaluated zero-shot against [ProteinGym](https://proteingym.org) — **197 deep
mutational scanning assays, 644,231 experimentally measured variants**. No
experimental data is shown to the model. Metric is per-assay Spearman ρ between
predicted and measured mutation effects.

| Model | Scoring mode | Mean ρ | Median ρ | ρ > 0.3 | Compute |
|---|---|---|---|---|---|
| ESM-2 35M | wildtype marginals | 0.302 | 0.329 | 52% | 14 s |
| ESM-2 150M | wildtype marginals | 0.392 | 0.441 | 71% | **11 s** |
| ESM-2 150M | masked marginals | 0.399 | 0.453 | 72% | 4,903 s |
| **ESM-2 650M** | **wildtype marginals** | **0.431** | **0.479** | **78%** | **37 s** |

Two findings worth acting on:

**Masked marginals is not worth it at this scale.** It wins on 72% of assays but
by **+0.007 mean ρ for 462× the compute**.

**Spend compute on a bigger model, not a more expensive scoring protocol.**
ESM-2 650M in *fast* mode (ρ = 0.431, 37 s) beats ESM-2 150M in *slow* mode
(ρ = 0.399, 4,903 s) — better accuracy for **133× less compute**. 650M improves
on 150M across 70% of assays.

Reproduce it yourself:

```bash
pip install -r validation/requirements.txt
curl -L -o validation/data/DMS_substitutions.parquet \
  https://proteingym.s3.amazonaws.com/DMS_substitutions.parquet
python validation/run_proteingym.py --mode wt
```

Per-assay results are in [`validation/results/`](validation/results/).

**Where it fails, and where that turned out to be fixable.** At 150M the weakest
assays (ρ ≈ 0) were influenza haemagglutinin, engineered fluorescent proteins and
bacterial pilin. It is tempting to explain that away as fast-evolving proteins
being fundamentally unsuited to a model of evolutionary constraint — but the
ablation shows that is only partly true. Influenza HA goes from ρ = 0.003 at
150M to **ρ = 0.464 at 650M**; the limitation there was model capacity, not the
premise. Designed fluorescent proteins with no evolutionary history stay near
zero at every model size, and that failure does appear to be fundamental.

Single-chain AlphaFold models mean interface residues that only matter in a
complex are under-weighted, and for proteins longer than 1022 residues a score
depends slightly on window placement.

## Citing

If you use DeepMutate-3D, please cite it (see [`CITATION.cff`](CITATION.cff)) and
the underlying resources — **AlphaFold's CC-BY-4.0 terms require attribution**:

- Lin, Z. et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science 379, 1123–1130 (2023).
- Meier, J. et al. *Language models enable zero-shot prediction of the effects of mutations on protein function.* NeurIPS (2021).
- Jumper, J. et al. *Highly accurate protein structure prediction with AlphaFold.* Nature 596, 583–589 (2021).
- Varadi, M. et al. *AlphaFold Protein Structure Database in 2024.* Nucleic Acids Research 52, D368–D375 (2024).

## License

[Apache-2.0](LICENSE) — free for academic and commercial use.

DeepMutate-3D retrieves but does not redistribute ESM-2 (MIT), AlphaFold DB
structures (CC-BY-4.0) and UniProt sequences (CC-BY-4.0). Full attributions in
[NOTICE](NOTICE).

## ⚠️ Not a clinical tool

DeepMutate-3D predicts evolutionary constraint, which correlates with — but is not
identical to — pathogenicity. It is a research tool for hypothesis generation and
must not be used to inform medical decisions.
