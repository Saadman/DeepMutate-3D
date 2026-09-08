# DeepMutate-3D: interactive protein language model mutation scanning on predicted structures

**Rashid Karim**

Independent Researcher, Cambridge, MA, USA

Correspondence: rashidsaadman@gmail.com

ORCID: https://orcid.org/0000-0002-5179-5259

Code: https://github.com/Saadman/DeepMutate-3D
Web application: https://huggingface.co/spaces/ras1992/DeepMutate-3D

**Keywords:** protein language model, variant effect prediction, deep mutational
scanning, AlphaFold, structural bioinformatics, ESM-2, web application

---

## Abstract

**Motivation.** Protein language models predict the functional effect of amino
acid substitutions without training on experimental data, but their output is a
table of numbers. The structural context that makes such a table interpretable,
which residues sit in an active site, a buried core, or a flexible loop, lives
in separate software. Bridging the two currently requires local installation,
model weights, and scripting, which places the method out of reach of many
laboratory scientists.

**Results.** DeepMutate-3D scores all 19 possible substitutions at every
position of a protein using ESM-2 log-likelihood ratios, condenses them into a
per-residue sensitivity score, retrieves the matching AlphaFold model, and
paints the scores onto the structure as an interactive heat map in a web
browser. Predictions are zero-shot. On the ProteinGym substitution benchmark
(217 assays, 696,311 measured variants) the deployed configuration reaches a
mean Spearman correlation of 0.425, consistent with the published ESM-2 650M
baseline of 0.414. On 38,901 ClinVar variants across 2,011 proteins it separates
pathogenic from benign substitutions with a mean per-protein AUROC of 0.881.
In three case studies with independent ground truth it recovers TP53 cancer
hotspots (permutation p = 0.0007), enzyme active sites, and all eight
disulfide-bonded cysteines of lysozyme C (hypergeometric p = 1.0e-11).

**Availability.** Apache-2.0. Runs in a browser with no installation, on shared
NVIDIA hardware, or locally on CPU or Apple Silicon.

---

## 1. Introduction

Determining which residues in a protein tolerate mutation is a recurring
question in variant interpretation, enzyme engineering, and functional
annotation. Deep mutational scanning answers it experimentally but requires
months of work per protein. Computational predictors offer a fast alternative,
and protein language models in particular have proven capable of zero-shot
variant effect prediction: trained only to reconstruct masked residues across
evolutionary sequence data, they implicitly learn the constraints that shape
protein families [1, 2].

The practical bottleneck is not accuracy but access and interpretation. A
predictor returns a score per substitution. Interpreting that score requires
knowing where the residue sits in the fold. A position scoring as highly
constrained is only actionable once one knows it lines a catalytic pocket rather
than a disordered terminus. Structural prediction has become universally
available [3, 4], yet in practice the score and the structure are produced by
different tools and joined by hand.

DeepMutate-3D closes that gap in a single interface. It is not a new scoring
method. The underlying computation is the masked-marginal log-likelihood ratio
of Meier et al. [2] applied to ESM-2 [1]. The contribution is integration and
accessibility: a scan that requires no installation, no model weights, and no
scripting, returning a structure that can be rotated and interrogated residue by
residue, together with exportable tables and coordinate files.

Several established predictors address variant effect prediction, including
alignment-based and language-model approaches evaluated in ProteinGym [5], and
web servers exist for structure-based stability prediction. DeepMutate-3D does
not compete with these on accuracy, and no head-to-head comparison against them
is attempted here. Its distinction is the combination of zero-shot scoring,
automatic structural mapping, and browser-based access with no installation.

Because the tool wraps an established method, its value depends on the wrapper
being faithful. This note therefore reports validation on two independent
benchmarks, an ablation across model sizes, a comparison of scoring protocols,
and three case studies against ground truth that was established independently
of this work.

## 2. Implementation

### 2.1 Scoring

For a sequence of length L, position i, and candidate residue m, the model
computes a log-likelihood ratio against the wild-type residue:

```
LLR(i, m) = log P(m | context) - log P(wt_i | context)
```

Values near zero indicate that the model finds the substitution as plausible as
the wild-type residue; strongly negative values indicate a substitution the
model regards as unlikely given the evolutionary context. Because the quantity
is a ratio, it is insensitive to positions at which the model is uniformly
uncertain.

The 19 non-wild-type ratios at each position are averaged into a single Residue
Sensitivity Score:

```
S_i = (1/19) * sum over m != wt_i of LLR(i, m)
```

Two scoring protocols are provided. *Masked marginals* replaces position i with
a mask token before each forward pass, following Meier et al. [2], and costs one
forward pass per residue. *Wild-type marginals* uses a single forward pass over
the intact sequence, reading the output distribution at every position. Three
ESM-2 sizes are selectable (35M, 150M, 650M parameters).

### 2.2 Sequences longer than the model context

ESM-2 has a 1,024-token context, which after special tokens accommodates 1,022
residues. Longer proteins are covered with overlapping windows of 1,022 residues
at a stride of 766. Log-probabilities are accumulated in sequence space and
divided by a per-position visit count, so that every interior residue is scored
in two different contexts and averaged. The overlap avoids the systematic error
that arises when a residue falls at a hard window boundary with no downstream
context.

### 2.3 Structure retrieval and mapping

The predicted structure is retrieved from the AlphaFold Protein Structure
Database [4] by UniProt accession, queried through the prediction API so that
the current model version is used rather than a hard-coded filename. Sensitivity
scores are written into the B-factor column of the PDB record, replacing pLDDT,
which allows 3Dmol.js [6] to colour the cartoon representation by a numeric
property without server-side rendering. The colour scale is a red-white-blue gradient with
symmetric limits, placing the neutral midpoint at exactly zero so that colour is
comparable across proteins. Exported coordinate files carry REMARK records
documenting that the B-factor column no longer contains pLDDT.

### 2.4 Interface and deployment

The application is a single-file Gradio program. Hovering any point on the
structure reports the residue's sensitivity score, its categorical verdict, and
its most damaging and most tolerated substitutions. Results are exportable as a
CSV table and as a scored PDB file that opens directly in PyMOL or ChimeraX.

The deployed instance runs on Hugging Face Spaces using dynamically allocated
NVIDIA hardware, with GPU time reserved per request in proportion to the size of
the job. The same source runs unmodified on CUDA, Apple Silicon, or CPU.

## 3. Validation

All evaluations are zero-shot. No experimental or clinical measurement is shown
to the model at any point.

### 3.1 Mutation effect prediction

ProteinGym [5] aggregates deep mutational scanning experiments into a common
benchmark. Predictions were generated for every measured single substitution and
correlated with experimental fitness per assay using Spearman's rho.

**Table 1.** ProteinGym substitution benchmark, 217 assays, 696,311 variants,
wild-type marginal scoring.

| Model | Mean rho | Median rho | rho > 0.3 | rho > 0.5 | Compute |
| --- | --- | --- | --- | --- | --- |
| ESM-2 35M | 0.291 | 0.321 | 51% | 19% | 17 s |
| ESM-2 150M | 0.383 | 0.430 | 70% | 35% | 19 s |
| ESM-2 650M | **0.425** | 0.477 | 77% | 44% | 47 s |

The deployed configuration (650M) reaches 0.425. Published values for ESM-2
650M on this benchmark are in the region of 0.41 to 0.42. The present figure is
therefore in the expected range, but it should not be read as outperforming
those reports: scoring protocol, assay filtering, and aggregation scheme all
differ between published evaluations, and the comparison here is indicative
rather than exact. The purpose of Table 1 is to establish that the wrapper does
not degrade the underlying method, not to claim an improvement over it.

### 3.2 Scoring protocol

Masked and wild-type marginal scoring were compared pairwise on an identical
subset of 197 assays at 150M parameters.

**Table 2.** Paired comparison of scoring protocols, 197 assays, ESM-2 150M.

| Protocol | Mean rho | Median rho | Compute |
| --- | --- | --- | --- |
| Wild-type marginals | 0.392 | 0.441 | 11 s |
| Masked marginals | 0.399 | 0.453 | 4,903 s |

Masked marginals is superior on 72.1% of assays, but the paired mean difference
is +0.0069 (95% CI +0.0050 to +0.0089) for 434 times the compute. The
theoretically preferable protocol therefore buys a real but very small
improvement at large cost.

Read alongside Table 1, this yields the practically useful observation that
compute is better spent on model capacity than on scoring protocol: ESM-2 650M
under wild-type marginals (rho 0.425) exceeds ESM-2 150M under masked marginals
(rho 0.399) while using two orders of magnitude less computation. This
comparison was performed at 150M; the corresponding measurement at 650M was not
made, and the claim is not generalised beyond the model size tested.

### 3.3 Clinical variant classification

An independent task on independent data. Using the ClinVar substitution set
curated within ProteinGym, variants of each protein were ranked by predicted
damage and scored by AUROC against their pathogenic or benign annotation.

**Table 3.** ClinVar classification, 2,011 proteins, 38,901 variants
(22,669 pathogenic), ESM-2 650M.

| Metric | Value |
| --- | --- |
| Mean per-protein AUROC | **0.881** |
| Median per-protein AUROC | 0.931 |
| Proteins with AUROC > 0.7 | 89.4% |
| Pooled AUROC, per-protein z-scored | 0.771 |
| Pooled AUROC, raw scores | 0.868 |

Per-protein AUROC is the primary metric because it matches the clinical
question: given the variants observed in one gene, which are most likely to be
damaging. Raw scores are not comparable between proteins, so pooled analysis
requires per-protein standardisation; the pooled figure is lower than the mean
per-protein figure because pooling mixes proteins of differing discriminability.
Section 6 discusses the limitations of ClinVar as a benchmark.

### 3.4 Runtime

**Table 4.** Inference time, ESM-2 650M, identical proteins. CPU and Apple
Silicon measured on an Apple M3 (6 threads for CPU); NVIDIA figures from
dynamically allocated Hugging Face hardware, taking the minimum of repeated
calls to exclude allocation overhead.

| Protein | Length | Protocol | M3 CPU | M3 GPU (mps) | NVIDIA |
| --- | --- | --- | --- | --- | --- |
| Haemoglobin alpha | 142 | wild-type | 0.41 s | 0.09 s | 0.18 s |
| Haemoglobin alpha | 142 | masked | 30.98 s | 7.82 s | 1.45 s |
| p53 | 393 | masked | 236.19 s | 62.27 s | 10.99 s |
| SARS-CoV-2 spike | 1,273 | wild-type | 3.50 s | 0.43 s | 0.31 s |
| SARS-CoV-2 spike | 1,273 | masked | not run | not run | 152.33 s |

Wild-type marginal scoring is fast on all hardware; the complete 1,273-residue
spike protein is scored in under four seconds even on CPU. GPU acceleration
matters for masked marginal scoring, where it delivers a 5.4 to 5.7-fold
speed-up at matched protein length and where the advantage grows with length
because attention cost scales quadratically. On small jobs under the fast
protocol, dynamically allocated GPUs are slower than a laptop, because a fixed
allocation overhead of approximately one second dominates.

## 4. Worked examples

Benchmarks establish that the scores are calibrated. These three examples test
whether they answer questions a researcher would pose, each against ground
truth established independently of this work.

Both scoring modes are reported. This turns out to matter: Section 3.2 showed
the two modes differ by only 0.007 mean Spearman on ProteinGym, yet they differ
substantially here. Rank correlation across a whole deep mutational scan and
recovery of a handful of functional residues are not the same task, and a
benchmark that measures the first does not settle the second.

### 4.1 Prioritising positions for mutagenesis

Tumour sequencing has identified six positions accounting for a large share of
TP53 missense mutations. Scanning the sequence alone, with no cancer data,
places those six at a median constraint percentile of 92.5 under masked
marginals and 93.4 under wildtype marginals, with four of six in the most
constrained tenth of the protein under either mode.

Against 20,000 random six-position draws from the DNA-binding domain, where all
six hotspots lie, p = 0.009 (masked). Drawing from the whole protein gives
p = 0.0007, but that null is too permissive: TP53's disordered terminal regions
are trivially unconstrained, so the domain-restricted value is the one to cite.

For a gene with an established hotspot catalogue the tool recovers what is
already known. For one without, the same ranking provides a prioritised
shortlist for panel design or functional follow-up.

### 4.2 Locating functional sites in uncharacterised proteins

Six enzymes were selected before scoring on two criteria: curated UniProt active
site or binding site annotations, and a length within the model context. No
enzyme was examined and then discarded.

| Mode | Panel median percentile | Enzymes at p < 0.01 |
| --- | --- | --- |
| Masked marginals | 86.8 | 4 of 6 |
| Wildtype marginals | 80.6 | 3 of 6 |

Under masked marginals, TEM-1 beta-lactamase reaches p = 0.0001,
phosphoglycerate kinase 1 p = 0.00005, GAPDH p = 0.00015 and carbonic anhydrase
2 p = 0.005. Two enzymes fail in both modes. Lysozyme C has three annotated
sites, too few for significance however well they rank. Cationic trypsin
(p = 0.081 masked) has an annotation set mixing substrate-binding with catalytic
residues, and binding residues sit under weaker evolutionary constraint than
catalytic ones.

Carbonic anhydrase 2 illustrates the mode dependence: significant under masked
marginals (p = 0.005) but not under wildtype (p = 0.101). Users pursuing
functional site discovery should select masked marginals rather than accept the
default.

### 4.3 Selecting positions for library design

Directed evolution requires the inverse question answered: which positions
tolerate substitution. Lysozyme C, 147 residues, provides a check at the
constrained end, because UniProt annotates four disulfide bridges
(C24-C145, C48-C133, C82-C98, C94-C112) whose eight cysteines should be
essentially unsubstitutable.

| Mode | Disulfide cysteines in the ten most constrained positions | Exact p |
| --- | --- | --- |
| Masked marginals | 8 of 8 | 1.0e-11 |
| Wildtype marginals | 6 of 8 | 4.4e-07 |

Under masked marginals the remaining two positions are a buried glycine and a
conserved tryptophan in the substrate-binding cleft. The model received sequence
alone: no structure, no annotation, no experimental data.

One qualification applies. Cysteines are constrained as a class, reaching a
median constraint percentile of 96.6 in lysozyme and 90.2 in trypsin against a
non-cysteine median near 48, so part of this effect is generic rather than
specific to disulfide bonding. A partial control exists in carbonic anhydrase 2,
whose single cysteine forms no disulfide bond and reaches only the 40.5
percentile, below the protein median. This suggests the model distinguishes
bonded from free cysteines rather than merely conserving the residue type, but
with a single control it remains suggestive rather than established.

The tolerant end of the ranking carries no comparable external check. Positions
reported as substitution-tolerant are the model's own output, and this work does
not verify experimentally that they tolerate mutation. That claim awaits a
prospective test.

## 5. Operation

A scan requires a UniProt accession. Leaving the sequence field empty retrieves
the canonical sequence from UniProt [7] automatically, which guarantees that
scores and structure share residue numbering; mismatched numbering is the most common source of
silent error when the two are assembled by hand.

Three outputs are produced. The interactive structure provides spatial context,
with a hover readout giving each position's score, verdict and extreme
substitutions. The per-residue table, exportable as CSV, provides the ranking
for downstream prioritisation. The scored PDB carries sensitivity in the
B-factor column, so it opens directly in PyMOL or ChimeraX and can be coloured
with a single command, which is the path from a browser scan to a publication
figure.

No installation, model download or account is required. A 142-residue protein
completes in approximately four seconds. Detailed usage documentation is
maintained with the source code rather than reproduced here.

## 6. Limitations

**The method is not novel.** Scoring is the established masked-marginal
procedure of Meier et al. [2]. Section 3.1 reproduces rather than improves on
published performance.

**ClinVar is an imperfect benchmark.** Pathogenic annotations are enriched in
conserved functional domains and benign annotations in tolerant regions, which
inflates apparent performance relative to prospective clinical use. Some ClinVar
submissions also incorporate computational predictions, introducing partial
circularity. The AUROC in Table 3 should be read as evidence that the score
behaves sensibly on clinically annotated variation, not as an estimate of
diagnostic accuracy.

**Designed proteins are out of scope.** Engineered fluorescent proteins score
near zero at every model size. The method estimates evolutionary constraint, and
sequences without evolutionary history have none to estimate.

**Model capacity limits some families.** Fast-evolving viral surface proteins
scored near zero at 150M parameters, but influenza haemagglutinin recovers to
rho 0.46 at 650M, indicating that this was a capacity limitation rather than a
property of the approach.

**Averaging discards direction.** A position may tolerate conservative
substitutions while forbidding drastic ones. The mean flattens this, which is
why the interface also reports the extreme substitutions at each position.

**Single chains only.** AlphaFold monomer models are used, so residues that
matter only within a complex are under-weighted.

**Window placement.** For proteins beyond 1,022 residues, scores depend
marginally on where window boundaries fall.

**Not a clinical tool.** The method predicts evolutionary constraint, which
correlates with but is not identical to pathogenicity. It is intended for
hypothesis generation and must not inform diagnostic decisions.

## 7. Figures

**Figure 1. The DeepMutate-3D interface**, scanning haemoglobin subunit alpha
(UniProt P69905, 142 residues) with ESM-2 650M under masked-marginal scoring.
(A) The complete interface: sequence and accession inputs, model and scoring
controls, CSV and PDB downloads, and a summary reporting the device, model,
inference time and the most constrained residues. The scan shown ran on
dynamically allocated NVIDIA hardware in 3.99 s, 28.1 ms per residue.
(B) The per-residue table, giving the sensitivity score, a categorical verdict,
and the most damaging and most tolerated substitution at each position.
(C) The AlphaFold model painted by residue sensitivity, with the hover readout
for H88. UniProt annotates this position as the proximal haem b binding
residue, and the scan ranks it third most constrained of 142 positions, in the
top 2%, having seen only the sequence. Red marks constrained positions and blue
substitution-tolerant ones, with the neutral midpoint fixed at zero so colour is
comparable between proteins.
Panels B and C are laid out as the application presents them.

**Figure 2. Effect of model size on ProteinGym.** (A) Empirical cumulative
distribution of per-assay Spearman correlation for three ESM-2 sizes across 217
assays. The distribution shifts uniformly rightward with model capacity rather
than improving only on easy assays. (B) Paired per-assay difference between
650M and 150M. Pairing removes between-assay variance; the mean gain is +0.041
and 71% of assays improve.

**Figure 3. Sensitivity along the lysozyme C sequence.** Per-residue sensitivity
for all 147 positions under ESM-2 650M with masked marginals, with the eight
cysteines forming the four UniProt-annotated disulfide bridges marked. They
occupy ranks 1, 2, 3, 5, 6, 7, 8 and 10 of 147. Under wildtype marginals six of
the eight fall in the ten most constrained positions. The dashed line is the
protein median.

## 8. Availability

Source code, validation scripts, and all result files are at
https://github.com/Saadman/DeepMutate-3D under Apache-2.0. The web application
is at https://huggingface.co/spaces/ras1992/DeepMutate-3D. Every table in this
note is reproducible from the scripts in `validation/` and `examples/`.

DeepMutate-3D retrieves but does not redistribute ESM-2 (MIT), AlphaFold
structures (CC-BY-4.0), and UniProt sequences (CC-BY-4.0).

## Acknowledgements

This work uses ESM-2 from Meta AI, the AlphaFold Protein Structure Database from
DeepMind and EMBL-EBI, UniProt, ProteinGym, and 3Dmol.js. Computation used
Hugging Face Spaces.

## References

1\. Lin Z, Akin H, Rao R, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science* 379(6637):1123-1130 (2023).

2\. Meier J, Rao R, Verkuil R, et al. Language models enable zero-shot prediction of the effects of mutations on protein function. *Advances in Neural Information Processing Systems* 34:29287-29303 (2021).

3\. Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 596(7873):583-589 (2021).

4\. Varadi M, Bertoni D, Magana P, et al. AlphaFold Protein Structure Database in 2024: providing structure coverage for over 214 million protein sequences. *Nucleic Acids Research* 52(D1):D368-D375 (2024).

5\. Notin P, Kollasch A, Ritter D, et al. ProteinGym: large-scale benchmarks for protein fitness prediction and design. *Advances in Neural Information Processing Systems* 36 (2023).

6\. Rego N, Koes D. 3Dmol.js: molecular visualization with WebGL. *Bioinformatics* 31(8):1322-1324 (2015).

7\. The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2025. *Nucleic Acids Research* 53(D1):D609-D617 (2025).
