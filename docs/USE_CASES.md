# Use cases

Three worked examples. Each is checked against ground truth that was decided
independently of this software, so the results can be verified rather than
taken on trust. All were produced with ESM-2 650M in masked-marginal mode by
`examples/use_cases.py`, and no experimental or clinical data is shown to the
model at any point.

Positions are reported as a constraint percentile: 100 means the most
mutation-intolerant residue in that protein, 50 means the median residue.

## 1. Finding cancer hotspots in TP53

TP53 is the most frequently mutated gene in human cancer. Tumour sequencing has
identified six positions that account for a large share of its missense
mutations. The question: given only the TP53 sequence, does the tool flag them?

| Hotspot | Sensitivity | Verdict | Constraint percentile | Worst substitution |
| --- | --- | --- | --- | --- |
| R175 | -8.24 | Critical | 93.9 | R175D (-13.01) |
| G245 | -10.52 | Critical | 98.2 | G245W (-12.68) |
| R248 | -11.65 | Critical | 99.2 | R248D (-14.43) |
| R249 | -7.21 | Sensitive | 89.8 | R249P (-10.69) |
| R273 | -7.54 | Sensitive | 91.1 | R273D (-11.29) |
| R282 | -6.06 | Sensitive | 83.2 | R282P (-10.62) |

Median constraint percentile 92.5. Four of six fall in the most constrained
10% of the protein, all six in the most constrained 17%. The protein's median
sensitivity is -2.13, so every hotspot is several times more constrained than a
typical residue.

**Why this is useful.** The same ranking applied to a gene with no hotspot
catalogue gives a prioritised shortlist for sequencing panels or functional
follow-up.

## 2. Recovering catalytic sites across an enzyme panel

UniProt curates active site and binding site residues from experimental
literature. Those annotations are ground truth the model never sees.

| Enzyme | Annotated sites | Median percentile | In top 10% |
| --- | --- | --- | --- |
| TEM-1 beta-lactamase | 4 | 97.8 | 4/4 |
| Phosphoglycerate kinase 1 | 29 | 90.1 | 15/29 |
| GAPDH | 8 | 91.2 | 4/8 |
| Lysozyme C | 3 | 83.6 | 1/3 |
| Carbonic anhydrase 2 | 8 | 79.5 | 4/8 |
| Cationic trypsin | 7 | 73.9 | 3/7 |

Panel median 86.8. Functional residues are consistently ranked as constrained,
with TEM-1 recovering perfectly. Trypsin is the weakest case, which is
informative: several of its annotated sites are substrate-binding rather than
catalytic, and binding residues are under weaker evolutionary constraint than
catalytic ones.

**Why this is useful.** For an uncharacterised protein with no structure paper
behind it, the top-ranked positions are candidate functional sites.

## 3. Choosing positions for protein engineering

Directed evolution and library design need the opposite question answered:
which positions can be changed without breaking the protein? Lysozyme C,
147 residues.

**Ten most tolerant positions**, candidates for randomisation:

| Position | WT | Sensitivity | Best substitution |
| --- | --- | --- | --- |
| 120 | G | -0.28 | G120P (+5.04) |
| 136 | T | -0.29 | T136R (+4.12) |
| 33 | H | -0.70 | H33A (+4.23) |
| 139 | Q | -0.98 | Q139S (+4.42) |
| 93 | L | -1.05 | L93A (+2.29) |
| 65 | T | -1.14 | T65S (+1.93) |
| 140 | A | -1.33 | A140S (+2.10) |
| 28 | A | -1.44 | A28R (+2.47) |
| 17 | L | -1.87 | L17Q (+0.86) |
| 5 | L | -2.11 | L5A (+0.97) |

**Ten most constrained positions**, to leave alone:

```
C98  C112  C82  G72  C24  C145  C48  C133  W126  C94
```

Eight of those ten are cysteines, and they are exactly the eight cysteines
UniProt annotates as forming lysozyme's four disulfide bridges:

```
C24-C145    C48-C133    C82-C98    C94-C112
```

The model was given a sequence and nothing else. It did not see the structure,
the disulfide annotations, or any experimental data, yet it identified every
residue in all four bridges as among the least substitutable in the protein.
The two non-cysteine entries, G72 and W126, are a buried glycine and a
conserved tryptophan in the substrate binding cleft.

**Why this is useful.** A library that randomises the tolerant positions and
preserves the constrained ones wastes far fewer variants on dead protein.

## Reproducing

```bash
python examples/use_cases.py
```

Outputs are written to `examples/results/`.
