# Contributing

Contributions are welcome, and issues are as useful as pull requests. If the
tool gives you a result that looks wrong, that is worth reporting even without
a fix.

## Reporting a problem

Open an issue with the UniProt accession or sequence, the model and scoring
mode you used, and what you expected. Scores are deterministic, so anything
reproducible on your side is reproducible here.

## Ideas that would genuinely help

These are the gaps I know about, roughly in order of value:

- **A head-to-head against AlphaMissense** on shared human proteins. The most
  obvious missing comparison, and the first thing a reviewer asks for.
- **Masked marginals at 650M on ProteinGym.** The scoring-protocol comparison
  was run at 150M only, so the 2x2 has a hole. It is a long run, not a hard one.
- **Experimental validation of the tolerant end.** Positions reported as
  substitution-tolerant are the model's own output; nothing external confirms
  they tolerate mutation.
- **Domain-aware windowing.** Sequences beyond 1,022 residues are split on
  fixed strides, which can cut across domain boundaries.
- **Multimer support.** AlphaFold monomer models mean interface residues that
  matter only within a complex are under-weighted.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

To reproduce the benchmarks, see [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Every
table in the paper is generated from files in `validation/results/` by the
scripts in `validation/`, so please regenerate rather than editing numbers by
hand.

## Conventions

- The deployed app must keep working on CPU, Apple Silicon and CUDA.
- Gradio is pinned to the 5.x line, because 6.x fails to render interactive
  inputs in Safari.
- Tables and figures are generated from result files, never transcribed.

## Licence

Contributions are accepted under the Apache-2.0 licence covering this project.
