# CrossVul-DG — paper source

- `main.tex` — IEEEtran conference paper (double-blind), centered on the
  proposed **CC-DRO** method.
- `main.pdf` — a compiled preview (built with IEEEtran V1.8, US-Letter, 5 pp).

## Recompiling
The paper uses `\documentclass[conference]{IEEEtran}`. To rebuild:
- **Overleaf (easiest):** upload `main.tex`, set the compiler to pdfLaTeX —
  IEEEtran is already installed there.
- **Local TeX Live:** ensure `texlive-publishers` is installed (provides
  `IEEEtran.cls`), then `pdflatex main.tex` twice.

## Before submitting
Every number is an illustrative placeholder (em-dashes / synthetic values).
Run the sweep in `../code` (`scripts/run_full_sweep.sh` then
`scripts/aggregate_results.py`), then paste the generated `main_table.tex`
and fill Tables I–V and Fig. 1 with measured results. Remove the red
"Placeholder notice" boxes once real numbers are in.
