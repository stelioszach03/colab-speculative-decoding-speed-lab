# Paper Package

This folder preserves an unpublished exploratory write-up and the historical
Colab measurements supporting it. It is not a peer-reviewed or final paper.
The prose was corrected to distinguish observed latency ratios from unsupported
causal claims about kernels/scheduling and from independent replication. The
CSV measurements are unchanged. This document does not report results from the
new serving harness.

## Contents
- `main.tex`: manuscript source
- `main.pdf`: compiled manuscript
- `references.bib`: bibliography database
- `data/phase2_results.csv`: Study 1 result table input
- `data/phase3_results.csv`: Study 2 aggregate + ablation input
- `data/phase3_latency_by_category.csv`: Study 2 per-category latency input

## Build
From this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Notes
- Figures are generated with `pgfplots` directly from CSV files in `data/`.
- Keep `paper/data/*.csv` aligned with notebook-exported metrics.
- `main.pdf` is the canonical rendered output tracked in the repository.
