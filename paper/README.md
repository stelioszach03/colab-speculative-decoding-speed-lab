# arXiv Paper Package

This folder contains a full LaTeX manuscript based on the executed notebook runs.

## Files
- `main.tex`: manuscript
- `references.bib`: bibliography
- `data/phase2_results.csv`: Phase-2 table data
- `data/phase3_results.csv`: Phase-3 table/ablation data
- `data/phase3_latency_by_category.csv`: category-level latency table

## Build
From `paper/`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Notes
- Figures are generated directly in LaTeX via `pgfplots` from CSV files in `data/`.
- Reported values are tied to the exact Colab/A100 runs saved in the notebook.
