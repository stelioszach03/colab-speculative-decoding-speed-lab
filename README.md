# Colab Speculative Decoding Speed Lab

Colab-scale empirical study of speculative decoding speed-quality trade-offs on **1x NVIDIA A100 (80GB)**.

## Overview
This repository benchmarks two practical inference stacks for LLM decoding acceleration:
- **Study 1 (Transformers-assisted decoding)**: target+draft assisted generation with block-size and draft-target mismatch ablations.
- **Study 2 (vLLM + EAGLE-3)**: speculative decoding with a production-style backend and speculative-token ablation.

The project includes:
- an end-to-end notebook pipeline,
- a manuscript-ready LaTeX paper,
- exported result artifacts used to generate the paper tables/figures.

## Key Results
- **Study 1:** no acceleration (best QA speedup remains `<1x`).
- **Study 2:** up to **1.387x aggregate speedup** and **1.460x category speedup** (math prompts).

## Repository Structure
| Path | Purpose |
|---|---|
| `Colab_Scale_Study_of_Speculative_Decoding.ipynb` | Main benchmark notebook (Study 1 + Study 2 paths) |
| `paper/main.pdf` | Final paper PDF |
| `paper/main.tex` | LaTeX source for the manuscript |
| `paper/references.bib` | Bibliography source |
| `paper/data/*.csv` | Tables consumed by `pgfplots/pgfplotstable` in the manuscript |
| `paper/README.md` | Paper-specific build/run notes |
| `extracted_metrics.json` | Consolidated metrics snapshot exported from notebook runs |
| `docs/REPRODUCIBILITY.md` | Environment, runtime, and reproducibility checklist |

## Quick Start
1. Open `Colab_Scale_Study_of_Speculative_Decoding.ipynb` in VS Code.
2. Connect notebook kernel to Google Colab (A100 High-RAM runtime).
3. Run cells top-to-bottom.
4. Export/update result artifacts and sync `paper/data/*.csv` if needed.

## Rebuild the Paper
From `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Notes
- The notebook records package/runtime details at execution time.
- Manuscript plots are generated directly from CSV inputs (no external image files required).
- Results are specific to this runtime profile (single A100, Colab stack).

## Citation
If this repository is useful in your work, cite it using [CITATION.cff](/Users/stelioszacharioudakis/Documents/LLM_Speculative_Decoding_Speed_Lab/CITATION.cff).

## License
This project is licensed under the MIT License. See [LICENSE](/Users/stelioszacharioudakis/Documents/LLM_Speculative_Decoding_Speed_Lab/LICENSE).
