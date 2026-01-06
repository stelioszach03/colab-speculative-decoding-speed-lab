# Colab Speculative Decoding Speed Lab

A reproducible Colab-scale study of speculative decoding speed/quality trade-offs on **1x NVIDIA A100**.

## What this repo contains
- `Colab_Scale_Study_of_Speculative_Decoding.ipynb`: end-to-end benchmark notebook
- `paper/`: arXiv-style manuscript and CSV tables used by plots
- `arxiv_submission/`: upload-ready arXiv source package
- `extracted_metrics.json`: consolidated run metrics snapshot

## Study design
- **Study 1 (Transformers-assisted decoding)**
  - Qwen2.5 target + draft variants
  - Block-size and mismatch ablations
  - Baseline vs speculative latency/speedup comparisons
- **Study 2 (vLLM + EAGLE-3)**
  - Qwen3-8B target + RedHatAI EAGLE-3 speculator
  - Speculative-token ablation (`n=3,5`)
  - Throughput and per-category latency comparisons

## Main reported results
- Study 1: no acceleration (best QA speedup `< 1x`)
- Study 2: practical gains up to **1.387x aggregate** and **1.460x** on math prompts

## Run instructions
1. Open the notebook in VS Code connected to Colab kernel (A100 High-RAM).
2. Run cells in order.
3. Study 1 path: Cells 1-10.
4. Optional extension: temperature ablation cell.
5. Study 2 path: Cells A-E.

## Reproducibility notes
- The notebook logs key package/runtime versions during execution.
- `paper/data/*.csv` files are generated from notebook artifacts and consumed by LaTeX plots.

## Paper build
From `paper/`:
```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```
