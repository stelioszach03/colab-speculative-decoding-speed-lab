# Reproducibility Notes

## Hardware Profile
- Runtime: Google Colab
- GPU: NVIDIA A100-SXM4-80GB (single GPU)

## Notebook Execution Order
1. Study 1 path (Cells 1-10)
2. Optional Study 1 sampling extension
3. Study 2 path (Cells A-E)

## Runtime/Library Signals Captured in Notebook
- Python version
- PyTorch version
- Transformers version
- vLLM version (Study 2 section)
- CUDA/GPU availability and device name

## Core Parameters Tracked
- Study 1:
  - target/draft model IDs
  - block size (`num_assistant_tokens`)
  - confidence threshold
  - deterministic decode configuration
- Study 2:
  - target/speculator IDs
  - `num_speculative_tokens` grid
  - `max_model_len`
  - `gpu_memory_utilization`
  - sampling params and seed

## Artifact Flow
- Notebook generates benchmark outputs and summary metrics.
- Exported values are reflected in:
  - `extracted_metrics.json`
  - `paper/data/phase2_results.csv`
  - `paper/data/phase3_results.csv`
  - `paper/data/phase3_latency_by_category.csv`

## Sanity Checks Before Claiming Reproducibility
- Confirm Colab A100 runtime is active.
- Confirm all study cells complete without runtime fallback errors.
- Confirm manuscript rebuild succeeds from `paper/main.tex`.
- Confirm table values in paper match `paper/data/*.csv`.
