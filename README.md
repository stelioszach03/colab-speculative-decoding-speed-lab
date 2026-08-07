# Speculative Decoding Speed Lab

Two speculative-decoding stacks measured on 1× NVIDIA A100-SXM4-80GB (Colab): Transformers assisted decoding, and vLLM + EAGLE-3.

[![License: MIT](https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square)](LICENSE)

Solo research project. **Manuscript only — not published, not peer-reviewed, not submitted.**

## Results

| Study | Setting | Result | Evidence |
|---|---|---|---|
| 1 | Transformers assisted decoding, 9 config×split cells | **Every cell was slower than baseline** — best 0.357×, worst 0.251×, i.e. 2.8–4× *slower* | [`paper/data/phase2_results.csv`](paper/data/phase2_results.csv) |
| 2 | vLLM + EAGLE-3, `num_speculative_tokens=3` | **1.387× aggregate latency speedup**; throughput 110.93 tok/s | [`paper/data/phase3_results.csv`](paper/data/phase3_results.csv) |
| 2 | same run, math-reasoning prompts only | 1.460× (1.575 s → 1.079 s) | [`paper/data/phase3_latency_by_category.csv`](paper/data/phase3_latency_by_category.csv) |
| 2 | same run, summarization prompts only | 1.323× | same file |

**The headline number is 1.387×, not 1.46×.** The 1.46× is one prompt category. Both are **latency** ratios — throughput is a separate quantity, reported as 110.93 tok/s. The Study 1 result is the more interesting one: assisted decoding is not free, and on this stack the scheduling overhead swamped any acceptance benefit. Quality barely moved (ΔF1 vs baseline ≤ 0.0005), so the regression is runtime cost, not output collapse.

### Models — one family, four checkpoints

| Study | Role | Checkpoint |
|---|---|---|
| 1 | target / draft / mismatched draft | `Qwen2.5-7B-Instruct` · `Qwen2.5-0.5B-Instruct` · `Qwen2.5-0.5B` |
| 2 | target / speculator | `Qwen3-8B` · `RedHatAI/Qwen3-8B-speculator.eagle3` |

All targets and drafts are Qwen. **This is not a cross-family study.**

## Run

Open `Colab_Scale_Study_of_Speculative_Decoding.ipynb` on a Colab A100 High-RAM runtime and run cells top to bottom (Study 1, then Study 2). Artifacts are exported to `extracted_metrics.json` and `paper/data/*.csv`, which the manuscript consumes via `pgfplotstable`. Rebuild the write-up:

```bash
cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex && bibtex main && \
  pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

## Limitations

- **24 prompts per study.** Study 1 uses 24 SQuAD-derived prompts, Study 2 uses 24 (12 summarization + 12 math) from Red Hat AI's `speculator_benchmarks`. A speed probe, not a benchmark suite; no confidence intervals.
- **The two studies are not directly comparable** — different targets, prompt sets and inference engines. Study 2 beating Study 1 is confounded by all three, and the manuscript says so.
- **Study 2 does not verify output equivalence.** With sampling on (temperature 0.6, top-p 0.95), exact match vs the baseline generation is 0.0 and quality F1 vs reference is 0.288. This measures speed, not losslessness.
- **Single run per configuration**, one GPU, one software stack (PyTorch 2.8.0+cu128, Transformers 4.57.6, vLLM 0.11.0). Results are specific to that profile.
- One model family only, as noted above.

## License

MIT — see [LICENSE](LICENSE).
