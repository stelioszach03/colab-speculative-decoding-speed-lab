# Colab Speculative Decoding Speed Lab

Empirical study of two speculative-decoding stacks on **1× NVIDIA A100-SXM4-80GB**
(Google Colab): what actually made decoding faster, and what made it dramatically
slower.

- **Study 1 — Transformers assisted decoding**: target + draft, ablating block size
  and draft/target mismatch.
- **Study 2 — vLLM + EAGLE-3**: production-style backend, ablating the number of
  speculative tokens.

## Results

| Study | Setting | Result | Evidence |
|---|---|---|---|
| **1** | Transformers assisted decoding, 9 config×split cells | **Every cell was slower than baseline.** Best 0.357×, worst 0.251× — i.e. 2.8–4× *slower* | [`paper/data/phase2_results.csv`](paper/data/phase2_results.csv) |
| **2** | vLLM + EAGLE-3, `num_speculative_tokens=3` | **1.387× aggregate latency speedup**, 110.9 tok/s global throughput | [`paper/data/phase3_results.csv`](paper/data/phase3_results.csv) |
| **2** | same run, math-reasoning prompts only | **1.460×** (1.575 s → 1.079 s) | [`paper/data/phase3_latency_by_category.csv`](paper/data/phase3_latency_by_category.csv) |
| **2** | same run, summarization prompts only | 1.323× | same file |

**The headline number is 1.387×, not 1.46×.** The 1.46× is one prompt category, and
both are **latency** ratios — throughput is reported separately as tok/s.

The Study 1 result is the more interesting one: assisted decoding is not free, and
on this stack the scheduling overhead swamped any acceptance benefit. Quality
barely moved (ΔF1 vs baseline ≤ 0.0005), so the regression is runtime cost, not
output collapse.

### Models — one family, four checkpoints

| Study | Role | Checkpoint |
|---|---|---|
| 1 | target | `Qwen/Qwen2.5-7B-Instruct` |
| 1 | draft (close-match) | `Qwen/Qwen2.5-0.5B-Instruct` |
| 1 | draft (mismatch) | `Qwen/Qwen2.5-0.5B` |
| 2 | target | `Qwen/Qwen3-8B` |
| 2 | speculator | `RedHatAI/Qwen3-8B-speculator.eagle3` |

All targets and drafts are Qwen. This is **not** a cross-family study.

## How it runs

1. Open `Colab_Scale_Study_of_Speculative_Decoding.ipynb` on a Colab A100 High-RAM
   runtime.
2. Run cells top to bottom (Study 1 path, then Study 2 path).
3. Result artifacts are exported to `extracted_metrics.json` and
   `paper/data/*.csv`, which the manuscript consumes directly via `pgfplotstable`.

Rebuild the write-up from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Repository structure

| Path | Purpose |
|---|---|
| `Colab_Scale_Study_of_Speculative_Decoding.ipynb` | Benchmark notebook (Study 1 + Study 2) |
| `paper/main.tex` · `paper/main.pdf` | Manuscript source and build |
| `paper/data/*.csv` | The three result tables above |
| `extracted_metrics.json` | Per-configuration latency/throughput snapshot from the runs |
| `docs/REPRODUCIBILITY.md` | Runtime, library versions, tracked parameters |

## Limitations

- **24 prompts per study.** Study 1 uses 24 SQuAD-derived prompts, Study 2 uses 24
  prompts (12 summarization + 12 math) from Red Hat AI's `speculator_benchmarks`.
  This is a speed probe, not a benchmark suite — no confidence intervals are
  claimed.
- **The two studies are not directly comparable.** Different targets (Qwen2.5-7B vs
  Qwen3-8B), different prompt sets, different inference engines. Study 2 beating
  Study 1 is confounded by all three; the manuscript says so explicitly.
- **Study 2 does not verify output equivalence.** With sampling on (temperature
  0.6, top-p 0.95), exact match vs the baseline generation is 0.0 and quality F1 vs
  reference is 0.288 — this measures speed, not lossless-ness.
- **Single run per configuration**, single GPU, single Colab software stack
  (PyTorch 2.8.0+cu128, Transformers 4.57.6, vLLM 0.11.0). Results are specific to
  that profile.
- **This is an unpublished manuscript**, not a peer-reviewed publication.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
