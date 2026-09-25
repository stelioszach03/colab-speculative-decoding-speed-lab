# Inference Systems Lab

A serving-measurement toolkit growing from the recorded Speculative Decoding Speed Lab. The repository name and historical notebook remain unchanged.

**Status:** a [controlled GPU pilot](artifacts/controlled-pilot-v1/RESULTS.md) completed 1,536 measured requests on one RTX 4090. A separate [frozen two-GPU extension](artifacts/factorial-serving-v1/RESULTS.md) completed 16,384 measured requests across RTX 4090 and A100 80GB, two Qwen2.5 model sizes, prefix patterns, cache modes and four repeats. All requests completed at the HTTP level, but paired output hashes often differed; neither report makes a **lossless speedup or answer-quality claim**. The earlier Colab measurements below remain separate historical, unpublished experiments.

## Streaming serving benchmark

The standard-library Python client targets a locally served OpenAI-compatible `/v1/chat/completions` endpoint. It records each request's status, first text-chunk arrival, subsequent chunk gaps, total latency, server-reported tokens, warmups, workload hash and server settings. Concurrency stages use a fixed-size closed-loop worker pool. No model is downloaded or started by the client.

```bash
# From the repository root; Python 3.11+. Prints a plan, no network calls.
python -m inference_lab.benchmark --model YOUR_ACTUAL_SERVED_MODEL_ID

# Once your model server is running on loopback, explicitly execute:
python -m inference_lab.benchmark --model YOUR_ACTUAL_SERVED_MODEL_ID \
  --base-url http://127.0.0.1:8000/v1 --concurrency 1 4 \
  --requests 16 --warmup 1 --max-tokens 128 \
  --output results/first-local-run --execute
```

Read [the serving protocol and metric definitions](docs/SERVING_BENCHMARK.md) before using results. The six original bundled prompts are synthetic smoke workloads. They do not establish representative serving performance.

The [controlled GPU pilot v1](docs/CONTROLLED_PILOT_V1.md) fixes a model revision, container digest, 64 synthetic prompts, cache OFF/ON settings and three paired concurrency sweeps. [Raw evidence and measured figures](artifacts/controlled-pilot-v1/) include NVIDIA-smi telemetry, request traces and output-hash comparisons. At concurrency 32, observed throughput ranged from 3,594–3,614 tokens/s with caching OFF and 4,160–4,712 with it ON; differing text and warm-state/order effects limit the comparison. `python -m inference_lab.pilot` prints the plan without allocating hardware or contacting an endpoint. A [provider billing record](artifacts/controlled-pilot-v1/billing.json) reports $0.225616 for the entire allocation, including setup and artifact collection. Output quality, quantization and speculative-decoding improvements are not established by this pilot.

The [factorial extension](artifacts/factorial-serving-v1/RESULTS.md) paired cache modes across 4 repeats on two GPU classes. At concurrency 32, cached/uncached throughput ratios for shared-prefix synthetic prompts ranged from 1.23x to 1.47x across the four GPU/model combinations; unique-prefix ratios were approximately 1.00x. These are measured throughput ratios with frequent exact-output differences, not equivalent-output improvements. Both GPU pods were terminated; their allocation-rate estimate was $3.45, while the exact provider bill is unavailable to the scoped API key. The [protocol](docs/FACTORIAL_SERVING_V1.md), [derived data](artifacts/factorial-serving-v1/aggregates.csv), and [offline analyzer](scripts/analyze_factorial_serving.py) give the scope and reproduction path.

## Recorded results

| Recorded setting | Observation | Evidence |
|---|---|---|
| Transformers assisted decoding, nine configuration/split cells | 0.251–0.357× baseline latency speed ratio: slower in every recorded cell | [Phase2 CSV](paper/data/phase2_results.csv) |
| vLLM + EAGLE-3, three speculative tokens | 1.387× aggregate latency speedup; 110.93 generated tokens/s | [Phase3 CSV](paper/data/phase3_results.csv) |
| Same Phase3 configuration, math prompts | 1.460× category-specific latency ratio | [Category CSV](paper/data/phase3_latency_by_category.csv) |

The aggregate figure is **1.387×**, not the category-specific 1.460×. Latency ratios and generated-token throughput are different measurements. These observations do not isolate scheduling overhead as the cause of the Study1 slowdown.

Study1 uses Qwen2.5 target/draft checkpoints; Study2 uses Qwen3-8B with `RedHatAI/Qwen3-8B-speculator.eagle3`. This is one model family, not a cross-family comparison. Different prompts, target models and engines prevent a controlled comparison between the two studies.

## Inspect or reproduce

Read the saved outputs in [the notebook](Colab_Scale_Study_of_Speculative_Decoding.ipynb), [extracted metrics](extracted_metrics.json) and `paper/data/` without running inference. The [reproducibility notes](docs/REPRODUCIBILITY.md) describe the recorded setup and artifact flow.

A fresh run requires a compatible GPU environment, model downloads and explicit compute allocation. The recorded device was one NVIDIA A100-SXM4-80GB, with PyTorch 2.8.0+cu128, Transformers 4.57.6 and vLLM 0.11.0. Inspect notebook installation cells before executing them; current package compatibility is not guaranteed by saved output.

To rebuild the optional write-up with a local TeX installation:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Offline artifact check

```bash
python3 scripts/check_recorded_results.py
python3 -m unittest discover -s tests -v
```

The first check compares recorded values. The second exercises the HTTP/SSE client with a local fixture server. Neither runs model inference or independently validates the historical experiment.

## Limits

Each study uses 24 prompts and a single run per configuration. There are no confidence intervals or multi-device replications. Phase3 samples outputs and does not establish output equivalence or losslessness; its recorded exact match against baseline is zero. Acceptance-rate proxies are not necessarily engine-level acceptance telemetry. Recorded results are useful for inspecting this setup, not selecting a universally fastest serving stack.

[MIT license](LICENSE). Models and datasets retain their own licenses.
