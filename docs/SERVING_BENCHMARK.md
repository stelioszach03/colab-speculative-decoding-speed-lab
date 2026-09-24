# Serving benchmark protocol v0.1

This slice implements the client and artifact contract. It does not provision a GPU, start a server, claim a serving speedup, or mix new measurements with the historical Colab study. Python 3.11 or newer is sufficient for client tests; no package installation is needed.

## Run boundary

`python -m inference_lab.benchmark --model MODEL` prints the exact request/token upper bounds and exits without making requests or creating an output directory. `--execute` enables requests. Only loopback HTTP(S) URLs are accepted; a user-managed SSH tunnel can reach a remote serving process. The supplied server can still consume rented GPU time: plan-only does not allocate or stop that infrastructure.

The client follows no redirects, retries no requests, and does not send model requests to external APIs. If authentication is necessary, set a secret in an environment variable and name it with `--api-key-env VARIABLE_NAME`. Neither its value nor headers/provider error bodies enter the artifacts. Use synthetic or properly licensed prompts; the workload itself is saved in the output directory.

Each concurrency level sends serial warmups first, then exactly `--requests` measured requests with at most that many worker threads in flight. The same seeded workload order repeats across stages. Warmups are recorded with `warmup: true` and excluded from summaries. The current six-prompt workload repeats, which can favor prefix caching. It is a smoke workload, not an independent throughput evaluation. Concurrency is bounded to 64, at most five unique stages, 10,000 requests per stage and 4,096 requested output tokens per request.

The timeout applies to blocking socket activity, with a deadline checked between reads. A blocked read may take up to one socket timeout beyond the checked deadline. The client stops after 4 MiB of streamed response data. Failures remain recorded; a stage with failures causes exit status 1. Interrupted runs retain completed request records and a manifest status of `interrupted`. Existing output directories are never overwritten.

## Artifacts and measurements

- `manifest.json`: schema, runner SHA-256, requested model, client environment, server metadata explicitly marked user-reported, decoding parameters, counts, workload hash, start/end times and completion status.
- `workload.jsonl`: exact shuffled prompt order. The manifest hash refers to the original input file before shuffling.
- `requests.jsonl`: every warmup/measured request, response/error category, server-reported model/token counts, text-chunk byte sizes and monotonic arrival times. Generated text is hashed rather than stored.
- `summary.json`: per-stage successful and failed counts, failures by kind, throughput, mean and linearly interpolated p50/p95/p99 request and streaming metrics.

| Metric | Exact interpretation |
| --- | --- |
| `ttft_s` | Request initiation through first nonempty **text content chunk**, including client/transport/server delays. Empty, role, reasoning and usage-only events do not count. |
| `itl_stream_s` | Consecutive text-chunk arrival gaps. As with client streaming measurements, one chunk can contain several tokens; this is **not an engine-level per-token trace**. |
| `tpot_s` | First-to-last text-chunk span divided by server-reported completion tokens minus one. Null if usage is absent, output is a single chunk, or non-text/reasoning output is reported. Undisclosed server reasoning can still affect counts. |
| Request latency | Request initiation through stream completion/failure, reported separately for all requests and successful requests. Client worker queue wait is excluded. |
| Requests/s | Successful measured requests divided by full measured stage wall time, including failures. Warmup time is excluded. |
| Completion tokens/s | Server-reported completion tokens of successful requests divided by stage time. Null when any successful request lacks usage. No character-to-token estimate is substituted; a server's count may include reasoning tokens. |
| p95 / p99 | Descriptive sample percentiles, with sample count shown. Tiny samples cannot establish stable tail performance. |

Timing terminology follows the [vLLM serving benchmark documentation](https://docs.vllm.ai/en/latest/benchmarking/cli/), with explicit chunk granularity and non-text-output limitations. HTTP completion establishes only a successful text stream; it is not a correctness or quality score. `finish_reason: length` is retained and can indicate truncation. Only one text choice is supported.

Server settings can be attached using `--server-metadata path/to/settings.json`. Include actual engine version, image digest, model and tokenizer revisions, GPU name/count, dtype, quantization, prefix-cache configuration, speculative configuration and launch command **without secrets**. These values are user supplied, not measured. GPU utilization/VRAM and dollar costs remain null until a separate validated measurement path is implemented.

## Gates before a paid study or publication

1. Specify a fixed model revision/license, one GPU class, exact engine/container version and a bounded spending/time allowance. Provisioning and shutdown must be verified independently; this client does not manage them.
2. Use the same prompts, input/output-length strata and request order across matched baseline/treatment runs. Make warm-cache and cold-cache policies explicit. Continuous batching is a server behavior, not an on/off client switch.
3. Add repeated runs and an output-quality/equivalence check before comparing quantization or speculative decoding. Never interpret unequal outputs as a pure serving speedup.
4. Use enough measured requests for tail metrics, preserve OOM/timeout/error requests, and report utilization/saturation limits. A concurrency target is not evidence the server sustained it.
5. Review logs, prompt licenses and metadata for private information; publish immutable artifacts and a report with actual hardware and failure counts. The ignored `results/` directory requires deliberate selection for publication.

## Offline checks

```bash
python -m unittest discover -s tests -v
python scripts/check_recorded_results.py
```

Tests use an in-process loopback fixture server and temporary output directories. Its token counts are fixtures for the metric contract, never GPU performance evidence. Historical Colab CSVs/notebook outputs are preserved unchanged.
