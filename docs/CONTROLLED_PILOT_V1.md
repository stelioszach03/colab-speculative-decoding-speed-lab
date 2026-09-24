# Controlled serving pilot v1

This protocol is fixed before a new GPU measurement. It is a small engineering experiment, not a model-quality benchmark or evidence that any serving technique wins. The historical Colab artifacts remain separate.

## Frozen design

- Model and tokenizer: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, Apache-2.0.
- Engine: vLLM `0.10.2`, official Linux amd64 image `vllm/vllm-openai@sha256:df2607b26bdda2875de4832f4d08da0055b4b6e3570347f3a849bcc652771dd6`.
- Hardware: one rented NVIDIA GPU, intended RTX 4090 24 GB; record the actual device, driver and provider receipt. A different GPU is a protocol deviation, not another replicate to pool silently.
- BF16 weights, tensor parallelism 1, maximum context 4,096, maximum sequences 32, maximum batched tokens 4,096, GPU memory utilization 0.8. No quantization or speculative decoding treatment. Default vLLM compilation/continuous batching remain enabled and their actual startup configuration is preserved.
- Two settings: prefix caching explicitly OFF and ON. Restart the server at the beginning of every cell. Three sequential paired replicates, cache-mode order OFF/ON, ON/OFF, OFF/ON. This balances direction as closely as three pairs allow; it does not eliminate time drift.
- Every cell measures concurrency 1, 4, 16 and 32 with exactly 64 requests per stage. Fixed stage orders are `[1,4,16,32]`, `[32,16,4,1]`, `[4,32,1,16]` for replicates 1–3. Workload shuffle seeds 18/19/20, matched across the cache modes of each replicate.
- Four serial warmups before every measured stage, logged and excluded. Cache state persists between stages within the cell. This is an explicitly warmed, repeating workload, not a cold-cache test. The ordering helps identify order sensitivity but cannot establish cache causality by itself.
- 64 original synthetic prompts: context lengths 64/192/448/896 **words**, eight unique prompts for each length and each of two prefix strata (shared notes versus an early document-specific header). Actual input-token counts come from the server. No third-party dataset or private text is used.
- Temperature zero; exactly 128 requested decode tokens with vLLM `min_tokens=128`, `ignore_eos=true`, `max_tokens=128`. Verify actual completion-token counts. Forced decode length is useful for controlled work volume, not a natural answer-quality measure.

There are **1,536 measured requests + 96 warmups**, at most **208,896 requested output tokens**. No retries. Failed and interrupted requests remain in the artifacts. Do not keep only successful cells, change settings after viewing results, or pool changed configurations as repeats.

The canonical machine-readable plan is [controlled-pilot-v1.json](../protocols/controlled-pilot-v1.json). Its workload hash is regenerated deterministically and checked before running. The version-control commit containing this plan must be recorded before paid execution. This is a repository preregistration timestamp, not an external peer-reviewed registration.

## Execution and hard bounds

The operator provisions and tears down the GPU separately. Before provisioning, set an independent cloud termination deadline of at most 60 minutes and a spend envelope. The runner's 45-minute experiment deadline includes all six model startups; each startup has an additional 8-minute readiness limit. Container pull, upload and external provisioning are outside the runner's timer. Local process termination does not stop cloud charges.

Use the exact pinned container with an entrypoint override that keeps it alive. Only SSH needs exposure. Transfer this repository (no credentials), then inside its root:

```bash
# No endpoint, downloads, output writes or hardware allocation:
python -m inference_lab.pilot

# Execute only inside the provisioned container after recording its image identity:
python -m inference_lab.pilot --output /workspace/results/controlled-pilot-v1 --execute

# After completion or interruption; never changes the raw source files:
python -m inference_lab.pilot_report /workspace/results/controlled-pilot-v1
```

The server binds only `127.0.0.1:8000`; no model API key is needed. The pilot starts and terminates only the server, client and telemetry process groups it creates. It refuses an existing HTTP server on port 8000, a different vLLM version, or more than one visible GPU. It never creates cloud resources or reads provider credentials. Copy evidence off the instance before teardown; verify the provider reports the GPU terminated.

## Direct evidence

Every cell preserves the server launch command, complete startup/runtime log, package versions, model/tokenizer revision, `/metrics` snapshots, client request traces and summaries. Request and stage timestamps use UTC; NVIDIA-smi sampling explicitly uses `TZ=UTC`.

`gpu.csv` is a directly sampled **device-wide** NVIDIA-smi trace at 500 ms: timestamp, index, UUID, model name, used/total VRAM, GPU/memory utilization, watts and temperature. NVIDIA-smi wraps NVIDIA's management interfaces; these measurements are not estimated from HTTP latency. The separate compute-process snapshots help identify competing processes, but containers may hide complete PID attribution. Device utilization is not attributed to an individual process. Peak observed VRAM is a sampled device peak, not a guaranteed instantaneous maximum; vLLM preallocates cache memory and used VRAM includes that allocation. Report the assigned GPU's sharing/isolation terms where available.

`report.json` and `stages.csv` join telemetry to measured stage intervals. They preserve all request failures, fixed-output-length mismatches, missing stages, sample counts, descriptive TTFT/latency percentiles, throughput, VRAM and utilization. They also compare output text hashes for matching successful request IDs across cache modes. Hash agreement is not an answer-quality score. A cache performance contrast with differing hashes must disclose the difference; do not label it lossless.

Each stage has only 64 requests; its p95/p99 are descriptive sample percentiles. Three repeated measurements on one GPU do not justify a broad population confidence interval or hardware-general speed claim. Report each replicate, not only the best. Cloud charge/cost remains null until an actual provider billing record is attached separately.

## Primary sources verified for this protocol

- [Official model card and license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306): pinned revision and Apache-2.0 license.
- [vLLM 0.10.2 Docker deployment](https://docs.vllm.ai/en/v0.10.2/deployment/docker.html): official `vllm/vllm-openai` image; immutable amd64 digest resolved from the Docker registry before execution.
- [vLLM 0.10.2 engine arguments](https://docs.vllm.ai/en/v0.10.2/configuration/engine_args.html): explicit cache flags, tokenizer/model revision, generation-config and serving bounds.
- [vLLM 0.10.2 OpenAI-compatible server](https://docs.vllm.ai/en/v0.10.2/serving/openai_compatible_server.html): streaming and generation parameter extensions.

Offline checks use fixture streams only: `python -m unittest discover -s tests -v`. Those checks are not serving results. When actual GPU artifacts exist, publish the measured outcomes and deviations under a separate results document without rewriting this frozen design.
