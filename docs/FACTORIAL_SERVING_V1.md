# Controlled serving extension: model size, prefix reuse and hardware

Prepared before execution on24September2026. This is a separate prospective experiment; the original1,536-request pilot and all historical Colab records remain unchanged. The frozen matrix has now completed on both GPU classes; [verified results and limitations](../artifacts/factorial-serving-v1/RESULTS.md) are reported separately.

## Fixed design

Two Apache-2.0 checkpoints: Qwen2.5-1.5B-Instruct at989aa7980e4cf806f80c7fef2b1adb7bc71aa306 and Qwen2.5-7B-Instruct at a09a35458c702b33eeacc393d103063234e8bc28. Both use the same immutable vLLM0.10.2 image digest recorded by the original pilot. Tokenizer revisions match weight revisions; remote model code is not enabled.

Two sequential Secure Cloud allocations are planned: one RTX4090 24GB and one A100-SXM4-80GB. Actual GPU name, memory, driver, runtime versions and allocation quote are recorded. One GPU per pod; no public inference listener. If the requested class is unavailable or above the authorized price ceiling, its cells remain unexecuted rather than silently changing hardware.

Per hardware:2models ×2prefix-workload conditions ×2cache modes ×4repeats ×4concurrency stages ×64measured requests =8,192requests. Both hardware classes together plan16,384measured requests. Each stage has four separately recorded, disjoint warmup prompts. The fixed output target is128tokens using explicit min_tokens=max_tokens and ignore_eos; generated answer quality is not evaluated.

The workload is newly generated synthetic operational-summary text using four context lengths (64/192/448/896words). A shared-prefix stage has16different suffix tasks per length. A unique-prefix stage places a different document identity before each context. Stage and repeat identities prevent exact cross-stage document reuse; common chat-template tokens can still be cached. Prompt selection is without replacement within each64-request stage. Identical paired prompts/order are used for both cache modes and hardware/model comparisons. All workload hashes are frozen.

A fresh model server is started for every model/prefix/repeat/cache cell. Model and cache orders follow ABBA across four repeats; prefix order alternates. Concurrency uses four predeclared permutations of1/4/16/32. Warmups do not reuse measured prompts and their outcomes remain recorded. The order design reduces some order imbalance; it does not create independent hardware installations or remove every runtime confound.

Both GPUs use bfloat16, context4096, max32sequences, max4096batched tokens, tensor parallel1, GPU-memory-utilization0.8, temperature0 and server seed17. Identical memory fraction gives different absolute KV-cache capacity across GPUs; this is a system comparison, not isolation of compute cores. Request-order seeds are31/47/73/101; deterministic decoding is not a guarantee of identical text.

## Measurement and interpretation

Record all request outcomes, first text-chunk latency, p50/p95total latency, server-reported tokens and decode throughput, with timestamped500msNVIDIA telemetry. Chunk timing is not token-level timing. Sample GPU utilization/memory over the corresponding measured request interval, excluding separately timed warmups where possible. Telemetry is device-wide, not complete attribution to one process. Preserve missing telemetry explicitly.

For every hardware/model/prefix/concurrency combination, report all four repeats and descriptive means/ranges. Pair cache OFF/ON by repeat and workload ID; report latency/throughput differences together with exact output-hash agreement and failed/missing pairs. Do not call a speed difference lossless or output-equivalent if text differs. Show failures in request denominators; no selective reruns, outcome-based exclusions, significance or population confidence claims. There is only one instance of each GPU class, two sizes from one model family and synthetic traffic.

The model server and benchmark client use loopback only. External network is used for public model downloads during setup; no personal/private data enter prompts. Only this runner's process groups are terminated. Original public demos and their service identities stay unchanged.

## Freeze, supervision and finite spending

Before allocation, commit reviewed code and write a new immutable freeze containing complete configuration, model/workload hashes and executed-source hashes. Recover that same freeze with actual runtime metadata. The driver checks it before starting measurements, refuses pre-existing output directories and has a3-hour experiment timeout. The operator maintains a separate4-hour maximum allocation lease with an independent VPS watchdog; the experiment cannot stop provider billing by itself.

The shared RunPod campaign cap is$18 including setup failures and storage, with at most one active pod. Planned GPU rate ceilings are$1/hour for RTX4090 and$2/hour for A100; the actual allocation quote must pass before execution. A new lease cannot be admitted if its full conservative allowance would exceed the remaining campaign envelope. No balance top-ups or persistent network volumes. Setup-only recovery requires its failure reason recorded and the same frozen scientific settings; measured runs are never replaced because of an unfavorable result.

The three-hour Codex heartbeat is additional supervision, not the cost stop mechanism. Original artifacts, failed setup logs, incomplete cells and provider billing uncertainty remain visible in final reporting. Public results require reconciliation and a source/privacy review before publication.
