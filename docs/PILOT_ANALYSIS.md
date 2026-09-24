# Analysis of the controlled serving pilot

The frozen execution protocol and measurement client are independent of this analysis code. Updating plotting or integrity checks must not change the running experiment's workload, server flags, request bounds or telemetry configuration.

After downloading actual evidence from the GPU container:

```bash
python -m inference_lab.pilot_report /path/to/controlled-pilot-v1
python -m pip install -r requirements-analysis.txt
python -m inference_lab.pilot_plots /path/to/controlled-pilot-v1 \
  --output /path/to/controlled-pilot-v1/figures
```

The analysis verifies the frozen protocol, recomputes every stage summary from raw request records, rejects duplicate requests/workload IDs, checks measured counts and actual response model IDs, and reports missing telemetry and protocol deviations. It never substitutes made-up measurements for missing cells. The figure command refuses an empty experiment or unresolved integrity issue. A partial run requires explicit `--allow-incomplete` and is labeled **PARTIAL EXPERIMENT**.

The six-panel PNG/SVG/PDF figure shows throughput, sample p95 first-text latency, sample p95 overall request latency, device utilization, device memory and exact output-text agreement. Each of the three replicates appears separately; there are no extrapolated population intervals or best-run-only curves. First-text latency and throughput use successful requests; overall latency also includes failures. Missing usage stays missing. Figure footnotes show the number of failures and completion-token mismatches. The final agreement panel is not a quality score or proof of lossless inference.

Raw artifacts remain unchanged. The figure manifest records Matplotlib's actual version, input hashes and rendered figure hashes. `figures/`, `report.json` and `stages.csv` are excluded from the raw-source hash set to make analysis reruns stable. Provider cost remains null unless separately supported by a billing record.

GPU plots use UTC samples in each measured stage's interval. The 500 ms sampling cadence may miss instantaneous peaks; observed VRAM is device-wide and includes vLLM's preallocated cache. Wall-clock correction or missing samples must be investigated rather than treated as zero GPU use. Sharing/isolation, hardware substitution and interrupted runs belong in the results report as limitations or deviations.
