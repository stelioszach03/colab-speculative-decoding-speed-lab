"""Verify and summarize the frozen two-GPU serving extension without model calls.

Input archives remain private. Outputs contain only per-cell measurements and
aggregate hash-match counts, never prompts, completions, or credentials.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


SLOTS = ("rtx4090", "a10080")
CELL = re.compile(r"r(\d+)-qwen-(1\.5b|7b)-(shared|unique)-cache-([01])")
STAGE = re.compile(r"^output/(r\d+-qwen-(?:1\.5b|7b)-(?:shared|unique)-cache-[01])/c(1|4|16|32)/client/(summary\.json|requests\.jsonl)$")
CONCURRENCY = (1, 4, 16, 32)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_slot(slot: str, evidence: Path, freeze: dict) -> tuple[list[dict], dict, dict]:
    archive = evidence / f"{slot}-output.tar.gz"
    execution = json.loads((evidence / f"{slot}-execution.json").read_text())
    lease = json.loads((evidence / f"{slot}-lease.json").read_text())
    archive_sha = digest(archive)
    assert archive_sha == execution["artifact_sha256"], f"{slot}: archive SHA mismatch"
    assert execution["stage"] == "pod_termination_verified" and lease["state"] == "terminated"
    assert execution["freeze_sha256"] == freeze["sha256"]

    summary_by_stage: dict[tuple[str, int], dict] = {}
    requests_by_stage: dict[tuple[str, int], list[dict]] = {}
    metrics = Counter()
    manifest = None
    embedded_freeze = None
    environment = None
    gpu_name = None

    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if name == "output/manifest.json":
                manifest = json.load(tar.extractfile(member))
            elif name == "output/freeze.json":
                embedded_freeze = json.load(tar.extractfile(member))
            elif name == "output/environment.json":
                environment = json.load(tar.extractfile(member))
            elif name == "output/gpu-identity.json":
                gpu = json.load(tar.extractfile(member))
                assert gpu["returncode"] == 0
                gpu_name = gpu["stdout"].split(",", 1)[0]
            elif name.endswith("/metrics-before.txt"):
                metrics["before"] += 1
            elif name.endswith("/metrics-after.txt"):
                metrics["after"] += 1
            else:
                match = STAGE.fullmatch(name)
                if not match:
                    continue
                stage_key = (match.group(1), int(match.group(2)))
                if match.group(3) == "summary.json":
                    stages = json.load(tar.extractfile(member))["stages"]
                    assert len(stages) == 1
                    assert stage_key not in summary_by_stage
                    summary_by_stage[stage_key] = stages[0]
                else:
                    assert stage_key not in requests_by_stage
                    requests_by_stage[stage_key] = [json.loads(line) for line in tar.extractfile(member)]

    expected_cells = {cell["id"] for cell in freeze["configuration"]["cells_per_hardware"]}
    expected_stages = {(cell, concurrency) for cell in expected_cells for concurrency in CONCURRENCY}
    assert manifest and manifest["status"] == "complete" and set(manifest["cells"]) == expected_cells
    assert embedded_freeze == freeze and environment and environment["hardware"] == slot
    assert environment["packages"]["vllm"] == freeze["configuration"]["vllm_version"]
    assert set(summary_by_stage) == set(requests_by_stage) == expected_stages
    assert metrics == {"before": 128, "after": 128}
    assert gpu_name == freeze["configuration"]["hardware"][slot]

    records = []
    hashes = {}
    for (cell, concurrency), summary in sorted(summary_by_stage.items()):
        match = CELL.fullmatch(cell)
        assert match
        repeat, model, prefix, cache = match.groups()
        requests = requests_by_stage[(cell, concurrency)]
        assert len(requests) == summary["requests"] == 64
        assert sum(bool(x["ok"]) for x in requests) == summary["successful"]
        assert sum(not x["ok"] for x in requests) == summary["failed"]
        assert summary["successful"] + summary["failed"] == 64
        assert all(x["concurrency"] == concurrency and not x["warmup"] for x in requests)
        row = {
            "hardware": slot,
            "repeat": int(repeat),
            "model": model,
            "prefix": prefix,
            "cache": int(cache),
            "concurrency": concurrency,
            "requests": 64,
            "successful": summary["successful"],
            "failed": summary["failed"],
            "completion_tokens_per_s": summary["completion_tokens_per_s"],
            "latency_p95_s": summary["latency_success_s"]["p95"],
            "ttft_p95_s": summary["ttft_s"]["p95"],
        }
        records.append(row)
        for request in requests:
            key = (int(repeat), model, prefix, int(cache), concurrency, request["workload_id"])
            assert key not in hashes
            hashes[key] = request["content_sha256"]

    assert len(records) == 128 and len(hashes) == 8192
    info = {
        "hardware": slot,
        "gpu_name": gpu_name,
        "archive_sha256": archive_sha,
        "freeze_sha256": freeze["sha256"],
        "cells": len(manifest["cells"]),
        "stages": len(records),
        "requests": sum(x["requests"] for x in records),
        "successful": sum(x["successful"] for x in records),
        "failed": sum(x["failed"] for x in records),
        "allocation_rate_estimate_usd": lease["allocation_rate_estimate_usd"],
        "pod_termination_verified_utc": lease["termination_verified_utc"],
        "vllm_version": environment["packages"]["vllm"],
    }
    return records, hashes, info


def hash_comparisons(by_slot: dict[str, dict]) -> dict:
    result = {"within_hardware_cache": {}, "across_hardware": {}}
    for slot, hashes in by_slot.items():
        counts = defaultdict(lambda: [0, 0])
        for (repeat, model, prefix, cache, concurrency, workload), value in hashes.items():
            if cache:
                continue
            other = hashes.get((repeat, model, prefix, 1, concurrency, workload))
            assert other is not None
            counts[model][0] += 1
            counts[model][1] += int(value == other)
        result["within_hardware_cache"][slot] = {
            model: {"paired": total, "hash_equal": equal, "hash_mismatch": total - equal}
            for model, (total, equal) in sorted(counts.items())
        }
    a, b = (by_slot[slot] for slot in SLOTS)
    assert set(a) == set(b)
    counts = defaultdict(lambda: [0, 0])
    for key, value in a.items():
        counts[key[1]][0] += 1
        counts[key[1]][1] += int(value == b[key])
    result["across_hardware"] = {
        model: {"paired": total, "hash_equal": equal, "hash_mismatch": total - equal}
        for model, (total, equal) in sorted(counts.items())
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    freeze = json.loads((args.evidence_dir / "factorial-serving-v1-freeze.json").read_text())
    complete = json.loads((args.evidence_dir / "campaign-complete.json").read_text())
    assert complete["pods_remaining"] == []

    records = []
    hashes = {}
    info = []
    for slot in SLOTS:
        these, hashes[slot], data = read_slot(slot, args.evidence_dir, freeze)
        records.extend(these)
        info.append(data)
    assert len(records) == 256
    assert sum(x["successful"] for x in records) == 16384
    assert sum(x["failed"] for x in records) == 0

    grouped = defaultdict(list)
    for row in records:
        grouped[(row["hardware"], row["model"], row["prefix"], row["cache"], row["concurrency"])].append(row)
    aggregates = []
    for key, rows in sorted(grouped.items()):
        assert len(rows) == 4 and {x["repeat"] for x in rows} == {1, 2, 3, 4}
        hardware, model, prefix, cache, concurrency = key
        aggregates.append({
            "hardware": hardware,
            "model": model,
            "prefix": prefix,
            "cache": cache,
            "concurrency": concurrency,
            "repeats": 4,
            "requests": sum(x["requests"] for x in rows),
            "successful": sum(x["successful"] for x in rows),
            "failed": sum(x["failed"] for x in rows),
            "mean_completion_tokens_per_s": statistics.mean(x["completion_tokens_per_s"] for x in rows),
            "min_completion_tokens_per_s": min(x["completion_tokens_per_s"] for x in rows),
            "max_completion_tokens_per_s": max(x["completion_tokens_per_s"] for x in rows),
            "mean_latency_p95_s": statistics.mean(x["latency_p95_s"] for x in rows),
            "mean_ttft_p95_s": statistics.mean(x["ttft_p95_s"] for x in rows),
        })
    assert len(aggregates) == 64

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("cells.csv", records), ("aggregates.csv", aggregates)):
        with (args.output_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    comparisons = hash_comparisons(hashes)
    (args.output_dir / "comparisons.json").write_text(json.dumps(comparisons, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema": "factorial-serving-reviewed-v1",
        "freeze_sha256": freeze["sha256"],
        "source_commit": freeze["source_commit"],
        "total_measured_requests": 16384,
        "total_http_successful": 16384,
        "total_http_failed": 0,
        "hardware": info,
        "runpod_allocation_rate_estimate_usd": sum(x["allocation_rate_estimate_usd"] for x in info),
        "runpod_actual_billed_usd": None,
        "cost_note": "Lease lifetime times quoted hourly rates. Runpod invoice/billing record not available to this scoped API key.",
        "comparisons": comparisons,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stages": len(records), "aggregate_rows": len(aggregates), "success": 16384, "failed": 0, "estimated_cost_usd": summary["runpod_allocation_rate_estimate_usd"]}, sort_keys=True))


if __name__ == "__main__":
    main()
