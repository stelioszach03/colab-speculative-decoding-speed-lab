"""Frozen, bounded single-GPU pilot. Plan-only unless --execute is supplied.

Runs INSIDE the pinned vLLM container. This never provisions paid hardware and
cannot stop cloud billing; the operator must set an independent teardown limit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
IMAGE = "vllm/vllm-openai@sha256:df2607b26bdda2875de4832f4d08da0055b4b6e3570347f3a849bcc652771dd6"
SCHEMA = "inference-lab-controlled-pilot-v1"
GPU_FIELDS = "timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu"
SENTENCES = (
    "The city library lends books and records their return dates.",
    "Each route connects stations and records a timetable for travelers.",
    "A team checks software changes before releasing the next version.",
    "The garden stores rainfall readings and schedules watering carefully.",
    "An archive keeps documents with dates authors and short descriptions.",
    "The workshop tracks inventory deliveries and equipment maintenance.",
    "A calendar lists meetings and identifies rooms for each event.",
    "The museum labels exhibits and records their conservation history.",
)


def workload():
    """64 original synthetic prompts: four lengths × two prefix strata × eight."""
    rows, strata = [], []
    words = " ".join(SENTENCES).split()
    for length in (64, 192, 448, 896):
        context = " ".join((words * (length // len(words) + 1))[:length])
        for kind in ("shared", "unique"):
            for index in range(8):
                identity = f"{kind}-w{length}-{index:02d}"
                prefix = "Synthetic reference notes.\n" if kind == "shared" else f"Independent document {identity}.\n"
                prompt = (prefix + context + "\n\n"
                          + f"Task {index}: write a concise structured operational summary of these notes. "
                          + "Mention concrete processes and possible checks. Use only the supplied notes.")
                rows.append({"id": identity, "prompt": prompt})
                strata.append({"id": identity, "context_words": length, "prefix_stratum": kind})
    return rows, strata


def workload_bytes():
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in workload()[0]).encode()


def server_command(cache):
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL, "--revision", REVISION, "--tokenizer-revision", REVISION,
            "--host", "127.0.0.1", "--port", "8000", "--dtype", "bfloat16",
            "--max-model-len", "4096", "--max-num-seqs", "32",
            "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.8",
            "--tensor-parallel-size", "1", "--seed", "17", "--generation-config", "vllm",
            "--disable-log-requests", "--enable-prefix-caching" if cache else "--no-enable-prefix-caching"]


def protocol():
    # Alternate paired modes; fixed in advance and not changed after measurements.
    cells = []
    orders = ([1, 4, 16, 32], [32, 16, 4, 1], [4, 32, 1, 16])
    for replicate, modes in enumerate(((False, True), (True, False), (False, True)), 1):
        for cache in modes:
            cells.append({"id": f"rep{replicate}-cache-{'on' if cache else 'off'}",
                          "replicate": replicate, "prefix_cache": cache,
                          "concurrency": orders[replicate - 1], "seed": 17 + replicate})
    return {"schema": SCHEMA, "model": MODEL, "revision": REVISION, "license": "Apache-2.0",
            "image": IMAGE, "vllm_version": "0.10.2", "cells": cells,
            "requests_per_stage": 64, "warmup_per_stage": 4, "max_tokens": 128,
            "fixed_output_tokens": True, "temperature": 0,
            "total_measured_requests": 1536, "total_requests_with_warmup": 1632,
            "maximum_requested_output_tokens": 208896, "request_timeout_s": 60,
            "experiment_deadline_s": 2700, "server_readiness_timeout_s": 480,
            "workload_sha256": hashlib.sha256(workload_bytes()).hexdigest(),
            "cache_policy": "Fresh server per cell; prefix cache persists between four warmed stages within a cell.",
            "telemetry_scope": "nvidia-smi device-wide measurements on the one visible GPU; not per-process utilization.",
            "gpu_sample_interval_ms": 500, "strata": workload()[1],
            "limitations": ["Single GPU and one model; descriptive pilot, no population-level confidence claim.",
                            "Fixed synthetic output length; generated text quality is not evaluated.",
                            "Warm repeated workload favors reuse; not representative production traffic.",
                            "Prefix cache effects may include runtime variation; matching output hashes are reported separately.",
                            "No quantization or speculative-decoding treatment in this pilot."]}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def capture(command, path):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    write_json(path, {"command": command, "returncode": result.returncode,
                      "stdout": result.stdout, "stderr": result.stderr})
    return result


def stop_owned(process):
    """Only signal the process group created by this runner."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    # A server parent can exit while engine children remain in its own group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def local_get(path):
    # Disable proxies: requests must remain on loopback, including health checks.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:8000" + path, timeout=3) as response:
        return response.read(2_000_000)


def wait_ready(server, deadline):
    until = min(deadline, time.monotonic() + 480)
    while time.monotonic() < until:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited before readiness (code {server.returncode})")
        try:
            models = json.loads(local_get("/v1/models"))
            if any(item.get("id") == MODEL for item in models.get("data", [])):
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise TimeoutError("server readiness deadline exceeded")


def run(output):
    plan = protocol()
    frozen = json.loads((ROOT / 'protocols/controlled-pilot-v1.json').read_text())
    if frozen != plan:
        raise RuntimeError('Runner plan differs from committed frozen protocol')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    (output / "workload.jsonl").write_bytes(workload_bytes())
    versions = {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers", "huggingface-hub")}
    write_json(output / "environment.json", {"packages": versions, "python": platform.python_version(),
                                             "platform": platform.platform(),
                                             "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                             "client_sha256": hashlib.sha256((ROOT / "inference_lab/benchmark.py").read_bytes()).hexdigest()})
    if versions["vllm"] != "0.10.2":
        raise RuntimeError("This protocol requires exactly vLLM 0.10.2")
    devices = capture(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader"], output / "gpu-identity.json")
    if devices.returncode or len(devices.stdout.strip().splitlines()) != 1:
        raise RuntimeError("Pilot requires exactly one visible NVIDIA GPU")
    capture(["nvidia-smi", "-q"], output / "gpu-environment.json")
    # This file detects competing processes where the provider exposes them;
    # container/PID namespace restrictions can prevent complete attribution.
    capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], output / "gpu-processes-before.json")
    try:
        local_get("/health")
    except OSError:
        pass
    else:
        raise RuntimeError("Port 8000 already serves HTTP; refusing to share an unknown server")
    manifest = {"schema": SCHEMA, "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "running", "cells": []}
    write_json(output / "manifest.json", manifest)
    deadline = time.monotonic() + plan["experiment_deadline_s"]
    telemetry = server = client = None
    try:
        for cell in plan["cells"]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("experiment deadline exceeded")
            cell_dir = output / cell["id"]
            cell_dir.mkdir()
            command = server_command(cell["prefix_cache"])
            metadata = {**cell, "model": MODEL, "revision": REVISION, "image_expected": IMAGE,
                        "launch_command": command, "packages_measured": versions,
                        "gpu_identity_measured": devices.stdout,
                        "telemetry_file": "gpu.csv", "telemetry_scope": plan["telemetry_scope"]}
            write_json(cell_dir / "server-settings.json", metadata)
            with (cell_dir / "server.log").open("w") as server_log, (cell_dir / "gpu.csv").open("w") as gpu_csv, (cell_dir / "gpu-telemetry.stderr").open("w") as gpu_error:
                gpu_csv.write(GPU_FIELDS + '\n')
                gpu_csv.flush()
                telemetry = subprocess.Popen(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits", "-lms", "500"], stdout=gpu_csv, stderr=gpu_error, env={**os.environ, "TZ": "UTC"}, start_new_session=True)
                environment = {**os.environ, "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
                server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
                wait_ready(server, deadline)
                if telemetry.poll() is not None:
                    raise RuntimeError("GPU telemetry exited; refusing uninstrumented benchmark")
                (cell_dir / "metrics-before.txt").write_bytes(local_get("/metrics"))
                capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], cell_dir / "gpu-processes-serving.json")
                client_command = [sys.executable, "-m", "inference_lab.benchmark", "--model", MODEL,
                                  "--workload", str(output / "workload.jsonl"),
                                  "--concurrency", *map(str, cell["concurrency"]), "--requests", "64",
                                  "--warmup", "4", "--max-tokens", "128", "--fixed-output-tokens",
                                  "--timeout", "60", "--seed", str(cell["seed"]),
                                  "--server-metadata", str(cell_dir / "server-settings.json"),
                                  "--output", str(cell_dir / "client"), "--execute"]
                with (cell_dir / "client.log").open("w") as client_log:
                    client = subprocess.Popen(client_command, cwd=ROOT, stdout=client_log, stderr=subprocess.STDOUT, start_new_session=True)
                    code = client.wait(timeout=max(1, deadline - time.monotonic()))
                if code not in (0, 1) or not (cell_dir / "client/summary.json").exists():
                    raise RuntimeError(f"Benchmark client failed before complete records (code {code})")
                (cell_dir / "metrics-after.txt").write_bytes(local_get("/metrics"))
                stop_owned(server)
                server = None
                stop_owned(telemetry)
                telemetry = None
                manifest["cells"].append({**cell, "client_exit_code": code})
                write_json(output / "manifest.json", manifest)
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update({"status": "interrupted", "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        stop_owned(client)
        stop_owned(server)
        stop_owned(telemetry)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/controlled-pilot-v1"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(protocol(), indent=2))
        print("PLAN ONLY. No hardware allocated, endpoint contacted or files written.")
        return 0
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("termination requested")
    signal.signal(signal.SIGTERM, interrupted)
    run(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
