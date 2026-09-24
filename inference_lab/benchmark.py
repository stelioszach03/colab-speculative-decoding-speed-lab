"""Bounded, closed-loop SSE benchmark for a locally served chat model.

No endpoint is contacted unless --execute is supplied. GPU utilization, VRAM,
cost and model quality are not inferred from HTTP timings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import time
from urllib.parse import urlsplit

SCHEMA = "inference-lab-serving-v0.1"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def endpoint(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") not in {"", "/v1"}):
        raise ValueError("Use a loopback http(s) base URL, optionally ending in /v1, without credentials.")
    # Parse the port here so invalid values fail before any output or request.
    _ = parsed.port
    return parsed


def load_workload(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or set(row) != {"id", "prompt"} or not isinstance(row["id"], str)
                or not row["id"] or not isinstance(row["prompt"], str)
                or not 1 <= len(row["prompt"]) <= 100_000):
            raise ValueError("Each workload row needs a nonempty string id and a 1–100000-character prompt.")
        rows.append(row)
    if not rows or len(rows) > 10_000 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Workload must have 1–10000 uniquely identified records.")
    return rows


class StreamRecord:
    """Consume complete SSE data payloads; role/usage-only events are not tokens."""

    def __init__(self):
        self.events = []
        self.usage = None
        self.finish_reason = None
        self.actual_model = None
        self.done = False
        self.non_text_output_seen = False
        self.content_hash = hashlib.sha256()

    def consume(self, payload, elapsed):
        if payload.strip() == "[DONE]":
            self.done = True
            return
        data = json.loads(payload)
        if not isinstance(data, dict) or data.get("error"):
            raise ValueError("invalid or error stream")
        if data.get("model"):
            self.actual_model = data["model"]
        if data.get("usage") is not None:
            usage = data["usage"]
            if not isinstance(usage, dict):
                raise ValueError("invalid usage")
            for name in ("completion_tokens", "prompt_tokens"):
                number = usage.get(name)
                if number is not None and (type(number) is not int or number < 0):
                    raise ValueError("invalid token count")
            self.usage = {name: usage.get(name) for name in ("prompt_tokens", "completion_tokens")}
            details = usage.get("completion_tokens_details") or {}
            if isinstance(details, dict) and details.get("reasoning_tokens", 0):
                self.non_text_output_seen = True
        choices = data.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("expected a single streamed choice")
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                raise ValueError("invalid choice")
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise ValueError("invalid delta")
            if any(delta.get(name) for name in ("reasoning", "reasoning_content", "tool_calls", "function_call")):
                self.non_text_output_seen = True
            # Unsupported reasoning/tool-only output cannot masquerade as text timing.
            content = delta.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("non-text delta")
            if content:
                encoded = content.encode("utf-8")
                self.content_hash.update(encoded)
                self.events.append({"elapsed_s": elapsed, "utf8_bytes": len(encoded)})
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]

    def metrics(self):
        times = [event["elapsed_s"] for event in self.events]
        count = self.usage.get("completion_tokens") if self.usage else None
        return {
            "ttft_s": times[0] if times else None,
            "itl_stream_s": [b - a for a, b in zip(times, times[1:])],
            "tpot_s": ((times[-1] - times[0]) / (count - 1)
                       if len(times) >= 2 and count is not None and count > 1
                       and not self.non_text_output_seen else None),
            "completion_tokens": count,
            "prompt_tokens": self.usage.get("prompt_tokens") if self.usage else None,
            "events": self.events,
            "content_sha256": self.content_hash.hexdigest(),
            "finish_reason": self.finish_reason,
            "actual_model": self.actual_model,
            "stream_done": self.done,
            "non_text_output_seen": self.non_text_output_seen,
        }


def stream_request(base_url, model, prompt, max_tokens, timeout, api_key=None):
    target = endpoint(base_url)
    connection_type = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(target.hostname, target.port, timeout=timeout)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True}, "n": 1}
    start = time.perf_counter()
    record = StreamRecord()
    status, error, response = None, None, None
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(payload), headers)
        response = connection.getresponse()
        status = response.status
        if status != 200:
            error = "http_error"
        elif "text/event-stream" not in response.getheader("Content-Type", "").lower():
            error = "unexpected_content_type"
        else:
            buffer, fields, received = b"", [], 0
            while not record.done:
                if time.perf_counter() - start >= timeout:
                    raise TimeoutError()
                # read1 does not wait for a full buffer, preserving streaming timing.
                chunk = response.read1(65536)
                if not chunk:
                    break
                received += len(chunk)
                if received > MAX_RESPONSE_BYTES:
                    error = "response_size_limit"
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.rstrip(b"\r")
                    if line.startswith(b"data:"):
                        fields.append(line[5:].lstrip(b" "))
                    elif not line and fields:
                        record.consume(b"\n".join(fields).decode("utf-8"), time.perf_counter() - start)
                        fields = []
                        if record.done:
                            break
            if error is None and (not record.done or record.finish_reason is None):
                error = "incomplete_stream"
            if error is None and not record.events:
                error = "no_text_output"
    except (TimeoutError, socket.timeout):
        error = "timeout"
    except (ValueError, TypeError, KeyError, UnicodeError):
        error = "invalid_stream"
    except (OSError, http.client.HTTPException):
        error = "transport_error"
    finally:
        if response is not None:
            response.close()
        connection.close()
    return {"ok": error is None, "error": error, "http_status": status,
            "latency_s": time.perf_counter() - start, **record.metrics()}


def percentiles(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    def quantile(q):
        index = (len(values) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)
    return {"count": len(values), "mean": sum(values) / len(values),
            **{key: quantile(q) for key, q in (("p50", .50), ("p95", .95), ("p99", .99))}}


def summarize(rows, elapsed, concurrency):
    successful = [row for row in rows if row["ok"]]
    counts = [row["completion_tokens"] for row in successful]
    complete_usage = bool(successful) and all(value is not None for value in counts)
    return {
        "concurrency": concurrency, "requests": len(rows), "successful": len(successful),
        "failed": len(rows) - len(successful), "elapsed_s": elapsed,
        "successful_requests_per_s": len(successful) / elapsed if elapsed else None,
        "completion_tokens": sum(counts) if complete_usage else None,
        "completion_tokens_per_s": sum(counts) / elapsed if complete_usage and elapsed else None,
        "latency_all_s": percentiles([row["latency_s"] for row in rows]),
        "latency_success_s": percentiles([row["latency_s"] for row in successful]),
        "ttft_s": percentiles([row["ttft_s"] for row in successful if row["ttft_s"] is not None]),
        "itl_stream_s": percentiles([gap for row in successful for gap in row["itl_stream_s"]]),
        "tpot_s": percentiles([row["tpot_s"] for row in successful if row["tpot_s"] is not None]),
        "failures_by_kind": {kind: sum(row["error"] == kind for row in rows)
                             for kind in sorted({row["error"] for row in rows if row["error"]})},
    }


def run_suite(args, workload, plan):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {**plan, "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                "client": {"python": platform.python_version(), "platform": platform.platform()}}
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "workload.jsonl").write_text("".join(json.dumps(row) + "\n" for row in workload))
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    summaries = []
    try:
        with (args.output / "requests.jsonl").open("w") as raw:
            def request(index, concurrency, warmup=False):
                item = workload[index % len(workload)]
                return {"request_id": f"c{concurrency}-{'warmup' if warmup else 'measured'}-{index}",
                        "workload_id": item["id"], "concurrency": concurrency, "warmup": warmup,
                        **stream_request(args.base_url, args.model, item["prompt"],
                                         args.max_tokens, args.timeout, api_key)}
            for concurrency in args.concurrency:
                for index in range(args.warmup):
                    row = request(index, concurrency, True)
                    raw.write(json.dumps(row) + "\n")
                    raw.flush()
                start = time.perf_counter()
                rows = []
                # Fixed concurrency, closed loop: next queued request starts when a worker frees.
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(request, index, concurrency) for index in range(args.requests)]
                    for future in as_completed(futures):
                        row = future.result()
                        rows.append(row)
                        raw.write(json.dumps(row) + "\n")
                        raw.flush()
                summaries.append(summarize(rows, time.perf_counter() - start, concurrency))
                (args.output / "summary.json").write_text(json.dumps({"schema": SCHEMA, "stages": summaries}, indent=2) + "\n")
        manifest["status"] = "completed"
    finally:
        if manifest["status"] != "completed":
            manifest["status"] = "interrupted"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="Actual served model ID, not a display label")
    parser.add_argument("--workload", type=Path, default=Path("workloads/synthetic.jsonl"))
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--requests", type=int, default=16, help="Measured requests per concurrency stage")
    parser.add_argument("--warmup", type=int, default=1, help="Serial warmups per stage, logged and excluded")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=30, help="Socket inactivity timeout and checked request deadline")
    parser.add_argument("--seed", type=int, default=17, help="Workload-order seed, not server determinism")
    parser.add_argument("--api-key-env", help="Optional environment variable; value never written to artifacts")
    parser.add_argument("--server-metadata", type=Path, help="JSON object of user-reported server settings, no secrets")
    parser.add_argument("--output", type=Path, default=Path("results/local-serving"))
    parser.add_argument("--execute", action="store_true", help="Actually contact the loopback model server")
    args = parser.parse_args(argv)
    try:
        endpoint(args.base_url)
        if (not 1 <= args.requests <= 10_000 or not 0 <= args.warmup <= 100
                or not 1 <= args.max_tokens <= 4096 or not 0 < args.timeout <= 300
                or not 1 <= len(args.concurrency) <= 5
                or len(set(args.concurrency)) != len(args.concurrency)
                or any(not 1 <= number <= min(64, args.requests) for number in args.concurrency)):
            raise ValueError("Invalid bounded request, warmup, token, timeout or concurrency setting.")
        workload = load_workload(args.workload)
        random.Random(args.seed).shuffle(workload)
        server_metadata = json.loads(args.server_metadata.read_text()) if args.server_metadata else {}
        if not isinstance(server_metadata, dict):
            raise ValueError("Server metadata must be a JSON object.")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    plan = {"schema": SCHEMA, "base_url": args.base_url, "requested_model": args.model,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "concurrency": args.concurrency, "requests_per_stage": args.requests,
            "warmup_per_stage": args.warmup, "max_tokens": args.max_tokens,
            "temperature": 0, "request_timeout_s": args.timeout, "workload_order_seed": args.seed,
            "total_requests_including_warmup": len(args.concurrency) * (args.requests + args.warmup),
            "maximum_requested_output_tokens": len(args.concurrency) * (args.requests + args.warmup) * args.max_tokens,
            "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
            "server_metadata_user_reported": server_metadata,
            "load_pattern": "closed_loop_fixed_concurrency",
            "timing_semantics": "Client-observed first text chunk and inter-chunk gaps; chunks may contain multiple tokens.",
            "gpu_telemetry": None, "cost": None, "output_quality_evaluated": False}
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("PLAN ONLY: no requests made and no results written. Add --execute to run.")
        return 0
    summaries = run_suite(args, workload, plan)
    print(f"Recorded {sum(stage['requests'] for stage in summaries)} measured requests in {args.output}")
    return 1 if any(stage["failed"] for stage in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
