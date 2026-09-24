"""Aggregate a real pilot artifact directory without overwriting raw evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from inference_lab.benchmark import percentiles, summarize


def gpu_rows(path):
    rows = []
    with path.open() as stream:
        for record in csv.DictReader(stream, skipinitialspace=True):
            values = {key.strip(): value.strip() for key, value in record.items() if key and value}
            try:
                moment = datetime.strptime(values['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
                def number(name):
                    value = next((v for k, v in values.items() if k.startswith(name)), None)
                    numeric = float(value.split()[0]) if value and value not in ('[N/A]', 'N/A', '[Not Supported]') else None
                    if numeric is not None and (not math.isfinite(numeric) or numeric < 0):
                        raise ValueError('GPU sample must be finite and nonnegative')
                    if name == 'utilization.gpu' and numeric is not None and numeric > 100:
                        raise ValueError('GPU utilization exceeds 100 percent')
                    return numeric
                rows.append({'time': moment, 'memory_mib': number('memory.used'),
                             'utilization_percent': number('utilization.gpu'), 'power_w': number('power.draw')})
            except (ValueError, KeyError):
                # A nonparseable row must be surfaced, never silently discarded.
                raise ValueError(f'Invalid GPU telemetry record in {path.name}')
    return rows


def check_summary(stage, measured):
    """Reject corrupted/inconsistent derived metrics; preserve genuine failures."""
    if len({row['request_id'] for row in measured}) != len(measured):
        raise ValueError('Duplicate measured request IDs')
    if len({row['workload_id'] for row in measured}) != len(measured):
        raise ValueError('Duplicate workload IDs within a 64-prompt pilot stage')
    expected = summarize(measured, stage['elapsed_s'], stage['concurrency'])
    def same(left, right):
        if type(left) is float or type(right) is float:
            return isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
        return left == right
    for key, value in expected.items():
        if not same(stage.get(key), value):
            raise ValueError(f'Summary field {key} does not match raw request evidence')


def aggregate(root):
    plan = json.loads((root / 'protocol.json').read_text())
    frozen = json.loads((Path(__file__).resolve().parents[1] / 'protocols/controlled-pilot-v1.json').read_text())
    if plan != frozen:
        raise ValueError('Artifact protocol differs from the frozen controlled pilot v1 plan')
    manifest = json.loads((root / 'manifest.json').read_text())
    result = {'schema': plan['schema'] + '-report', 'source_status': manifest['status'],
              'limitations': plan['limitations'], 'stages': [], 'output_agreement': [],
              'incomplete_cells': [], 'cost_usd': None, 'integrity_issues': []}
    pairs = {}
    for cell in plan['cells']:
        directory = root / cell['id']
        summary_path = directory / 'client/summary.json'
        if not summary_path.exists():
            result['incomplete_cells'].append(cell['id'])
            continue
        records = [json.loads(line) for line in (directory / 'client/requests.jsonl').read_text().splitlines()]
        readings = gpu_rows(directory / 'gpu.csv')
        summary = json.loads(summary_path.read_text())
        if len({stage['concurrency'] for stage in summary['stages']}) != len(summary['stages']):
            raise ValueError('Duplicate stages in a cell summary')
        observed_order = [stage['concurrency'] for stage in summary['stages']]
        if observed_order != cell['concurrency'][:len(observed_order)]:
            result['integrity_issues'].append(f"{cell['id']}: stage order differs from frozen protocol")
        for stage in summary['stages']:
            start, end = (datetime.fromisoformat(stage[key]) for key in ('started_at', 'finished_at'))
            samples = [row for row in readings if start <= row['time'] <= end]
            memory = [row['memory_mib'] for row in samples if row['memory_mib'] is not None]
            use = [row['utilization_percent'] for row in samples if row['utilization_percent'] is not None]
            power = [row['power_w'] for row in samples if row['power_w'] is not None]
            concurrent = stage['concurrency']
            measured = [row for row in records if not row['warmup'] and row['concurrency'] == concurrent]
            check_summary(stage, measured)
            if len(measured) != plan['requests_per_stage']:
                result['integrity_issues'].append(f"{cell['id']} c{concurrent}: measured request count differs from frozen protocol")
            if any(row.get('actual_model') != plan['model'] for row in measured if row['ok']):
                result['integrity_issues'].append(f"{cell['id']} c{concurrent}: successful response model differs from protocol")
            if not samples:
                result['integrity_issues'].append(f"{cell['id']} c{concurrent}: no GPU sample in measured interval")
            fixed_violations = sum(row['ok'] and row.get('completion_tokens') != plan['max_tokens'] for row in measured)
            result['stages'].append({'cell': cell['id'], 'replicate': cell['replicate'],
                                     'prefix_cache': cell['prefix_cache'], **stage,
                                     'fixed_output_length_mismatches': fixed_violations,
                                     'gpu_sample_count': len(samples), 'gpu_telemetry_scope': plan['telemetry_scope'],
                                     'gpu_memory_peak_mib': max(memory) if memory else None,
                                     'gpu_memory_mean_mib': sum(memory) / len(memory) if memory else None,
                                     'gpu_utilization_percent': percentiles(use), 'gpu_power_w': percentiles(power)})
            pairs[(cell['replicate'], cell['prefix_cache'], concurrent)] = {row['workload_id']: row for row in measured}
        if len(summary['stages']) != len(cell['concurrency']):
            result['incomplete_cells'].append(cell['id'])
    for replicate in (1, 2, 3):
        for concurrency in (1, 4, 16, 32):
            off, on = pairs.get((replicate, False, concurrency), {}), pairs.get((replicate, True, concurrency), {})
            if not off or not on:
                continue
            comparable = [key for key in off.keys() & on.keys() if off[key]['ok'] and on[key]['ok']]
            matches = sum(off[key]['content_sha256'] == on[key]['content_sha256'] for key in comparable)
            result['output_agreement'].append({'replicate': replicate, 'concurrency': concurrency,
                                                'paired_successful_requests': len(comparable), 'exact_text_hash_matches': matches,
                                                'not_a_quality_score': True})
    result['source_file_sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(root.rglob('*')) if path.is_file()
                                    and path.name not in {'report.json', 'stages.csv'}
                                    and 'figures' not in path.relative_to(root).parts}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    result = aggregate(args.directory)
    destination = args.directory / 'report.json'
    # Analysis can be rerun explicitly; raw records are never modified.
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    fields = ['cell', 'replicate', 'prefix_cache', 'concurrency', 'requests', 'successful', 'failed',
              'elapsed_s', 'completion_tokens_per_s', 'ttft_mean_s', 'ttft_p95_s', 'latency_p95_s',
              'gpu_sample_count', 'gpu_memory_peak_mib', 'gpu_utilization_mean_percent', 'fixed_output_length_mismatches']
    with (args.directory / 'stages.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for stage in result['stages']:
            row = {key: stage[key] for key in fields if key in stage}
            row.update(ttft_mean_s=stage['ttft_s']['mean'], ttft_p95_s=stage['ttft_s']['p95'],
                       latency_p95_s=stage['latency_all_s']['p95'],
                       gpu_utilization_mean_percent=stage['gpu_utilization_percent']['mean'])
            writer.writerow(row)
    print(f"Aggregated {len(result['stages'])} stages; {len(result['incomplete_cells'])} incomplete cells; {destination}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
