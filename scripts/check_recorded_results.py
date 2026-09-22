"""Validate recorded latency tables against extracted summaries without inference."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def close(actual, expected, label):
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=2e-6):
        raise ValueError(f'{label}: recorded {actual}, expected {expected}')


def rows(name):
    with (ROOT / 'paper' / 'data' / name).open() as file:
        return list(csv.DictReader(file))


def main():
    summaries = json.loads((ROOT / 'extracted_metrics.json').read_text())['summaries']
    checks = 0
    for row in rows('phase2_results.csv'):
        for split in ['short', 'long', 'qa']:
            measured = summaries[f'{row["config"]}_{split}']['mean_latency_s']
            baseline = summaries[f'baseline_{split}']['mean_latency_s']
            close(row[f'speedup_{split}'], baseline / measured, f'{row["config"]}/{split}')
            checks += 1
        for split in ['short', 'long']:
            close(row[f'latency_{split}_s'], summaries[f'{row["config"]}_{split}']['mean_latency_s'], 'latency')
            checks += 1
    categories = rows('phase3_latency_by_category.csv')
    for row in categories:
        close(row['speedup'], float(row['baseline_latency_s']) / float(row['best_spec_latency_s']), row['category'])
        checks += 1
    baseline_mean = sum(float(row['baseline_latency_s']) for row in categories) / len(categories)
    # Both categories have twelve prompts in the recorded 24-prompt study.
    for row in rows('phase3_results.csv'):
        summary = summaries[f'phase3_{row["config"]}']
        if summary['n'] != 24:
            raise ValueError('Category-weighting assumption requires the recorded 24 prompts')
        close(row['latency_mean_s'], summary['mean_latency_s'], 'phase3 latency')
        close(row['global_tps'], summary['total_tokens'] / summary['total_time_s'], 'phase3 tokens/s')
        close(row['speedup'], baseline_mean / summary['mean_latency_s'], 'phase3 aggregate speedup')
        checks += 3
    print(f'PASS: {checks} recorded latency/throughput consistency checks; no model execution.')
    print('Saved measurements are not independent replications or output-equivalence evidence.')


if __name__ == '__main__':
    main()
