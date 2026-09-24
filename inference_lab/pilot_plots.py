"""Scientific plots from validated measured pilot artifacts; no fixture data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inference_lab.pilot_report import aggregate


def plot(root, output, allow_incomplete=False):
    # Rebuild and validate measurements from raw records, not a hand-edited table.
    result = aggregate(root)
    if not result['stages']:
        raise ValueError('No measured stages exist; refusing to produce a results figure')
    incomplete = result['source_status'] != 'completed' or bool(result['incomplete_cells'])
    if incomplete and not allow_incomplete:
        raise ValueError('Incomplete experiment: use --allow-incomplete to produce explicitly partial figures')
    if result['integrity_issues']:
        raise ValueError('Artifact integrity issues must be reviewed before plotting: ' + '; '.join(result['integrity_issues']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#18283b', 'text.color': '#18283b',
                         'axes.titleweight': 'bold', 'svg.fonttype': 'none'})
    plan = json.loads((root / 'protocol.json').read_text())
    identity = json.loads((root / 'gpu-identity.json').read_text())
    gpu_line = identity.get('stdout', '').strip().splitlines()
    # Device report is a short nvidia-smi CSV; preserve actual reported GPU name.
    import csv
    gpu_name = next(csv.reader(gpu_line), ['', '', 'Unreported GPU'])[2].strip()
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 8.8), layout='constrained')
    colors = {False: '#263b62', True: '#007f78'}
    markers = {1: 'o', 2: 's', 3: '^'}
    panels = [
        ('Decode throughput', 'Completion tokens / second', lambda row: row['completion_tokens_per_s']),
        ('Time to first text · p95', 'Milliseconds', lambda row: None if row['ttft_s']['p95'] is None else row['ttft_s']['p95'] * 1000),
        ('Request latency · p95', 'Seconds, including failed requests', lambda row: row['latency_all_s']['p95']),
        ('Device utilization', 'Mean sampled GPU utilization (%)', lambda row: row['gpu_utilization_percent']['mean']),
        ('Device memory', 'Peak sampled VRAM (GiB)', lambda row: None if row['gpu_memory_peak_mib'] is None else row['gpu_memory_peak_mib'] / 1024),
    ]
    for axis, (title, label, value) in zip(axes.flat, panels):
        for cache in (False, True):
            for replicate in (1, 2, 3):
                rows = sorted((row for row in result['stages'] if row['prefix_cache'] == cache and row['replicate'] == replicate), key=lambda row: row['concurrency'])
                rows = [row for row in rows if value(row) is not None]
                if rows:
                    axis.plot([row['concurrency'] for row in rows], [value(row) for row in rows],
                              color=colors[cache], marker=markers[replicate], linewidth=1.15,
                              markersize=5, alpha=.83, linestyle='-' if cache else '--')
        axis.set(title=title, xlabel='Concurrent client workers', ylabel=label, ylim=(0, None))
        axis.set_xscale('log', base=2)
        axis.set_xticks([1, 4, 16, 32], labels=['1', '4', '16', '32'])
        axis.grid(axis='y', color='#dfe5ed', linewidth=.65)
    axes[1, 0].set_ylim(0, 105)
    agreement_axis = axes[1, 2]
    for replicate in (1, 2, 3):
        rows = [row for row in result['output_agreement'] if row['replicate'] == replicate and row['paired_successful_requests']]
        if rows:
            agreement_axis.plot([row['concurrency'] for row in rows],
                                [100 * row['exact_text_hash_matches'] / row['paired_successful_requests'] for row in rows],
                                color='#9b6535', marker=markers[replicate], linewidth=1.15, alpha=.8)
    agreement_axis.set(title='Cache modes · exact text agreement', xlabel='Concurrent client workers',
                       ylabel='Matching hashes / paired successful requests (%)', ylim=(0, 105))
    agreement_axis.set_xscale('log', base=2)
    agreement_axis.set_xticks([1, 4, 16, 32], labels=['1', '4', '16', '32'])
    agreement_axis.grid(axis='y', color='#dfe5ed', linewidth=.65)
    handles = [Line2D([], [], color=colors[False], linestyle='--', label='Prefix cache OFF'),
               Line2D([], [], color=colors[True], label='Prefix cache ON')]
    handles += [Line2D([], [], color='#697484', marker=markers[replicate], linestyle='', label=f'Replicate {replicate}') for replicate in (1, 2, 3)]
    figure.legend(handles=handles, loc='outside upper center', ncol=5, frameon=False)
    failures = sum(row['failed'] for row in result['stages'])
    requested = sum(row['requests'] for row in result['stages'])
    mismatches = sum(row['fixed_output_length_mismatches'] for row in result['stages'])
    label = 'PARTIAL EXPERIMENT · ' if incomplete else ''
    figure.suptitle(label + f"{plan['model'].split('/')[-1]} · {gpu_name}\nvLLM {plan['vllm_version']} · 128-token fixed decode · warm synthetic workload", fontsize=15)
    figure.supxlabel(f'{requested:,} measured requests; {failures} failures; {mismatches} output-length mismatches. '
                     'Every replicate shown; no population confidence intervals.\n'
                     'p95 values are sample percentiles. Text agreement is not answer quality. VRAM includes allocated KV cache.', fontsize=9)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ('png', 'svg', 'pdf'):
        path = output / ('controlled-serving-pilot.' + extension)
        figure.savefig(path, dpi=220, bbox_inches='tight')
        paths.append(path)
    plt.close(figure)
    metadata = {'source_status': result['source_status'], 'incomplete': incomplete,
                'plot_library': {'matplotlib': matplotlib.__version__},
                'measured_requests': requested, 'failures': failures,
                'source_file_sha256': result['source_file_sha256'],
                'figure_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (output / 'figure-manifest.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args(argv)
    if args.output.resolve() == args.directory.resolve():
        parser.error('Use a separate figure directory; raw files must remain separate')
    for path in plot(args.directory, args.output, args.allow_incomplete):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
