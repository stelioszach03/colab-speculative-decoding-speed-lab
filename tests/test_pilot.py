import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inference_lab import pilot
from inference_lab.benchmark import load_workload
from inference_lab.pilot_report import aggregate, gpu_rows


class PilotTests(unittest.TestCase):
    def test_frozen_workload_covers_matched_strata_without_external_data(self):
        rows, strata = pilot.workload()
        self.assertEqual(len(rows), 64)
        self.assertEqual(len({row['prompt'] for row in rows}), 64)
        for length in (64, 192, 448, 896):
            for kind in ('shared', 'unique'):
                self.assertEqual(sum(s['context_words'] == length and s['prefix_stratum'] == kind for s in strata), 8)
        self.assertEqual(pilot.protocol()['workload_sha256'], hashlib.sha256(pilot.workload_bytes()).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'workload.jsonl'
            path.write_bytes(pilot.workload_bytes())
            self.assertEqual(load_workload(path), rows)

    def test_cells_prespecified_paired_and_bounded(self):
        plan = pilot.protocol()
        self.assertEqual(len(plan['cells']), 6)
        for replicate in (1, 2, 3):
            cells = [cell for cell in plan['cells'] if cell['replicate'] == replicate]
            self.assertEqual({cell['prefix_cache'] for cell in cells}, {True, False})
            self.assertEqual(cells[0]['concurrency'], cells[1]['concurrency'])
            self.assertEqual(cells[0]['seed'], cells[1]['seed'])
            self.assertEqual(set(cells[0]['concurrency']), {1, 4, 16, 32})
        self.assertEqual(plan['total_measured_requests'], 6 * 4 * 64)
        self.assertEqual(plan['maximum_requested_output_tokens'], 6 * 4 * 68 * 128)
        self.assertEqual(plan['experiment_deadline_s'], 2700)

    def test_server_variants_only_differ_by_cache_flag(self):
        off, on = pilot.server_command(False), pilot.server_command(True)
        self.assertEqual(off[:-1], on[:-1])
        self.assertEqual(off[-1], '--no-enable-prefix-caching')
        self.assertEqual(on[-1], '--enable-prefix-caching')
        self.assertEqual(on[on.index('--revision') + 1], pilot.REVISION)
        self.assertIn('127.0.0.1', on)

    def test_plan_has_no_side_effects(self):
        with patch('inference_lab.pilot.run') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pilot.main([]), 0)
        run.assert_not_called()

    def test_machine_readable_protocol_matches_runner(self):
        frozen = json.loads((pilot.ROOT / 'protocols/controlled-pilot-v1.json').read_text())
        self.assertEqual(frozen, pilot.protocol())

    def test_gpu_records_are_direct_timestamps_units_and_missing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gpu.csv'
            path.write_text('timestamp, memory.used [MiB], utilization.gpu [%], power.draw [W]\n'
                            '2026/09/23 12:00:00.500, 4096 MiB, 51 %, 120.5 W\n'
                            '2026/09/23 12:00:01.000, 4100 MiB, 0 %, [N/A]\n')
            rows = gpu_rows(path)
            self.assertEqual(rows[0]['time'], datetime(2026, 9, 23, 12, 0, 0, 500000, timezone.utc))
            self.assertEqual(rows[0]['memory_mib'], 4096)
            self.assertEqual(rows[0]['utilization_percent'], 51)
            self.assertIsNone(rows[1]['power_w'])
            path.write_text('timestamp, memory.used [MiB]\ninvalid, 123 MiB\n')
            with self.assertRaises(ValueError):
                gpu_rows(path)

    def test_interrupted_pilot_reports_missing_cells_without_fabricating_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot.write_json(root / 'protocol.json', pilot.protocol())
            pilot.write_json(root / 'manifest.json', {'status': 'interrupted'})
            result = aggregate(root)
            self.assertEqual(result['stages'], [])
            self.assertEqual(result['output_agreement'], [])
            self.assertEqual(len(result['incomplete_cells']), 6)
            self.assertIsNone(result['cost_usd'])


if __name__ == '__main__':
    unittest.main()
