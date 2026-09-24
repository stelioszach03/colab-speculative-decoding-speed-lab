from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from inference_lab.benchmark import summarize
from inference_lab.pilot import protocol, write_json
from inference_lab.pilot_report import aggregate, check_summary, gpu_rows
from inference_lab.pilot_plots import plot


class AnalysisIntegrityTests(unittest.TestCase):
    def rows(self):
        return [{'request_id': 'measured-a', 'workload_id': 'a', 'ok': True, 'latency_s': 1.2,
                 'ttft_s': .2, 'itl_stream_s': [.1, .2], 'tpot_s': .15, 'completion_tokens': 3, 'error': None},
                {'request_id': 'measured-b', 'workload_id': 'b', 'ok': False, 'latency_s': 3,
                 'ttft_s': None, 'itl_stream_s': [], 'tpot_s': None, 'completion_tokens': None, 'error': 'timeout'}]

    def test_recomputation_preserves_failed_requests_and_detects_edited_metrics(self):
        rows = self.rows()
        stage = summarize(rows, 4.5, 1)
        check_summary(stage, rows)
        for key, value in [('failed', 0), ('completion_tokens_per_s', 99), ('ttft_s', {'p95': 0})]:
            changed = deepcopy(stage)
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'does not match raw'):
                check_summary(changed, rows)

    def test_duplicate_requests_cannot_boost_sample_count_or_hash_agreement(self):
        rows = self.rows()
        duplicate = deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError, 'Duplicate measured request'):
            check_summary(summarize(rows + [duplicate], 4, 1), rows + [duplicate])
        duplicate['request_id'] = 'new-id-same-prompt'
        with self.assertRaisesRegex(ValueError, 'Duplicate workload'):
            check_summary(summarize(rows + [duplicate], 4, 1), rows + [duplicate])

    def test_gpu_nonfinite_and_impossible_utilization_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gpu.csv'
            for value in ('NaN', 'Infinity', '-1', '101'):
                path.write_text(f'timestamp,utilization.gpu\n2026/09/23 12:00:00.500,{value}\n')
                with self.subTest(value=value), self.assertRaises(ValueError):
                    gpu_rows(path)

    def test_changed_protocol_is_not_silently_reported_as_frozen_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            altered = protocol()
            altered['max_tokens'] = 64
            write_json(root / 'protocol.json', altered)
            write_json(root / 'manifest.json', {'status': 'completed'})
            with self.assertRaisesRegex(ValueError, 'differs from the frozen'):
                aggregate(root)

    def test_no_measurements_cannot_produce_placeholder_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / 'protocol.json', protocol())
            write_json(root / 'manifest.json', {'status': 'interrupted'})
            with self.assertRaisesRegex(ValueError, 'No measured stages'):
                plot(root, root / 'figures', allow_incomplete=True)
            self.assertFalse((root / 'figures').exists())


if __name__ == '__main__':
    unittest.main()
