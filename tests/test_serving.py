import contextlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from inference_lab.benchmark import StreamRecord, endpoint, load_workload, main, percentiles, stream_request, summarize


class MetricsTests(unittest.TestCase):
    def test_stream_observations_ignore_role_and_usage(self):
        record = StreamRecord()
        record.consume('{"choices":[{"delta":{"role":"assistant"}}]}', .05)
        record.consume('{"model":"fixture-model","choices":[{"delta":{"content":"hello"}}]}', .2)
        record.consume('{"choices":[{"delta":{"content":" world"}}]}', .5)
        record.consume('{"choices":[{"delta":{},"finish_reason":"stop"}]}', .7)
        record.consume('{"choices":[],"usage":{"completion_tokens":3,"prompt_tokens":8}}', .8)
        record.consume('[DONE]', .9)
        result = record.metrics()
        self.assertEqual(result['ttft_s'], .2)
        self.assertEqual(result['itl_stream_s'], [.3])
        self.assertEqual(result['tpot_s'], .15)
        self.assertEqual(result['completion_tokens'], 3)
        self.assertEqual(result['actual_model'], 'fixture-model')
        self.assertTrue(record.done)

    def test_missing_usage_is_unknown_not_char_count(self):
        record = StreamRecord()
        record.consume('{"choices":[{"delta":{"content":"many words here"}}]}', .1)
        self.assertIsNone(record.metrics()['completion_tokens'])
        self.assertIsNone(record.metrics()['tpot_s'])

    def test_invalid_usage_rejected(self):
        for value in [-1, True, 2.5, '3']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                StreamRecord().consume(json.dumps({'usage': {'completion_tokens': value}}), 0)

    def test_reasoning_tokens_do_not_produce_text_tpot(self):
        record = StreamRecord()
        record.consume('{"choices":[{"delta":{"reasoning_content":"reasoning"}}]}', .1)
        record.consume('{"choices":[{"delta":{"content":"a"}}]}', .2)
        record.consume('{"choices":[{"delta":{"content":"b"}}]}', .3)
        record.consume('{"usage":{"completion_tokens":20}}', .4)
        self.assertTrue(record.metrics()['non_text_output_seen'])
        self.assertIsNone(record.metrics()['tpot_s'])

    def test_percentiles_and_failures_are_not_hidden(self):
        self.assertEqual(percentiles([1, 3])['p50'], 2)
        self.assertIsNone(percentiles([])['p99'])
        good = dict(ok=True, latency_s=2, ttft_s=.5, itl_stream_s=[.2], tpot_s=.2, completion_tokens=4, error=None)
        bad = {**good, 'ok': False, 'latency_s': 10, 'error': 'timeout'}
        result = summarize([good, bad], 12, 1)
        self.assertEqual(result['failed'], 1)
        self.assertEqual(result['latency_all_s']['count'], 2)
        self.assertEqual(result['latency_success_s']['count'], 1)
        self.assertEqual(result['completion_tokens_per_s'], 4 / 12)
        self.assertEqual(result['failures_by_kind'], {'timeout': 1})
        self.assertIsNone(summarize([{**good, 'completion_tokens': None}], 1, 1)['completion_tokens_per_s'])

    def test_endpoint_rejects_remote_credentials_and_redirect_like_paths(self):
        for url in ['https://example.com/v1', 'http://user:secret@localhost/v1',
                    'http://localhost/v1?api_key=secret', 'file:///tmp/x', 'http://localhost/redirect']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                endpoint(url)
        self.assertEqual(endpoint('http://[::1]:8000/v1').hostname, '::1')

    def test_workload_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            path.write_text('{"id":"a","prompt":"x"}\n' * 2)
            with self.assertRaises(ValueError):
                load_workload(path)

    def test_plan_has_no_network_or_output(self):
        with tempfile.TemporaryDirectory() as directory, patch('inference_lab.benchmark.stream_request') as request:
            output = Path(directory) / 'results'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--model', 'fixture', '--output', str(output)]), 0)
            request.assert_not_called()
            self.assertFalse(output.exists())


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.server.observed.append({'path': self.path, 'payload': data, 'authorization': self.headers.get('Authorization')})
                prompt = data['messages'][0]['content']
                if prompt == 'failure':
                    self.send_response(429)
                    self.end_headers()
                    self.wfile.write(b'private error body must not be recorded')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                if prompt == 'malformed':
                    self.wfile.write(b'data: not-json\n\n')
                    return
                events = [': keepalive\r\n\r\n',
                          'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
                          'data: {"model":"fixture-model","choices":[{"delta":{"content":"Hello"}}]}\n\n',
                          'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
                          'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
                          'data: {"choices":[],"usage":{"completion_tokens":2,"prompt_tokens":3}}\n\n']
                if prompt != 'incomplete':
                    events.append('data: [DONE]\n\n')
                for event in events:
                    # Deliberately split JSON payloads across transport writes.
                    encoded = event.encode()
                    self.wfile.write(encoded[:9])
                    self.wfile.flush()
                    self.wfile.write(encoded[9:])
                    self.wfile.flush()
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.observed = []
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/v1'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_stream_end_to_end_and_error_categories(self):
        row = stream_request(self.url, 'fixture-model', 'success', 8, 3, 'test-secret')
        self.assertTrue(row['ok'])
        self.assertEqual(row['completion_tokens'], 2)
        self.assertEqual(len(row['events']), 2)
        self.assertGreaterEqual(row['latency_s'], row['ttft_s'])
        self.assertNotIn('test-secret', json.dumps(row))
        self.assertEqual(self.server.observed[-1]['authorization'], 'Bearer test-secret')
        for prompt, expected in [('failure', 'http_error'), ('malformed', 'invalid_stream'), ('incomplete', 'incomplete_stream')]:
            with self.subTest(prompt=prompt):
                row = stream_request(self.url, 'fixture-model', prompt, 8, 3)
                self.assertEqual(row['error'], expected)
                self.assertFalse(row['ok'])
                self.assertNotIn('private error', json.dumps(row))

    def test_fixed_decode_request_is_explicit_and_default_unchanged(self):
        stream_request(self.url, 'fixture-model', 'success', 8, 3, fixed_output=True)
        payload = self.server.observed[-1]['payload']
        self.assertEqual(payload['min_tokens'], 8)
        self.assertTrue(payload['ignore_eos'])
        stream_request(self.url, 'fixture-model', 'success', 8, 3)
        self.assertNotIn('min_tokens', self.server.observed[-1]['payload'])

    def test_suite_writes_measured_and_warmup_raw_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(['--model', 'fixture-model', '--base-url', self.url, '--requests', '4',
                             '--concurrency', '1', '2', '--warmup', '1', '--max-tokens', '8',
                             '--output', str(output), '--execute'])
            self.assertEqual(code, 0)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertIsNone(manifest['gpu_telemetry'])
            self.assertIsNone(manifest['cost'])
            records = [json.loads(line) for line in (output / 'requests.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 10)
            self.assertEqual(sum(row['warmup'] for row in records), 2)
            summaries = json.loads((output / 'summary.json').read_text())['stages']
            self.assertEqual([stage['requests'] for stage in summaries], [4, 4])
            self.assertEqual([stage['failed'] for stage in summaries], [0, 0])
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                main(['--model', 'fixture-model', '--base-url', self.url, '--output', str(output), '--execute'])


if __name__ == '__main__':
    unittest.main()
