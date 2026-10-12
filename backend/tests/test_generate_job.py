import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
for _heavy in ('torch', 'torch.nn', 'transformers', 'highres_3d'):
    try:
        __import__(_heavy)
    except ImportError:
        sys.modules[_heavy] = mock.MagicMock()
import app as soog  # noqa: E402


class GenerateJobTest(unittest.TestCase):
    def setUp(self):
        soog.GENERATE_JOBS.clear()
        self.client = soog.app.test_client()

    def test_start_requires_auth(self):
        res = self.client.post('/api/generate/start', json={'request_id': 'r1', 'prompt': 'x'})
        self.assertIn(res.status_code, (401, 403))

    def test_job_runs_in_background_and_result_is_polled(self):
        def slow_generate():
            time.sleep(0.2)
            return soog.jsonify({'summary': 'ok'}), 200

        with mock.patch.object(soog, 'authenticate_request', return_value={}), \
                mock.patch.object(soog, 'generate', slow_generate):
            res = self.client.post('/api/generate/start', json={'request_id': 'r2', 'prompt': 'x'})
            self.assertEqual(res.status_code, 202)
            pending = self.client.get('/api/generate/result/r2')
            self.assertEqual(pending.status_code, 202)
            for _ in range(50):
                done = self.client.get('/api/generate/result/r2')
                if done.status_code != 202:
                    break
                time.sleep(0.05)
        self.assertEqual(done.status_code, 200)
        self.assertEqual(done.get_json()['summary'], 'ok')

    def test_failed_job_returns_original_error_status(self):
        def failing_generate():
            return soog.jsonify({'error': 'quota'}), 429

        with mock.patch.object(soog, 'authenticate_request', return_value={}), \
                mock.patch.object(soog, 'generate', failing_generate):
            self.client.post('/api/generate/start', json={'request_id': 'r3', 'prompt': 'x'})
            for _ in range(50):
                done = self.client.get('/api/generate/result/r3')
                if done.status_code != 202:
                    break
                time.sleep(0.05)
        self.assertEqual(done.status_code, 429)

    def test_missing_request_id_and_unknown_job(self):
        with mock.patch.object(soog, 'authenticate_request', return_value={}):
            self.assertEqual(self.client.post('/api/generate/start', json={'prompt': 'x'}).status_code, 400)
        self.assertEqual(self.client.get('/api/generate/result/nope').status_code, 404)


if __name__ == '__main__':
    unittest.main()
