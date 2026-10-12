import os
import sys
import threading
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
import auth_usage  # noqa: E402


class CancelTest(unittest.TestCase):
    def setUp(self):
        soog.GENERATION_PROGRESS.clear()
        soog.GENERATE_JOBS.clear()
        self.client = soog.app.test_client()

    def test_update_after_cancel_raises_and_terminal_update_does_not(self):
        soog.update_generation_progress('c1', stage='plot_llm', status='running')
        self.assertEqual(soog.request_generation_cancel('c1'), 'cancelling')
        with self.assertRaises(soog.GenerationCancelled):
            soog.update_generation_progress('c1', stage='building_geometry', status='running')
        with self.assertRaises(soog.GenerationCancelled):
            soog.update_generation_progress('c1', reasoning_preview='streaming token')
        soog.update_generation_progress('c1', stage='cancelled', status='cancelled')
        self.assertEqual(soog.get_generation_progress('c1')['status'], 'cancelled')

    def test_cancel_unknown_and_finished_requests(self):
        self.assertIsNone(soog.request_generation_cancel('nope'))
        self.assertIsNone(soog.request_generation_cancel(''))
        soog.update_generation_progress('c2', stage='done', status='completed')
        self.assertEqual(soog.request_generation_cancel('c2'), 'completed')
        soog.update_generation_progress('c2', stage='again', status='running')

    def test_cancelled_exception_is_not_swallowed_by_except_exception(self):
        soog.update_generation_progress('c3', stage='a', status='running')
        soog.request_generation_cancel('c3')
        reached_fallback = []
        try:
            try:
                soog.update_generation_progress('c3', stage='b', status='running')
            except Exception:
                reached_fallback.append(True)
        except soog.GenerationCancelled:
            pass
        self.assertEqual(reached_fallback, [])

    def test_cancellable_view_returns_499_and_marks_cancelled(self):
        soog.update_generation_progress('c4', stage='a', status='running')
        soog.request_generation_cancel('c4')

        @soog.cancellable
        def view():
            soog.update_generation_progress('c4', stage='b', status='running')
            return soog.jsonify({'ok': True}), 200

        with soog.app.test_request_context():
            body, status = view()
        self.assertEqual(status, 499)
        self.assertTrue(body.get_json()['cancelled'])
        self.assertEqual(soog.get_generation_progress('c4')['status'], 'cancelled')

    def test_endpoint_requires_auth_and_reports_state(self):
        soog.update_generation_progress('c5', stage='a', status='running')
        self.assertIn(self.client.post('/api/generate/cancel/c5').status_code, (401, 403))
        with mock.patch.object(auth_usage, 'authenticate_request', return_value={'is_admin': False}):
            ok = self.client.post('/api/generate/cancel/c5')
            missing = self.client.post('/api/generate/cancel/zzz')
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.get_json()['status'], 'cancelling')
        self.assertEqual(missing.status_code, 404)

    def test_background_job_ends_with_499_result(self):
        started = threading.Event()

        def long_generate():
            soog.update_generation_progress('c6', stage='plot_llm', status='running')
            started.set()
            for _ in range(100):
                time.sleep(0.02)
                soog.update_generation_progress('c6', reasoning_preview='thinking')
            return soog.jsonify({'ok': True}), 200

        with mock.patch.object(soog, 'authenticate_request', return_value={'is_admin': False}), \
                mock.patch.object(auth_usage, 'authenticate_request', return_value={'is_admin': False}), \
                mock.patch.object(soog, 'generate', soog.cancellable(long_generate)):
            self.client.post('/api/generate/start', json={'request_id': 'c6', 'prompt': 'x'})
            self.assertTrue(started.wait(2))
            self.client.post('/api/generate/cancel/c6')
            for _ in range(100):
                done = self.client.get('/api/generate/result/c6')
                if done.status_code != 202:
                    break
                time.sleep(0.03)
        self.assertEqual(done.status_code, 499)
        self.assertTrue(done.get_json()['cancelled'])


if __name__ == '__main__':
    unittest.main()
