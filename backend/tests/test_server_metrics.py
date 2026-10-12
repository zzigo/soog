import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
for _heavy in ('torch', 'torch.nn', 'transformers', 'highres_3d'):
    try:
        __import__(_heavy)
    except ImportError:
        sys.modules[_heavy] = mock.MagicMock()

import server_metrics as sm  # noqa: E402
import app as soog  # noqa: E402

STAT_A = "cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 50 0 50 400 0 0 0 0 0 0\n"
STAT_B = "cpu  200 0 200 1000 0 0 0 0 0 0\ncpu0 100 0 100 500 0 0 0 0 0 0\n"
MEMINFO = "MemTotal:       16000000 kB\nMemFree:         1000000 kB\nMemAvailable:    6000000 kB\nSwapTotal:       8000000 kB\nSwapFree:        4000000 kB\n"


class ParseTest(unittest.TestCase):
    def test_proc_stat_busy_and_total(self):
        self.assertEqual(sm.parse_proc_stat(STAT_A), (200, 1000))

    def test_proc_stat_counts_iowait_as_idle(self):
        busy, total = sm.parse_proc_stat("cpu  10 0 10 70 10 0 0 0\n")
        self.assertEqual((busy, total), (20, 100))

    def test_proc_stat_unreadable(self):
        self.assertIsNone(sm.parse_proc_stat(""))
        self.assertIsNone(sm.parse_proc_stat("cpu  1 2\n"))

    def test_meminfo(self):
        mem = sm.parse_meminfo(MEMINFO)
        self.assertEqual(mem['MemTotal'], 16000000)
        self.assertEqual(mem['SwapFree'], 4000000)


class SamplerTest(unittest.TestCase):
    def test_utilisation_between_two_samples(self):
        readings = iter([STAT_A, STAT_A, STAT_B])
        sampler = sm.CpuSampler(reader=lambda: next(readings))
        with mock.patch.object(sm.time, 'sleep'):
            first = sampler.sample()  # primes with STAT_A, re-reads STAT_A -> no elapsed jiffies
        self.assertIsNone(first)
        with mock.patch.object(sm.time, 'monotonic', return_value=sampler._last_at + 5):
            second = sampler.sample()  # STAT_B: busy +200 of total +400
        self.assertEqual(second, 50.0)

    def test_unreadable_proc_returns_none(self):
        self.assertIsNone(sm.CpuSampler(reader=lambda: '').sample())


class MetricsTest(unittest.TestCase):
    def test_snapshot_reports_memory_and_swap(self):
        sampler = mock.Mock()
        sampler.sample.return_value = 42.5
        with mock.patch.object(sm, '_read', side_effect=lambda path: MEMINFO if path == sm.PROC_MEMINFO else ''):
            metrics = sm.read_server_metrics(sampler)
        self.assertEqual(metrics['cpu_pct'], 42.5)
        self.assertEqual(metrics['mem_total_mb'], round(16000000 / 1024))
        self.assertEqual(metrics['mem_used_mb'], round(10000000 / 1024))
        self.assertEqual(metrics['swap_used_mb'], round(4000000 / 1024))


class StageEventsTest(unittest.TestCase):
    def setUp(self):
        soog.GENERATION_PROGRESS.clear()

    def test_stage_change_appends_one_event_with_timing_and_cpu(self):
        with mock.patch.object(soog.CPU_SAMPLER, 'sample', return_value=37.0):
            soog.update_generation_progress('r1', stage='received', status='starting')
            soog.update_generation_progress('r1', stage='received', status='running')  # same stage
            soog.update_generation_progress('r1', stage='plot_llm', status='running')
        progress = soog.get_generation_progress('r1')
        self.assertEqual([e['stage'] for e in progress['events']], ['received', 'plot_llm'])
        self.assertEqual(progress['events'][0]['t_ms'], 0)
        self.assertEqual(progress['events'][1]['cpu_pct'], 37.0)
        self.assertEqual(progress['events'][0]['status'], 'running')  # follows status updates
        self.assertIn('started_at', progress)

    def test_events_are_capped(self):
        with mock.patch.object(soog.CPU_SAMPLER, 'sample', return_value=None):
            for i in range(soog.STAGE_EVENT_LIMIT + 10):
                soog.update_generation_progress('r2', stage=f's{i}', status='running')
        self.assertEqual(len(soog.get_generation_progress('r2')['events']), soog.STAGE_EVENT_LIMIT)

    def test_progress_endpoint_hides_cpu_from_non_operators(self):
        with mock.patch.object(soog.CPU_SAMPLER, 'sample', return_value=37.0):
            soog.update_generation_progress('r3', stage='received', status='starting')
        client = soog.app.test_client()
        public = client.get('/api/generate/progress/r3').get_json()
        self.assertNotIn('cpu_pct', public['events'][0])
        self.assertNotIn('server', public)
        with mock.patch.object(soog, 'authenticate_request', return_value={'is_admin': True}), \
                mock.patch.object(soog, 'read_server_metrics', return_value={'cpu_pct': 12.0}):
            operator = client.get('/api/generate/progress/r3').get_json()
        self.assertEqual(operator['events'][0]['cpu_pct'], 37.0)
        self.assertEqual(operator['server']['cpu_pct'], 12.0)

    def test_server_metrics_endpoint_requires_admin(self):
        client = soog.app.test_client()
        self.assertEqual(client.get('/api/admin/server-metrics').status_code, 401)


if __name__ == '__main__':
    unittest.main()
