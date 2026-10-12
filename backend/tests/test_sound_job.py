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


class SoundJobTest(unittest.TestCase):
    def test_job_reports_error_through_progress(self):
        with mock.patch.object(soog, 'generate_sound_samples', side_effect=RuntimeError('boom')):
            soog._run_sound_job('item', '/nonexistent.json', 'p', 's', 'm', 'req-err')
        progress = soog.get_generation_progress('req-err')
        self.assertEqual(progress['status'], 'error')
        self.assertIn('boom', progress['reasoning_preview'])
        self.assertNotIn('item', soog.SOUND_JOBS)

    def test_job_marks_completed_after_saving_metadata(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'item.json')
            with open(path, 'w') as f:
                json.dump({'prompt': 'p'}, f)
            with mock.patch.object(soog, 'generate_sound_samples', return_value=([{'url': 'u', 'prompt': 'x'}], 'model')):
                soog._run_sound_job('item', path, 'p', 's', 'm', 'req-ok')
            saved = json.load(open(path))
        self.assertEqual(saved['sound_model'], 'model')
        self.assertEqual(soog.get_generation_progress('req-ok')['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
