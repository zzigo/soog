import base64
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
import app as soog  # noqa: E402
import organogram_spec as spec_lib  # noqa: E402


FLUTE = {
    'parts': [
        {'name': 'bamboo tube', 'class': 'aerophone', 'material': 'bamboo', 'count': 1},
        {'name': 'finger holes', 'class': 'idiophone', 'material': 'wood', 'count': 5},
        {'name': 'air jet', 'class': 'aerophone', 'material': 'other', 'count': 1},
    ],
    'flows': [{'source': 2, 'target': 0, 'kind': 'airflow'}],
}
HARP = {
    'parts': [
        {'name': 'glass strings', 'class': 'chordophone', 'material': 'glass', 'count': 4},
        {'name': 'copper resonator', 'class': 'idiophone', 'material': 'copper', 'count': 1},
        {'name': 'bow', 'class': 'membranophone', 'material': 'skin', 'count': 1},
    ],
    'flows': [],
}


class SpecTest(unittest.TestCase):
    def test_normalize_cleans_classes_materials_and_flows(self):
        spec = spec_lib.normalize_spec({
            'parts': [{'name': 'x', 'class': 'Strings', 'material': 'Steel', 'count': '3'},
                      {'name': 'y', 'class': '??', 'material': 'unknown', 'count': 0}],
            'flows': [{'source': 0, 'target': 9, 'kind': 'x'}],
        })
        self.assertEqual(spec['parts'][0]['class'], 'chordophone')
        self.assertEqual(spec['parts'][0]['material'], 'metal')
        self.assertEqual(spec['parts'][0]['count'], 3)
        self.assertEqual(spec['parts'][1]['count'], 1)
        self.assertEqual(len(spec['flows']), 1)  # invalid flow dropped, chain added
        self.assertIsNone(spec_lib.normalize_spec({'parts': []}))
        self.assertIsNone(spec_lib.normalize_spec('nope'))

    def test_rendered_code_runs_and_follows_hood_conventions(self):
        code = spec_lib.render_spec_code(spec_lib.normalize_spec(FLUTE))
        self.assertIn('Circle(', code)
        self.assertIn('#ffff00', code)          # bamboo
        self.assertIn("ax.set_axis_off()", code)
        image = soog.execute_matplotlib_code(code)
        self.assertTrue(base64.b64decode(image).startswith(b'\x89PNG'))

    def test_different_instruments_give_different_organograms(self):
        a = spec_lib.render_spec_code(spec_lib.normalize_spec(FLUTE))
        b = spec_lib.render_spec_code(spec_lib.normalize_spec(HARP))
        self.assertNotEqual(a, b)
        self.assertNotEqual(soog.execute_matplotlib_code(a), soog.execute_matplotlib_code(b))

    def test_text_fallback_reflects_the_prompt(self):
        spec = spec_lib.spec_from_text('instrument powered by electrical signals of a Venus flytrap', 'bamboo tube')
        classes = {p['class'] for p in spec['parts']}
        self.assertIn('electronophone', classes)
        plain = spec_lib.spec_from_text('a struck glass bar')
        self.assertNotEqual(spec, plain)
        soog.execute_matplotlib_code(spec_lib.render_spec_code(spec))


class PipelineTest(unittest.TestCase):
    def test_spec_mode_draws_the_model_spec(self):
        data = dict(FLUTE, summary='A bamboo flute.', materials='1 bamboo tube')

        def fake_structured(system_prompt, user_prompt, **kwargs):
            return data, {'model': 'fake', 'endpoint': 'x'}

        with mock.patch.object(soog, 'call_ollama_structured', fake_structured):
            summary, materials, code, image, raw, meta = soog.generate_organogram_from_spec(
                'system', 'bamboo flute', llm_timeout_sec=5, llm_max_tokens=64, call_timeout=5
            )
        self.assertEqual(summary, 'A bamboo flute.')
        self.assertIn('bamboo tube', code)
        self.assertTrue(base64.b64decode(image).startswith(b'\x89PNG'))

    def test_empty_spec_raises_so_fallback_can_run(self):
        with mock.patch.object(soog, 'call_ollama_structured', lambda *a, **k: ({'summary': 's', 'materials': 'm', 'parts': []}, {})):
            with self.assertRaises(RuntimeError):
                soog.generate_organogram_from_spec('s', 'p', llm_timeout_sec=5, llm_max_tokens=64, call_timeout=5)


if __name__ == '__main__':
    unittest.main()
