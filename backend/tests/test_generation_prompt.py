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


class Captured(BaseException):
    pass


class GenerationPromptTest(unittest.TestCase):
    def test_plot_call_uses_prompt_txt_as_system_prompt(self):
        seen = {}

        def fake_structured(system_prompt, user_prompt, **kwargs):
            seen['system'] = system_prompt
            raise Captured()

        system_prompt, instruction, template = soog.build_generation_inputs(
            soog.get_prompt_content(), 'a bowed glass harp', {}, 'REGISTRY-NOTE'
        )
        with mock.patch.object(soog, 'call_ollama_structured', fake_structured):
            with self.assertRaises(Captured):
                soog.generate_with_image_required(
                    system_prompt, 'a bowed glass harp', instruction,
                    max_attempts=1, llm_timeout_sec=5, llm_max_tokens=64,
                    deadline_at=None, plot_output_template=template,
                )
        self.assertIn('Mantle Hood', seen['system'])
        self.assertIn('ORGANOGRAM PRIMITIVES', seen['system'])
        self.assertIn('REGISTRY-NOTE', seen['system'])

    def test_instruction_does_not_ask_for_charts(self):
        _, instruction, _ = soog.build_generation_inputs('x', 'a flute', {}, '')
        self.assertNotIn('PipeElement', instruction)
        self.assertNotIn('plot_geometry_vs_frequency', instruction)


if __name__ == '__main__':
    unittest.main()
