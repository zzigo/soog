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


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


TAGS = {'models': [{'name': 'qwen3:4b'}, {'name': 'nomic-embed-text:latest'}, {'name': 'gpt-4o'}]}
CAPS = {'qwen3:4b': ['completion'], 'nomic-embed-text:latest': ['embedding'], 'gpt-4o': ['completion']}


def fake_get(url, **kwargs):
    return FakeResponse(200, TAGS)


def fake_post(url, json=None, **kwargs):
    return FakeResponse(200, {'capabilities': CAPS[json['model']]})


class ModelListTest(unittest.TestCase):
    def setUp(self):
        soog._MODEL_LIST_CACHE.update(at=0.0, key=None, value=None)
        soog._MODEL_CAPS_CACHE.clear()

    def test_embedding_models_and_hidden_models_are_excluded(self):
        with mock.patch.dict(os.environ, {'SOOG_MODEL_HIDE': 'gpt-4o'}), \
                mock.patch.object(soog.requests, 'get', fake_get), \
                mock.patch.object(soog.requests, 'post', fake_post):
            res = soog.fetch_ollama_models(force=True)
        self.assertEqual(res['models'], ['qwen3:4b'])

    def test_new_model_appears_after_force_refresh(self):
        with mock.patch.object(soog.requests, 'get', fake_get), \
                mock.patch.object(soog.requests, 'post', fake_post):
            first = soog.fetch_ollama_models(force=True)
            self.assertIn('gpt-4o', first['models'])
        tags = {'models': TAGS['models'] + [{'name': 'new:1b'}]}
        caps = dict(CAPS, **{'new:1b': ['completion']})
        with mock.patch.object(soog.requests, 'get', lambda url, **k: FakeResponse(200, tags)), \
                mock.patch.object(soog.requests, 'post', lambda url, json=None, **k: FakeResponse(200, {'capabilities': caps[json['model']]})):
            cached = soog.fetch_ollama_models()
            self.assertNotIn('new:1b', cached['models'])
            fresh = soog.fetch_ollama_models(force=True)
            self.assertIn('new:1b', fresh['models'])

    def test_cloud_models_hidden_unless_allowed(self):
        tags = {'models': [{'name': 'qwen3:4b'}, {'name': 'glm-5.3:cloud'}, {'name': 'gemma4:31b-cloud'}]}
        caps = {'qwen3:4b': ['completion'], 'glm-5.3:cloud': ['completion'], 'gemma4:31b-cloud': ['completion']}
        get = lambda url, **k: FakeResponse(200, tags)
        post = lambda url, json=None, **k: FakeResponse(200, {'capabilities': caps[json['model']]})
        with mock.patch.object(soog.requests, 'get', get), mock.patch.object(soog.requests, 'post', post):
            self.assertEqual(soog.fetch_ollama_models(force=True)['models'], ['qwen3:4b'])
            with mock.patch.dict(os.environ, {'SOOG_ALLOW_CLOUD_MODELS': '1'}):
                self.assertEqual(len(soog.fetch_ollama_models(force=True)['models']), 3)

    def test_set_model_requires_auth(self):
        client = soog.app.test_client()
        res = client.post('/api/ollama/model', json={'model': 'qwen3:4b'})
        self.assertIn(res.status_code, (401, 403))


if __name__ == '__main__':
    unittest.main()
