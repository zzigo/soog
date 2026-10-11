import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import som_registry  # noqa: E402


def write(base, rel, text):
    path = os.path.join(base, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    return path


class RegistryGraphTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = self.tmp.name
        som_registry._cache.update(signature=None, graph=None)
        write(self.base, '1mat/wood.md', '---\nid: wood\ntitle: Wood\ncategory: 1mat\n---\n')
        write(self.base, '0ins/Violin.md',
              '---\nid: violin\ntype: instrument\nkind: individual\norigin: traditional\n'
              'connect: ["[[Wood]]", "[[missing note]]", "[[a/b/wood]]"]\n'
              'moaie:\n  M: "[[wood]]"\n  O: ""\ncrit:\n  O.elec: none\n---\n')
        write(self.base, '0ins/Empty.md', '---\nid: empty\ncategory:\ntype: instrument\n---\n')
        write(self.base, '0ins/Broken.md', '---\nid: [unclosed\n---\n')
        write(self.base, 'soog-templates/t-node.md', '---\nid: tpl\ncategory: 1mat\n---\n')
        write(self.base, '3agn/MOC.md', '---\nid: moc\ntype: moc\ncategory: 3agn\n---\n')
        write(self.base, 'README.md', '# no frontmatter\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_graph(self):
        g = som_registry.build_registry_graph(self.base)
        nodes = {n['id']: n for n in g['nodes']}
        self.assertEqual(set(nodes), {'wood', 'violin', 'empty', 'ghost:missing note'})
        self.assertEqual(nodes['violin']['category'], '2obj')
        self.assertEqual(nodes['empty']['category'], '')
        self.assertEqual(nodes['violin']['crit'], {'O.elec': 'none'})
        self.assertEqual(nodes['violin']['n_crit'], 1)
        links = {(l['source'], l['target'], l['type']) for l in g['links']}
        self.assertEqual(links, {('violin', 'wood', 'connect'),
                                 ('violin', 'ghost:missing note', 'connect'),
                                 ('violin', 'wood', 'moaie:M')})
        self.assertEqual([s['file'] for s in g['meta']['skipped']], ['0ins/Broken.md'])

    def test_cache_invalidates_on_change(self):
        first = som_registry.build_registry_graph(self.base)
        self.assertIs(first, som_registry.build_registry_graph(self.base))
        time.sleep(0.01)
        write(self.base, '5env/air.md', '---\nid: air\ncategory: 5env\n---\n')
        second = som_registry.build_registry_graph(self.base)
        self.assertIsNot(first, second)
        self.assertIn('air', {n['id'] for n in second['nodes']})


if __name__ == '__main__':
    unittest.main()
