"""
Graph builder for the Speculative Organology registry (github.com/zzigo/som-registry).

Reads a local checkout of the registry (SOM_REGISTRY_DIR, default /opt/som-registry)
and returns {nodes, links, meta} for /api/somap/registry. Independent of backend/soopub.
"""
import datetime
import os
import re
import subprocess
import threading
import unicodedata

import yaml

DEFAULT_REGISTRY_DIR = '/opt/som-registry'
CATEGORIES = ('1mat', '2obj', '3agn', '4int', '5env', '0pri', '0dict')
LINK_KEYS = ('hyper', 'hypo', 'connect')
MOAIE_KEYS = ('M', 'O', 'A', 'I', 'E')
SKIP_DIRS = {'soog-templates', '.git', '.github', 'node_modules'}

_FRONTMATTER_RE = re.compile(r'^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)', re.S)
_WIKILINK_RE = re.compile(r'\[\[([^\]]+)\]\]')

MAX_NOTE_CHARS = 300_000

_cache = {'signature': None, 'graph': None}
_cache_lock = threading.Lock()


def registry_dir():
    return os.environ.get('SOM_REGISTRY_DIR') or DEFAULT_REGISTRY_DIR


def _nfc(text):
    return unicodedata.normalize('NFC', str(text))


def _slug(text):
    text = unicodedata.normalize('NFKD', _nfc(text)).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def _key(text):
    return _nfc(text).strip().casefold()


def _str(value):
    if value is None:
        return ''
    return _nfc(value).strip()


def _wikilink_targets(value):
    """Wikilink targets in a frontmatter value (string, list or nested list)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_wikilink_targets(item))
        return out
    if isinstance(value, dict):
        return []
    targets = []
    for raw in _WIKILINK_RE.findall(str(value)):
        target = raw.split('|', 1)[0].split('#', 1)[0].strip()
        target = re.sub(r'\.md$', '', target, flags=re.IGNORECASE)
        if target:
            targets.append(_nfc(target))
    return targets


def _markdown_files(base):
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith('.'))
        for name in sorted(files):
            if name.lower().endswith('.md') and not name.startswith('.'):
                yield os.path.join(root, name)


def _signature(base, files):
    parts = [_git_commit(base)]
    for path in files:
        try:
            st = os.stat(path)
        except OSError:
            continue
        parts.append(f"{os.path.relpath(path, base)}:{st.st_mtime_ns}:{st.st_size}")
    return '|'.join(parts)


def _git_commit(base):
    try:
        out = subprocess.run(['git', '-C', base, 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return _read_git_head(base)


def _read_git_head(base):
    """Read HEAD without git, for checkouts owned by another user (git safe.directory)."""
    git_dir = os.path.join(base, '.git')
    try:
        with open(os.path.join(git_dir, 'HEAD'), encoding='utf-8') as fh:
            head = fh.read().strip()
        if not head.startswith('ref: '):
            return head
        ref = head[5:]
        ref_path = os.path.join(git_dir, ref)
        if os.path.isfile(ref_path):
            with open(ref_path, encoding='utf-8') as fh:
                return fh.read().strip()
        with open(os.path.join(git_dir, 'packed-refs'), encoding='utf-8') as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        pass
    return ''


def _parse_note(path, base):
    """Return (frontmatter dict | None, error str | None). None, None means no frontmatter."""
    with open(path, 'r', encoding='utf-8') as fh:
        text = fh.read()
    match = _FRONTMATTER_RE.match(text.lstrip('﻿'))
    if not match:
        return None, None
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        return None, str(exc).splitlines()[0]
    if data is None:
        return None, None
    if not isinstance(data, dict):
        return None, 'frontmatter is not a mapping'
    return data, None


def build_registry_graph(base=None):
    base = base or registry_dir()
    if not os.path.isdir(base):
        raise FileNotFoundError(f"registry checkout not found: {base}")

    files = list(_markdown_files(base))
    signature = _signature(base, files)
    with _cache_lock:
        if _cache['signature'] == signature and _cache['graph'] is not None:
            return _cache['graph']

    records = []
    skipped = []
    seen_ids = set()
    for path in files:
        rel = _nfc(os.path.relpath(path, base).replace(os.sep, '/'))
        stem = _nfc(os.path.splitext(os.path.basename(path))[0])
        if stem.startswith('t-'):
            continue
        try:
            fm, err = _parse_note(path, base)
        except (OSError, UnicodeDecodeError) as exc:
            skipped.append({'file': rel, 'reason': str(exc)})
            continue
        if err:
            skipped.append({'file': rel, 'reason': err})
            continue
        if fm is None:
            continue
        note_type = _str(fm.get('type'))
        if note_type == 'moc':
            continue
        has_category_key = 'category' in fm
        category = _str(fm.get('category'))
        if not has_category_key and note_type != 'instrument':
            continue
        if not has_category_key and note_type == 'instrument':
            category = '2obj'
        if category and category not in CATEGORIES:
            category = ''
        title = _str(fm.get('title')) or stem
        node_id = _str(fm.get('id')) or _slug(title) or _slug(stem)
        if node_id in seen_ids:
            # NFC/NFD duplicate files of the same note (e.g. Ü.md) share an id.
            continue
        seen_ids.add(node_id)
        crit = fm.get('crit') if isinstance(fm.get('crit'), dict) else None
        node = {
            'id': node_id,
            'title': title,
            'category': category,
            'kind': _str(fm.get('kind')),
            'type': note_type,
            'origin': _str(fm.get('origin')),
            'def': _str(fm.get('def')),
            'n_crit': len(crit) if crit else 0,
            'ghost': False,
            'file': rel,
        }
        if crit:
            node['crit'] = crit
        records.append((node, fm, stem))

    lookup = {}
    for index in (lambda n, s: s, lambda n, s: n['title'], lambda n, s: n['id']):
        for node, _, stem in records:
            k = _key(index(node, stem))
            if k and k not in lookup:
                lookup[k] = node['id']

    nodes = [node for node, _, _ in records]
    ghosts = {}
    links = []
    link_keys = set()

    def resolve(target):
        hit = lookup.get(_key(target))
        if hit is None and '/' in target:
            hit = lookup.get(_key(target.rsplit('/', 1)[1]))
        if hit is not None:
            return hit
        ghost_id = f"ghost:{target}"
        if ghost_id not in ghosts:
            ghosts[ghost_id] = {
                'id': ghost_id, 'title': target, 'category': '', 'kind': '', 'type': '',
                'origin': '', 'def': '', 'n_crit': 0, 'ghost': True,
            }
        return ghost_id

    def add(source, target, link_type):
        target_id = resolve(target)
        if target_id == source:
            return
        key = (source, target_id, link_type)
        if key in link_keys:
            return
        link_keys.add(key)
        links.append({'source': source, 'target': target_id, 'type': link_type})

    for node, fm, _ in records:
        for key in LINK_KEYS:
            for target in _wikilink_targets(fm.get(key)):
                add(node['id'], target, key)
        moaie = fm.get('moaie')
        if isinstance(moaie, dict):
            for facet in MOAIE_KEYS:
                for target in _wikilink_targets(moaie.get(facet)):
                    add(node['id'], target, f"moaie:{facet}")
        for key in ('derives_from', 'principle'):
            for target in _wikilink_targets(fm.get(key)):
                add(node['id'], target, key)

    nodes.extend(ghosts.values())
    counts = {
        'nodes': len(nodes),
        'real': len(records),
        'ghost': len(ghosts),
        'links': len(links),
        'by_category': {},
        'by_link_type': {},
    }
    for node, _, _ in records:
        cat = node['category'] or 'uncategorised'
        counts['by_category'][cat] = counts['by_category'].get(cat, 0) + 1
    for link in links:
        counts['by_link_type'][link['type']] = counts['by_link_type'].get(link['type'], 0) + 1

    graph = {
        'nodes': nodes,
        'links': links,
        'meta': {
            'source': 'github.com/zzigo/som-registry',
            'commit': signature.split('|', 1)[0],
            'built_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            'counts': counts,
            'skipped': skipped,
        },
    }
    with _cache_lock:
        _cache['signature'] = signature
        _cache['graph'] = graph
    return graph


def _json_safe(value):
    """YAML values (dates, sets, ...) as JSON-serialisable data."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def read_note(node_id, base=None):
    """
    Full text of one registry note: raw YAML, parsed frontmatter and markdown body.

    The file is looked up through the graph (node id -> relative path found by the
    registry walk), never from a caller-supplied path. Raises KeyError for unknown
    ids, pending (ghost) notes and files that resolve outside the checkout.
    """
    base = base or registry_dir()
    graph = build_registry_graph(base)
    node = next((n for n in graph['nodes'] if n['id'] == node_id and not n.get('ghost')), None)
    if node is None or not node.get('file'):
        raise KeyError(node_id)
    root = os.path.realpath(base)
    path = os.path.realpath(os.path.join(base, node['file']))
    if os.path.commonpath([root, path]) != root:
        raise KeyError(node_id)
    with open(path, 'r', encoding='utf-8') as fh:
        text = fh.read(MAX_NOTE_CHARS + 1)
    truncated = len(text) > MAX_NOTE_CHARS
    text = text[:MAX_NOTE_CHARS].lstrip('\ufeff')
    raw, body, parsed = '', text, None
    match = _FRONTMATTER_RE.match(text)
    if match:
        raw = match.group(1)
        body = text[match.end():]
        try:
            data = yaml.safe_load(raw)
            parsed = _json_safe(data) if isinstance(data, dict) else None
        except yaml.YAMLError:
            parsed = None
    return {
        'id': node['id'],
        'title': node['title'],
        'file': node['file'],
        'category': node['category'],
        'frontmatter_raw': raw,
        'frontmatter': parsed,
        'body': body,
        'truncated': truncated,
    }
