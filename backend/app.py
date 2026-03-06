from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
from logger import log_activity, get_logs, format_logs_html
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle, RegularPolygon
import numpy as np
from io import BytesIO
import base64
import os
import logging
import re
import sys
import torch
import scipy
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from torch import nn
from flask import send_from_directory
import json
import datetime
import importlib
import time
import hashlib

try:
    import trimesh  # for STL mode
except Exception:
    trimesh = None

try:
    from shapely.geometry import LineString as ShapelyLineString, Point as ShapelyPoint, Polygon as ShapelyPolygon
    from shapely.ops import unary_union as shapely_unary_union
except Exception:
    ShapelyLineString = None
    ShapelyPoint = None
    ShapelyPolygon = None
    shapely_unary_union = None

HF_CACHE_DIR = os.path.join(os.getcwd(), ".cache", "huggingface")
OFFLOAD_DIR = os.path.join(os.getcwd(), "offload")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR

# Gallery directory
GALLERY_DIR = os.path.join(OFFLOAD_DIR, 'gallery')
os.makedirs(GALLERY_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'api.log'))
    ]
)

load_dotenv()

# Runtime model override (changed via /api/ollama/model).
OLLAMA_MODEL_OVERRIDE = None
SOOPUB_CONTEXT_CACHE = {}
SOOPUB_DOCS_CACHE = None
SOOPUB_GRAPH_CACHE = {}
SOOPUB_SECTION_LABELS = {
    '1mat': 'materials',
    '2obj': 'objects',
    '3agn': 'agents',
    '4int': 'interfaces',
    '5pri': 'principles of functions',
    '6env': 'environments',
    '7dict': 'dictionary'
}

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/log": {"origins": "*"}
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_version():
    try:
        with open(os.path.join(os.path.dirname(__file__), 'version.txt'), 'r') as file:
            return file.read().strip()
    except Exception as e:
        logging.error(f"Error reading version file: {e}")
        return "0.0.0"

def get_prompt_content():
    try:
        with open(os.path.join(os.path.dirname(__file__), 'prompt.txt'), 'r') as file:
            return file.read().strip()
    except Exception as e:
        logging.error(f"Error reading prompt file: {e}")
        return "ERROR: Unable to load prompt content."


def get_ollama_config():
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').strip().rstrip('/')
    if not base_url:
        base_url = 'http://localhost:11434'

    model_name = (OLLAMA_MODEL_OVERRIDE or os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct')).strip()
    if not model_name:
        model_name = 'qwen2.5:7b-instruct'

    api_key = os.getenv('OLLAMA_API_KEY', '').strip()
    return base_url, model_name, api_key


def get_ollama_headers():
    _, _, api_key = get_ollama_config()
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    # Optional for reverse proxies that protect Ollama with bearer auth.
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
        headers['X-API-Key'] = api_key
    return headers


def _env_int(name: str, default: int, min_value: int = None, max_value: int = None) -> int:
    raw = os.getenv(name, '').strip()
    if not raw:
        value = int(default)
    else:
        try:
            value = int(raw)
        except ValueError:
            value = int(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, '').strip().lower()
    if not raw:
        return bool(default)
    return raw in ('1', 'true', 'yes', 'on')


def get_generation_limits():
    return {
        # LLM HTTP request timeout per call. Set to 0 to disable.
        'llm_timeout_sec': _env_int('OLLAMA_REQUEST_TIMEOUT_SEC', 90, min_value=0, max_value=86400),
        # Internal correction loops inside one generation.
        'max_attempts': _env_int('SOOG_LLM_MAX_ATTEMPTS', 2, min_value=1, max_value=6),
        # Absolute wall-clock budget for /api/generate.
        # Set to 0 to disable deadline (stress-testing mode).
        'deadline_sec': _env_int('SOOG_GENERATE_DEADLINE_SEC', 240, min_value=0, max_value=86400),
        # Optional second pass without soopub context (disabled by default to avoid long waits).
        'retry_without_context': _env_bool('SOOG_RETRY_WITHOUT_CONTEXT', default=False)
    }


def _read_context_excerpt(path: str, max_chars: int = 3000):
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except Exception:
        return ""
    if not content:
        return ""
    return content[:max_chars]


def _keyword_tokens(text: str):
    return set(re.findall(r"[a-zA-Z]{4,}", (text or "").lower()))


def _load_soopub_docs():
    global SOOPUB_DOCS_CACHE
    if SOOPUB_DOCS_CACHE is not None:
        return SOOPUB_DOCS_CACHE

    base = os.path.join(os.path.dirname(__file__), 'soopub')
    if not os.path.isdir(base):
        SOOPUB_DOCS_CACHE = []
        return SOOPUB_DOCS_CACHE

    docs = []
    for root, _, files in os.walk(base):
        for filename in files:
            if not filename.lower().endswith(('.md', '.txt')):
                continue
            fp = os.path.join(root, filename)
            excerpt = _read_context_excerpt(fp, max_chars=2200)
            if not excerpt:
                continue
            label = os.path.relpath(fp, os.path.dirname(__file__))
            docs.append({
                'label': label,
                'text': excerpt,
                'tokens': _keyword_tokens(excerpt)
            })
    SOOPUB_DOCS_CACHE = docs
    return SOOPUB_DOCS_CACHE


def get_soopub_context(query_text: str = "", max_docs: int = 4, max_chars: int = 5600):
    """
    Load compact contextual notes from soopub for stronger organological guidance.
    Retrieval keeps prompt.txt primary, adding only a small ranked context slice.
    """
    global SOOPUB_CONTEXT_CACHE
    cache_key = f"{(query_text or '').strip().lower()}|{max_docs}|{max_chars}"
    if cache_key in SOOPUB_CONTEXT_CACHE:
        return SOOPUB_CONTEXT_CACHE[cache_key]

    docs = _load_soopub_docs()
    if not docs:
        SOOPUB_CONTEXT_CACHE[cache_key] = ""
        return ""

    query_tokens = _keyword_tokens(query_text)
    preferred = {
        'soopub/7dict/mantle-hood organogram system.md',
        'soopub/4int/sound motion object.md',
        'soopub/5pri/acoustic.md',
        'soopub/1mat/material-acoustic-atlas.md',
        'soopub/2obj/platonic-geometry-acoustics.md'
    }

    def score(doc):
        overlap = len(doc['tokens'] & query_tokens) if query_tokens else 0
        preferred_bonus = 3 if doc['label'] in preferred else 0
        return overlap * 5 + preferred_bonus

    ranked = sorted(docs, key=score, reverse=True)
    selected = ranked[:max_docs]

    chunks = []
    used_chars = 0
    for doc in selected:
        block = f"[{doc['label']}]\n{doc['text']}".strip()
        if not block:
            continue
        if used_chars + len(block) > max_chars and chunks:
            break
        chunks.append(block)
        used_chars += len(block)

    context = "\n\n".join(chunks).strip()
    SOOPUB_CONTEXT_CACHE[cache_key] = context
    return context


def _soopub_base_dir():
    return os.path.join(os.path.dirname(__file__), 'soopub')


def _normalize_graph_token(value: str) -> str:
    token = (value or '').strip().lower().replace('\\', '/')
    token = re.sub(r'\.[a-z0-9]+$', '', token)
    token = re.sub(r'[/\s]+', ' ', token)
    token = re.sub(r'[^a-z0-9 _\-/]+', '', token)
    token = re.sub(r'\s+', ' ', token).strip()
    return token


def _split_frontmatter(content: str):
    if not content or not content.startswith('---'):
        return "", content
    lines = content.splitlines()
    if not lines or lines[0].strip() != '---':
        return "", content
    for idx in range(1, len(lines)):
        if lines[idx].strip() == '---':
            frontmatter = "\n".join(lines[1:idx])
            body = "\n".join(lines[idx + 1:])
            return frontmatter, body
    return "", content


def _strip_yaml_scalar(value: str) -> str:
    text = (value or '').strip().strip('"\'')
    if text.startswith('[[') and text.endswith(']]'):
        text = text[2:-2]
    if text.startswith('#'):
        text = text[1:]
    return text.strip()


def _parse_frontmatter_tags(frontmatter: str):
    if not frontmatter:
        return []
    tags = []
    seen = set()
    lines = frontmatter.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r'^\s*tags\s*:\s*(.*)$', line, flags=re.IGNORECASE)
        if not match:
            i += 1
            continue

        remainder = (match.group(1) or '').strip()
        if remainder:
            remainder = remainder.strip('[]')
            for token in re.split(r'[,\s]+', remainder):
                tag = _strip_yaml_scalar(token)
                normalized = _normalize_graph_token(tag)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    tags.append(normalized)
            i += 1
            continue

        i += 1
        while i < len(lines):
            row = lines[i]
            item = re.match(r'^\s*-\s*(.+)$', row)
            if item:
                tag = _strip_yaml_scalar(item.group(1))
                normalized = _normalize_graph_token(tag)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    tags.append(normalized)
                i += 1
                continue
            if not row.strip():
                i += 1
                continue
            break
    return tags


def _extract_inline_tags(body: str):
    tags = []
    seen = set()
    for line in (body or '').splitlines():
        if re.match(r'^\s{0,3}#{1,6}\s', line):
            continue
        for match in re.finditer(r'(?<![\w/])#([a-zA-Z0-9][a-zA-Z0-9_/\-]{1,63})', line):
            tag = _normalize_graph_token(match.group(1))
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
    return tags


def _extract_wikilinks(content: str):
    links = []
    seen = set()
    for raw in re.findall(r'\[\[([^\]]+)\]\]', content or ''):
        cleaned = raw.strip()
        if not cleaned:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            links.append(cleaned)
    return links


def _normalize_wikilink_target(value: str) -> str:
    target = (value or '').strip()
    if not target:
        return ''
    target = target.split('|', 1)[0].split('#', 1)[0].strip()
    target = target.replace('\\', '/')
    target = re.sub(r'\.md$', '', target, flags=re.IGNORECASE)
    return target.strip()


def _soopub_files_signature(base_dir: str):
    parts = []
    for root, _, files in os.walk(base_dir):
        for filename in sorted(files):
            if filename.startswith('.'):
                continue
            if not filename.lower().endswith(('.md', '.txt')):
                continue
            fp = os.path.join(root, filename)
            try:
                stat = os.stat(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, base_dir).replace('\\', '/')
            parts.append(f"{rel}:{int(stat.st_mtime)}:{stat.st_size}")
    return "|".join(parts)


def _build_soopub_graph():
    base_dir = _soopub_base_dir()
    if not os.path.isdir(base_dir):
        return {'nodes': [], 'links': [], 'stats': {'nodes': 0, 'links': 0}}

    signature = _soopub_files_signature(base_dir)
    cached = SOOPUB_GRAPH_CACHE.get('snapshot')
    if cached and cached.get('signature') == signature:
        return cached.get('graph')

    nodes = {}
    links = []
    link_keys = set()
    note_records = []

    def add_node(payload: dict):
        node_id = payload.get('id')
        if not node_id:
            return
        if node_id in nodes:
            return
        nodes[node_id] = payload

    def add_link(source: str, target: str, edge_type: str):
        if not source or not target or source == target:
            return
        key = (source, target, edge_type)
        if key in link_keys:
            return
        link_keys.add(key)
        links.append({
            'source': source,
            'target': target,
            'type': edge_type
        })

    root_folder_id = 'folder:soopub'
    add_node({
        'id': root_folder_id,
        'type': 'folder',
        'title': 'soopub',
        'path': 'soopub',
        'section_key': 'root',
        'section_label': 'vault',
        'depth': 0
    })

    for root, dirs, files in os.walk(base_dir):
        dirs[:] = sorted([d for d in dirs if not d.startswith('.')])
        files = sorted(files)

        rel_dir = os.path.relpath(root, base_dir).replace('\\', '/')
        rel_dir = '' if rel_dir == '.' else rel_dir

        folder_id = root_folder_id if not rel_dir else f"folder:{rel_dir}"
        folder_name = 'soopub' if not rel_dir else os.path.basename(root)
        folder_depth = 0 if not rel_dir else rel_dir.count('/') + 1
        top_section = rel_dir.split('/')[0] if rel_dir else 'root'
        top_label = SOOPUB_SECTION_LABELS.get(top_section, top_section)

        folder_title = folder_name
        if rel_dir and rel_dir.count('/') == 0 and top_section in SOOPUB_SECTION_LABELS:
            folder_title = f"{folder_name} ({top_label})"

        add_node({
            'id': folder_id,
            'type': 'folder',
            'title': folder_title,
            'path': rel_dir or 'soopub',
            'section_key': top_section,
            'section_label': top_label,
            'depth': folder_depth
        })

        if rel_dir:
            parent_rel = os.path.dirname(rel_dir).replace('\\', '/')
            parent_id = root_folder_id if not parent_rel else f"folder:{parent_rel}"
            add_link(parent_id, folder_id, 'folder_contains')

        for filename in files:
            if filename.startswith('.'):
                continue
            if not filename.lower().endswith(('.md', '.txt')):
                continue

            fp = os.path.join(root, filename)
            rel_file = os.path.relpath(fp, base_dir).replace('\\', '/')
            rel_no_ext = re.sub(r'\.[^.]+$', '', rel_file)
            title = os.path.splitext(os.path.basename(filename))[0].strip()
            note_id = f"note:{rel_no_ext}"
            section_key = rel_file.split('/')[0] if '/' in rel_file else 'root'
            section_label = SOOPUB_SECTION_LABELS.get(section_key, section_key)

            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                continue

            frontmatter, body = _split_frontmatter(content)
            preview = re.sub(r'\s+', ' ', (body or '').strip())[:360]
            tags = []
            tag_seen = set()
            for tag in _parse_frontmatter_tags(frontmatter) + _extract_inline_tags(body):
                normalized = _normalize_graph_token(tag)
                if normalized and normalized not in tag_seen:
                    tag_seen.add(normalized)
                    tags.append(normalized)
            wikilinks = _extract_wikilinks(content)

            add_node({
                'id': note_id,
                'type': 'note',
                'title': title or rel_no_ext,
                'path': rel_file,
                'section_key': section_key,
                'section_label': section_label,
                'folder_id': folder_id,
                'preview': preview,
                'tag_count': len(tags)
            })
            add_link(folder_id, note_id, 'folder_contains')

            note_records.append({
                'id': note_id,
                'title': title,
                'path_no_ext': rel_no_ext,
                'wikilinks': wikilinks,
                'tags': tags
            })

    note_lookup = {}
    for record in note_records:
        keys = {
            _normalize_graph_token(record['title']),
            _normalize_graph_token(record['path_no_ext']),
            _normalize_graph_token(os.path.basename(record['path_no_ext'])),
            _normalize_graph_token(record['path_no_ext'].replace('/', ' '))
        }
        for key in keys:
            if key and key not in note_lookup:
                note_lookup[key] = record['id']

    for record in note_records:
        note_id = record['id']

        for tag in record['tags']:
            tag_id = f"tag:{tag}"
            tag_leaf = tag.split('/')[-1]
            add_node({
                'id': tag_id,
                'type': 'tag',
                'title': f"#{tag_leaf}",
                'path': tag,
                'section_key': 'tag',
                'section_label': 'tags'
            })
            add_link(note_id, tag_id, 'tagged')

        for raw_target in record['wikilinks']:
            target = _normalize_wikilink_target(raw_target)
            if not target:
                continue

            target_key = _normalize_graph_token(target)
            target_basename = _normalize_graph_token(os.path.basename(target))
            target_id = note_lookup.get(target_key) or note_lookup.get(target_basename)

            if target_id and target_id != note_id:
                add_link(note_id, target_id, 'wikilink')
                continue

            ghost_key = _normalize_graph_token(target)
            if not ghost_key:
                continue
            ghost_id = f"ghost:{ghost_key}"
            add_node({
                'id': ghost_id,
                'type': 'ghost',
                'title': target,
                'path': target,
                'section_key': 'reference',
                'section_label': 'external references'
            })
            add_link(note_id, ghost_id, 'wikilink_unresolved')

    degree = {node_id: 0 for node_id in nodes.keys()}
    for edge in links:
        source = edge.get('source')
        target = edge.get('target')
        if source in degree:
            degree[source] += 1
        if target in degree:
            degree[target] += 1

    for node in nodes.values():
        node_degree = degree.get(node['id'], 0)
        node['degree'] = node_degree
        if node.get('type') == 'folder':
            node['val'] = 3 + min(node_degree, 20) * 0.18
        elif node.get('type') == 'note':
            node['val'] = 2.4 + min(node_degree, 24) * 0.16
        elif node.get('type') == 'tag':
            node['val'] = 1.8 + min(node_degree, 18) * 0.14
        else:
            node['val'] = 1.6 + min(node_degree, 12) * 0.12

    graph = {
        'nodes': list(nodes.values()),
        'links': links,
        'stats': {
            'nodes': len(nodes),
            'links': len(links),
            'notes': len([n for n in nodes.values() if n.get('type') == 'note']),
            'folders': len([n for n in nodes.values() if n.get('type') == 'folder']),
            'tags': len([n for n in nodes.values() if n.get('type') == 'tag']),
            'ghosts': len([n for n in nodes.values() if n.get('type') == 'ghost'])
        },
        'section_labels': SOOPUB_SECTION_LABELS
    }
    SOOPUB_GRAPH_CACHE['snapshot'] = {'signature': signature, 'graph': graph}
    return graph


def _repair_matplotlib_imports(code: str) -> str:
    lines = code.splitlines()
    fixed = []
    needs_line2d = False

    for line in lines:
        stripped = line.strip()
        match = re.match(r'from\s+matplotlib\.patches\s+import\s+(.+)$', stripped)
        if not match:
            fixed.append(line)
            continue

        names = [name.strip() for name in match.group(1).split(',') if name.strip()]
        if 'Line2D' not in names:
            fixed.append(line)
            continue

        names = [name for name in names if name != 'Line2D']
        needs_line2d = True
        indent = line[:len(line) - len(line.lstrip())]
        if names:
            fixed.append(f"{indent}from matplotlib.patches import {', '.join(names)}")

    patched = "\n".join(fixed)
    if needs_line2d and not re.search(r'from\s+matplotlib\.lines\s+import\s+Line2D', patched):
        patched = f"from matplotlib.lines import Line2D\n{patched}"
    return patched


def _safe_slug(text: str, fallback: str = "item") -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', (text or "").lower()).strip('_')
    return slug or fallback


def _version_index_from_value(value, default: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, float):
        return max(1, int(round(value)))
    text = str(value).strip().lower()
    if not text:
        return default
    if text.startswith('v'):
        text = text[1:]
    plain = re.match(r'^(\d+)$', text)
    if plain:
        return max(1, int(plain.group(1)))
    legacy = re.match(r'^(\d+)\.(\d+)$', text)
    if legacy:
        major = int(legacy.group(1))
        minor = int(legacy.group(2))
        # Legacy v1.0/v1.1/v1.2... maps to 1/2/3...
        if major == 1:
            return max(1, minor + 1)
        return max(1, major * 10 + minor)
    return default


def _version_label_from_index(version_index: int) -> str:
    return f"v{max(1, int(version_index or 1))}"


def _title_from_basename(basename: str) -> str:
    if not basename:
        return "untitled"
    parts = basename.split('_')
    if len(parts) <= 1:
        return basename
    # drop timestamp prefix
    return parts[1].replace('_', '-')


def _basename_body(basename: str) -> str:
    value = str(basename or '').strip()
    if not value:
        return ''
    return re.sub(r'^\d{8}-\d{6}_', '', value)


def _infer_group_id_from_basename(basename: str) -> str:
    body = _basename_body(basename)
    if not body:
        return ''
    # Match both v1 and legacy v1_0 / v1_0_1 suffixes.
    stripped = re.sub(r'_v\d+(?:_\d+)?(?:_\d+)?$', '', body, flags=re.IGNORECASE)
    return stripped or body


def _meta_group_id(meta: dict) -> str:
    if not isinstance(meta, dict):
        return ''
    explicit = str(meta.get('group_id') or '').strip()
    if explicit:
        return explicit
    return _infer_group_id_from_basename(str(meta.get('basename') or '').strip())


def parse_refact_marker(prompt_text: str):
    """
    Parse optional first-line refactor markers:
    - [REFACT key=value key2=value2]
    - * ... (short refactor mode)
    Returns: (clean_prompt, meta_dict)
    """
    text = (prompt_text or "").strip()
    meta = {'is_refact': False}
    if not text:
        return "", meta

    lines = text.splitlines()
    first = lines[0].strip()

    if first.startswith('[REFACT'):
        meta['is_refact'] = True
        payload = first[len('[REFACT'):].rstrip(']').strip()
        for token in re.split(r'\s+', payload):
            if '=' not in token:
                continue
            key, val = token.split('=', 1)
            key = key.strip().lower()
            val = val.strip().strip('"\'')
            if key:
                meta[key] = val
        clean = "\n".join(lines[1:]).lstrip()
        return clean, meta

    if first.startswith('*') or first.startswith('+'):
        meta['is_refact'] = True
        clean_first = first[1:].strip()
        rest = "\n".join(lines[1:]).strip()
        clean = f"{clean_first}\n{rest}".strip()
        return clean, meta

    return text, meta


def _iter_gallery_metas():
    if not os.path.isdir(GALLERY_DIR):
        return []
    metas = []
    for name in os.listdir(GALLERY_DIR):
        if not name.endswith('.json'):
            continue
        path = os.path.join(GALLERY_DIR, name)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                metas.append(json.load(f))
        except Exception:
            continue
    return metas


def _group_defaults(group_id: str):
    title = None
    title_slug = None
    max_version_index = 0
    for meta in _iter_gallery_metas():
        if _meta_group_id(meta) != group_id:
            continue
        if title is None:
            title = meta.get('title')
        if title_slug is None:
            title_slug = meta.get('title_slug')
        max_version_index = max(
            max_version_index,
            _version_index_from_value(meta.get('version_index'), default=0)
        )
    return {'title': title, 'title_slug': title_slug, 'max_version_index': max_version_index}


def _next_group_version_index(group_id: str, seed_version_index: int = 1) -> int:
    group_items = [meta for meta in _iter_gallery_metas() if _meta_group_id(meta) == group_id]
    if not group_items:
        return max(1, seed_version_index)

    # Prefer explicit integer versions (v1, v2, v3).
    max_plain = 0
    for meta in group_items:
        version_text = str(meta.get('version') or '').strip().lower()
        match = re.match(r'^v?(\d+)$', version_text)
        if match:
            max_plain = max(max_plain, int(match.group(1)))
            continue
        idx_value = meta.get('version_index')
        if isinstance(idx_value, int) and 1 <= idx_value < 10:
            max_plain = max(max_plain, idx_value)

    if max_plain > 0:
        return max(max_plain + 1, max(1, seed_version_index))

    # Legacy v1.0/v1.1 data: continue with simple ordinal count.
    return len(group_items) + 1


def extract_llm_content(payload: dict) -> str:
    # OpenAI-compatible format
    choices = payload.get('choices')
    if isinstance(choices, list) and choices:
        message = choices[0].get('message', {})
        content = message.get('content')
        if isinstance(content, list):
            text_parts = [
                part.get('text', '')
                for part in content
                if isinstance(part, dict)
            ]
            text = ''.join(text_parts).strip()
            if text:
                return text
        elif content is not None:
            return str(content).strip()

    # Native Ollama /api/chat format
    message = payload.get('message')
    if isinstance(message, dict) and message.get('content') is not None:
        return str(message.get('content')).strip()

    # Native Ollama /api/generate format
    if payload.get('response') is not None:
        return str(payload.get('response')).strip()

    raise ValueError('Response does not contain text content')


def call_ollama_chat(system_prompt: str, user_prompt: str, timeout=90):
    base_url, model_name, _ = get_ollama_config()
    headers = get_ollama_headers()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    openai_body = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.9,
        "max_tokens": 1000
    }
    native_body = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.9,
            "num_predict": 1000
        }
    }
    generate_body = {
        "model": model_name,
        "prompt": f"System:\n{system_prompt}\n\nUser:\n{user_prompt}",
        "stream": False,
        "options": {
            "temperature": 0.9,
            "num_predict": 1000
        }
    }

    attempts = [
        (f"{base_url}/api/chat", native_body),
        (f"{base_url}/v1/chat/completions", openai_body),
        (f"{base_url}/api/generate", generate_body),
    ]
    last_error = None

    for url, body in attempts:
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)
        except requests.exceptions.RequestException as req_err:
            last_error = f"Request failed for {url}: {req_err}"
            continue

        if response.status_code >= 400:
            if response.status_code in (400, 404, 405, 422):
                last_error = f"Endpoint {url} returned {response.status_code}: {response.text[:300]}"
                continue
            if response.status_code == 401:
                raise RuntimeError(
                    f"Ollama unauthorized at {url}. Check OLLAMA_API_KEY or proxy auth."
                )
            body_excerpt = response.text[:600]
            if 'requires more system memory' in body_excerpt.lower():
                raise RuntimeError(
                    f"Ollama model OOM at {url}: {body_excerpt}. "
                    "Try a smaller model or free RAM/swap."
                )
            raise RuntimeError(f"Ollama API error ({response.status_code}) at {url}: {response.text[:600]}")

        try:
            payload = response.json()
        except Exception as json_err:
            last_error = f"Invalid JSON from {url}: {json_err}"
            continue

        try:
            content = extract_llm_content(payload)
        except Exception as extract_err:
            last_error = f"Could not extract content from {url}: {extract_err}"
            continue
        if not content:
            last_error = f"Empty model response from {url}"
            continue
        return content, {
            'endpoint': url,
            'base_url': base_url,
            'model': model_name
        }

    raise RuntimeError(last_error or "Could not connect to Ollama endpoints")


def extract_python_code_blocks(raw_text: str):
    # First pass: closed fenced python blocks
    code_blocks = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", raw_text, re.DOTALL | re.IGNORECASE)

    # Fallback: detect opening fence without closure and grab to end
    if not code_blocks:
        m = re.search(r"```(?:python|py)?\s*([\s\S]*)$", raw_text, re.IGNORECASE)
        if m:
            code_blocks = [m.group(1)]

    # Last resort: heuristic when model emits bare code
    if not code_blocks and (('import matplotlib' in raw_text) or ('plt.' in raw_text)):
        start_idx = raw_text.find('import matplotlib')
        if start_idx == -1:
            start_idx = raw_text.find('plt.')
        if start_idx != -1:
            code_blocks = [raw_text[start_idx:]]

    return [c.strip() for c in code_blocks if c and c.strip()]


def split_plot_and_stl_code(code_blocks):
    plot_code = None
    stl_code = None
    for code in code_blocks:
        if plot_code is None and ('matplotlib' in code or 'plt.' in code):
            plot_code = code
        if stl_code is None and ('trimesh' in code or 'import trimesh' in code):
            stl_code = code
    return plot_code, stl_code


def split_summary_and_materials(text_no_code: str):
    """
    Split prose into conceptual summary and materials section.
    Supports headings like:
    - Materials
    - ## Materials
    - **Materials**
    - Bill of Materials / BOM
    """
    if not text_no_code:
        return "", None

    heading = re.search(
        r"(?im)^\s*(?:#+\s*)?(?:\*\*|__)?\s*(materials|bill of materials|bom)\s*(?:\*\*|__)?\s*:?\s*$",
        text_no_code
    )
    if heading:
        summary = text_no_code[:heading.start()].strip()
        materials = text_no_code[heading.start():].strip()
        return summary, materials if materials else None

    # Fallback for inline "Materials:" mentions
    inline = re.search(r"(?is)\b(materials|bill of materials|bom)\s*:\s*([\s\S]+)$", text_no_code)
    if inline:
        summary = text_no_code[:inline.start()].strip()
        materials = inline.group(0).strip()
        return summary, materials if materials else None

    return text_no_code.strip(), None


def fetch_ollama_models():
    """Return installed Ollama models and endpoint metadata."""
    base_url, configured_model, _ = get_ollama_config()
    headers = get_ollama_headers()
    last_status = None
    last_error = None

    for path in ('/api/tags', '/v1/models'):
        url = f"{base_url}{path}"
        try:
            r = requests.get(url, headers=headers, timeout=30)
        except requests.exceptions.RequestException as req_err:
            last_error = str(req_err)
            continue

        last_status = r.status_code
        if r.status_code == 200:
            payload = r.json()
            models = []

            # Native Ollama payload: {"models": [{"name": "..."}]}
            if isinstance(payload, dict) and isinstance(payload.get('models'), list):
                for model_item in payload.get('models', []):
                    if isinstance(model_item, dict):
                        name = model_item.get('name') or model_item.get('model')
                        if name:
                            models.append(str(name))
                    elif isinstance(model_item, str):
                        models.append(model_item)

            # OpenAI-style payload: {"data": [{"id": "..."}]}
            if not models and isinstance(payload, dict) and isinstance(payload.get('data'), list):
                for model_item in payload.get('data', []):
                    if isinstance(model_item, dict):
                        model_id = model_item.get('id') or model_item.get('name')
                        if model_id:
                            models.append(str(model_id))
                    elif isinstance(model_item, str):
                        models.append(model_item)

            return {
                'ok': True,
                'endpoint': url,
                'base_url': base_url,
                'configured_model': configured_model,
                'models': models,
                'model_count': len(models)
            }

        if r.status_code == 401:
            return {'ok': False, 'status': 401, 'error': 'Unauthorized', 'base_url': base_url}

    if last_error:
        return {'ok': False, 'status': 502, 'error': f'Unable to connect to Ollama at {base_url}: {last_error}', 'base_url': base_url}
    return {'ok': False, 'status': last_status or 500, 'error': 'Could not list Ollama models', 'base_url': base_url}


def generate_with_image_required(
    prompt_content: str,
    user_instruction: str,
    max_attempts: int = 2,
    llm_timeout_sec: int = 90,
    deadline_at: float = None
):
    """
    Run up to N LLM passes and require a valid matplotlib-rendered image.
    Returns: raw_response, summary_text, materials_text, plot_code, stl_code, image_base64, stl_bytes, llm_meta
    """
    last_reason = "No valid matplotlib output produced."
    for attempt_idx in range(1, max_attempts + 1):
        if deadline_at is not None:
            remaining = deadline_at - time.perf_counter()
            if remaining <= 8:
                raise RuntimeError(
                    f"Generation deadline exceeded after {attempt_idx - 1} attempt(s). "
                    f"Last reason: {last_reason}"
                )

        if attempt_idx == 1:
            attempt_prompt = user_instruction
        else:
            attempt_prompt = (
                user_instruction
                + "\n\nCorrection required:"
                + f"\n- Previous attempt failed with: {last_reason}"
                + "\n- Fix both matplotlib and trimesh code so they execute without errors."
                + "\n- Keep matplotlib code as the first Python block and trimesh code as the second block."
                + "\n- The trimesh block must define `mesh` as a non-empty trimesh.Trimesh."
                + "\n- Do not include placeholder text or recovery notes."
            )
        call_timeout = int(llm_timeout_sec) if int(llm_timeout_sec) > 0 else None
        if deadline_at is not None:
            # Keep margin for code execution/parsing after LLM response.
            remaining = max(1, int(deadline_at - time.perf_counter()))
            deadline_timeout = max(12, remaining - 5)
            if call_timeout is None:
                call_timeout = deadline_timeout
            else:
                call_timeout = max(12, min(call_timeout, deadline_timeout))

        try:
            raw_response, llm_meta = call_ollama_chat(prompt_content, attempt_prompt, timeout=call_timeout)
        except Exception as e:
            last_reason = f"LLM call failed: {e}"
            logging.error(last_reason)
            if 'requires more system memory' in str(e).lower():
                raise RuntimeError(last_reason)
            continue
        logging.info(
            f"LLM generate attempt={attempt_idx} endpoint={llm_meta.get('endpoint')} model={llm_meta.get('model')}"
        )

        text_no_code = re.sub(r"```[\s\S]*?```", "", raw_response).strip()
        summary_text, materials_text = split_summary_and_materials(text_no_code)

        code_blocks = extract_python_code_blocks(raw_response)
        plot_code, stl_code = split_plot_and_stl_code(code_blocks)

        if not plot_code:
            last_reason = "Model response did not include matplotlib code."
            continue

        try:
            image_base64 = execute_matplotlib_code(plot_code)
            image_bytes = base64.b64decode(image_base64)
            if not image_bytes:
                last_reason = "Matplotlib code executed but produced empty image bytes."
                continue
        except Exception as e:
            last_reason = f"Matplotlib execution failed: {e}"
            logging.error(last_reason)
            continue

        stl_bytes = None
        if stl_code:
            try:
                stl_bytes = execute_trimesh_code(stl_code)
            except Exception as e:
                last_reason = f"Trimesh execution failed: {e}"
                logging.error(last_reason)
                if attempt_idx < max_attempts:
                    continue
                stl_bytes = None
        else:
            last_reason = "Model response did not include trimesh code block."
            if attempt_idx < max_attempts:
                continue

        return (
            raw_response,
            summary_text,
            materials_text,
            plot_code,
            stl_code,
            image_base64,
            stl_bytes,
            llm_meta
        )

    raise RuntimeError(last_reason)

def execute_matplotlib_code(code: str) -> str:
    """Execute matplotlib code safely and return the PNG as base64.
    - Strips code fences
    - Removes plt.show()
    - Ensures trailing newline
    - Balances common brackets if SyntaxError occurs
    """
    # Normalize and sanitize code first
    cleaned = code.strip()
    # Strip stray code fences if present
    cleaned = re.sub(r'^```(?:python|py)?', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    # Remove plt.show() to avoid blocking
    cleaned = re.sub(r'plt\.show\(\s*\)', '', cleaned)
    cleaned = _repair_matplotlib_imports(cleaned)
    # Ensure trailing newline
    if not cleaned.endswith('\n'):
        cleaned += '\n'

    def balance_brackets(s: str) -> str:
        pairs = [('(', ')'), ('[', ']'), ('{', '}')]
        for open_c, close_c in pairs:
            missing = s.count(open_c) - s.count(close_c)
            if missing > 0:
                s += close_c * missing + '\n'
        return s

    code_to_run = cleaned
    # Try pre-compilation; on SyntaxError, retry with balanced code
    try:
        compile(code_to_run, '<string>', 'exec')
    except SyntaxError:
        code_to_run = balance_brackets(cleaned)
        compile(code_to_run, '<string>', 'exec')

    # Execute and capture the current figure
    try:
        plt.close('all')

        # LLM-generated code often references patch classes through plt.<Class> or without imports.
        # These aliases/locals improve execution robustness without mutating the generated logic.
        def _safe_arrow(x, y, dx, dy, *args, **kwargs):
            color = kwargs.pop('color', kwargs.pop('fc', kwargs.pop('facecolor', 'white')))
            linewidth = kwargs.pop('linewidth', kwargs.pop('lw', 1.5))
            alpha = kwargs.pop('alpha', 1.0)
            arrowstyle = kwargs.pop('arrowstyle', '-|>')
            head_width = kwargs.pop('head_width', 0.25)
            mutation_scale = kwargs.pop('mutation_scale', max(8.0, float(head_width) * 20.0))
            return FancyArrowPatch(
                (x, y),
                (x + dx, y + dy),
                arrowstyle=arrowstyle,
                mutation_scale=mutation_scale,
                color=color,
                linewidth=linewidth,
                alpha=alpha
            )

        def _safe_fancy_arrow_patch(*args, **kwargs):
            # Accept both FancyArrowPatch(posA, posB, ...) and Arrow-like (x, y, dx, dy, ...) signatures.
            if len(args) >= 4 and all(isinstance(v, (int, float)) for v in args[:4]):
                return _safe_arrow(*args, **kwargs)
            kwargs.pop('head_width', None)
            kwargs.pop('head_length', None)
            kwargs.pop('width', None)
            return FancyArrowPatch(*args, **kwargs)

        if not hasattr(Polygon, 'copy'):
            def _polygon_copy(self):
                return Polygon(
                    self.get_xy(),
                    closed=getattr(self, '_closed', True),
                    facecolor=self.get_facecolor(),
                    edgecolor=self.get_edgecolor(),
                    linewidth=self.get_linewidth(),
                    alpha=self.get_alpha()
                )
            try:
                Polygon.copy = _polygon_copy
            except Exception:
                pass

        setattr(plt, 'Arc', Arc)
        setattr(plt, 'Ellipse', Ellipse)
        setattr(plt, 'Polygon', Polygon)
        setattr(plt, 'Circle', Circle)
        setattr(plt, 'Rectangle', Rectangle)
        setattr(plt, 'RegularPolygon', RegularPolygon)
        setattr(plt, 'Arrow', _safe_arrow)
        setattr(plt, 'FancyArrowPatch', _safe_fancy_arrow_patch)

        safe_globals = {
            '__builtins__': __builtins__,
            'matplotlib': matplotlib,
            'np': np,
            'plt': plt,
            'Arc': Arc,
            'Circle': Circle,
            'Ellipse': Ellipse,
            'FancyArrowPatch': _safe_fancy_arrow_patch,
            'Line2D': Line2D,
            'Polygon': Polygon,
            'Rectangle': Rectangle,
            'RegularPolygon': RegularPolygon,
        }
        exec(code_to_run, safe_globals, safe_globals)

        fig = plt.gcf()
        if not fig.axes:
            raise RuntimeError("Matplotlib code executed but produced no axes")
        has_visible_content = any(
            ax.has_data() or ax.patches or ax.lines or ax.collections or ax.images or ax.texts
            for ax in fig.axes
        )
        if not has_visible_content:
            raise RuntimeError("Matplotlib code executed but produced an empty figure")

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error executing matplotlib code: {e}")
        raise


def _compile_with_truncation(code: str, filename: str):
    """Compile code; if trailing lines are broken, truncate until a valid prefix is found."""
    try:
        compile(code, filename, 'exec')
        return code
    except SyntaxError:
        lines = code.splitlines()
        # Keep at least a few lines; drop from the end to recover from partial model output.
        for keep in range(len(lines) - 1, 3, -1):
            candidate = "\n".join(lines[:keep]).strip()
            if not candidate:
                continue
            candidate += "\n"
            try:
                compile(candidate, filename, 'exec')
                return candidate
            except SyntaxError:
                continue
        raise


def _stable_unit(seed_text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}|{seed_text}".encode('utf-8', errors='ignore')).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _infer_stl_profile(prompt: str = "", summary_text: str = "", materials_text: str = "", plot_code: str = "") -> str:
    text = " ".join([prompt or "", summary_text or "", materials_text or "", plot_code or ""]).lower()
    profile_keywords = {
        'aerophone': ['aerophone', 'flute', 'pipe', 'reed', 'organ', 'wind', 'air column', 'mouthpiece'],
        'chordophone': ['chordophone', 'string', 'guitar', 'violin', 'harp', 'lute', 'bowed', 'plucked'],
        'membranophone': ['membranophone', 'drum', 'membrane', 'skin', 'percussion'],
        'idiophone': ['idiophone', 'plate', 'bell', 'gong', 'bar', 'chime', 'xylophone'],
        'electro': ['electro', 'speaker', 'microphone', 'sensor', 'dmi', 'hybrid', 'digital', 'interface']
    }
    scores = {name: 0 for name in profile_keywords}
    for name, words in profile_keywords.items():
        for w in words:
            if w in text:
                scores[name] += 1

    # Extra cues from organogram code.
    lowered_plot = (plot_code or "").lower()
    if 'circle(' in lowered_plot:
        scores['aerophone'] += 1
    if 'rectangle(' in lowered_plot:
        scores['chordophone'] += 1
    if 'polygon(' in lowered_plot:
        scores['idiophone'] += 1

    winner = max(scores, key=scores.get)
    return winner if scores[winner] > 0 else 'hybrid'


def _build_contextual_fallback_stl_code(prompt: str = "", summary_text: str = "", materials_text: str = "", plot_code: str = ""):
    seed_text = " ".join([prompt or "", summary_text or "", materials_text or "", plot_code or ""]).strip() or "soog"
    profile = _infer_stl_profile(prompt, summary_text, materials_text, plot_code)

    u1 = _stable_unit(seed_text, 'u1')
    u2 = _stable_unit(seed_text, 'u2')
    u3 = _stable_unit(seed_text, 'u3')
    u4 = _stable_unit(seed_text, 'u4')

    sections = 28 + int(u1 * 36)  # 28..64
    core_length = round(130 + u2 * 170, 2)  # 130..300
    core_radius = round(8 + u3 * 16, 2)  # 8..24
    aux_scale = round(0.35 + u4 * 0.55, 3)  # 0.35..0.9

    if profile == 'aerophone':
        bell_height = round(core_length * (0.18 + aux_scale * 0.15), 2)
        bell_radius = round(core_radius * (1.15 + aux_scale * 0.6), 2)
        mouth_height = round(core_length * (0.12 + aux_scale * 0.08), 2)
        mouth_radius = round(core_radius * (0.42 + aux_scale * 0.18), 2)
        return f"""import numpy as np
import trimesh

# map: air-column -> cylinder, bell radiation -> cone, mouthpiece interface -> cylinder
body = trimesh.creation.cylinder(radius={core_radius}, height={core_length}, sections={sections})
bell = trimesh.creation.cone(radius={bell_radius}, height={bell_height}, sections={sections})
bell.apply_translation((0, 0, -{round(core_length / 2 + bell_height / 2, 2)}))
mouth = trimesh.creation.cylinder(radius={mouth_radius}, height={mouth_height}, sections={sections})
mouth.apply_translation((0, 0, {round(core_length / 2 + mouth_height / 2, 2)}))
spine = trimesh.creation.box(extents=({round(core_radius * 0.8, 2)}, {round(core_radius * 1.1, 2)}, {round(core_length * 0.4, 2)}))
mesh = trimesh.util.concatenate([body, bell, mouth, spine])
""", profile

    if profile == 'chordophone':
        body_x = round(core_length * 0.9, 2)
        body_y = round(core_radius * 5.5, 2)
        body_z = round(core_radius * 1.35, 2)
        neck_x = round(core_length * 0.75, 2)
        neck_y = round(core_radius * 1.25, 2)
        neck_z = round(core_radius * 0.85, 2)
        return f"""import numpy as np
import trimesh

# map: resonator plate -> box, neck interface -> box, bridge coupling -> bar
resonator = trimesh.creation.box(extents=({body_x}, {body_y}, {body_z}))
neck = trimesh.creation.box(extents=({neck_x}, {neck_y}, {neck_z}))
neck.apply_translation(({round(body_x * 0.62, 2)}, 0, {round(body_z * 0.45, 2)}))
bridge = trimesh.creation.box(extents=({round(body_x * 0.18, 2)}, {round(body_y * 0.16, 2)}, {round(body_z * 0.42, 2)}))
bridge.apply_translation(({round(-body_x * 0.1, 2)}, 0, {round(body_z * 0.72, 2)}))
peg = trimesh.creation.cylinder(radius={round(core_radius * 0.22, 2)}, height={round(body_y * 0.75, 2)}, sections={sections})
peg.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
peg.apply_translation(({round(body_x * 0.94, 2)}, 0, {round(body_z * 0.6, 2)}))
mesh = trimesh.util.concatenate([resonator, neck, bridge, peg])
""", profile

    if profile == 'membranophone':
        shell_h = round(core_length * 0.55, 2)
        shell_r = round(core_radius * 1.35, 2)
        ring_h = round(max(2.0, shell_h * 0.045), 2)
        return f"""import numpy as np
import trimesh

# map: shell cavity -> cylinder, membrane rims -> rings, support feet -> boxes
shell = trimesh.creation.cylinder(radius={shell_r}, height={shell_h}, sections={sections})
rim_top = trimesh.creation.cylinder(radius={round(shell_r * 1.06, 2)}, height={ring_h}, sections={sections})
rim_top.apply_translation((0, 0, {round(shell_h * 0.5, 2)}))
rim_bottom = trimesh.creation.cylinder(radius={round(shell_r * 1.06, 2)}, height={ring_h}, sections={sections})
rim_bottom.apply_translation((0, 0, {round(-shell_h * 0.5, 2)}))
foot_a = trimesh.creation.box(extents=({round(shell_r * 0.55, 2)}, {round(shell_r * 0.2, 2)}, {round(shell_h * 0.25, 2)}))
foot_a.apply_translation(({round(shell_r * 0.65, 2)}, 0, {round(-shell_h * 0.45, 2)}))
foot_b = foot_a.copy()
foot_b.apply_translation(({round(-shell_r * 1.3, 2)}, 0, 0))
mesh = trimesh.util.concatenate([shell, rim_top, rim_bottom, foot_a, foot_b])
""", profile

    if profile == 'idiophone':
        plate_x = round(core_length * 0.95, 2)
        plate_y = round(core_radius * 3.8, 2)
        plate_z = round(core_radius * 0.55, 2)
        return f"""import numpy as np
import trimesh

# map: striking plate -> box, resonant bars -> cylinders, frame -> rails
plate = trimesh.creation.box(extents=({plate_x}, {plate_y}, {plate_z}))
bar_a = trimesh.creation.cylinder(radius={round(core_radius * 0.18, 2)}, height={round(plate_x * 0.9, 2)}, sections={sections})
bar_a.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
bar_a.apply_translation((0, {round(plate_y * 0.32, 2)}, {round(plate_z * 1.15, 2)}))
bar_b = bar_a.copy()
bar_b.apply_translation((0, {round(-plate_y * 0.64, 2)}, 0))
rail_l = trimesh.creation.box(extents=({round(plate_x, 2)}, {round(plate_y * 0.08, 2)}, {round(plate_z * 1.5, 2)}))
rail_l.apply_translation((0, {round(plate_y * 0.52, 2)}, 0))
rail_r = rail_l.copy()
rail_r.apply_translation((0, {round(-plate_y * 1.04, 2)}, 0))
mesh = trimesh.util.concatenate([plate, bar_a, bar_b, rail_l, rail_r])
""", profile

    if profile == 'electro':
        body_x = round(core_length * 0.65, 2)
        body_y = round(core_radius * 4.8, 2)
        body_z = round(core_radius * 2.1, 2)
        horn_h = round(core_radius * (2.4 + aux_scale * 1.8), 2)
        horn_r = round(core_radius * (0.7 + aux_scale * 0.45), 2)
        return f"""import numpy as np
import trimesh

# map: electronic core -> chassis box, transducers -> conical emitters, control node -> sphere
chassis = trimesh.creation.box(extents=({body_x}, {body_y}, {body_z}))
horn_l = trimesh.creation.cone(radius={horn_r}, height={horn_h}, sections={sections})
horn_l.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
horn_l.apply_translation(({round(body_x * 0.54, 2)}, {round(body_y * 0.28, 2)}, 0))
horn_r = horn_l.copy()
horn_r.apply_translation((0, {round(-body_y * 0.56, 2)}, 0))
control = trimesh.creation.icosphere(subdivisions=2, radius={round(core_radius * 0.48, 2)})
control.apply_translation(({round(-body_x * 0.28, 2)}, 0, {round(body_z * 0.72, 2)}))
mesh = trimesh.util.concatenate([chassis, horn_l, horn_r, control])
""", profile

    # Hybrid fallback profile.
    return f"""import numpy as np
import trimesh

# map: core resonance -> cylinder, interaction body -> box, nodal focus -> sphere
core = trimesh.creation.cylinder(radius={core_radius}, height={round(core_length * 0.72, 2)}, sections={sections})
body = trimesh.creation.box(extents=({round(core_length * 0.55, 2)}, {round(core_radius * 3.2, 2)}, {round(core_radius * 1.6, 2)}))
body.apply_translation((0, 0, {round(core_radius * 0.52, 2)}))
node = trimesh.creation.icosphere(subdivisions=2, radius={round(core_radius * 0.5, 2)})
node.apply_translation(({round(core_radius * 0.1, 2)}, {round(core_radius * 0.1, 2)}, {round(core_length * 0.42, 2)}))
mesh = trimesh.util.concatenate([core, body, node])
""", profile


def _build_legacy_fallback_stl_bytes() -> bytes:
    if trimesh is None:
        raise RuntimeError("trimesh is not installed")
    body = trimesh.creation.cylinder(radius=18, height=180, sections=48)
    mouth = trimesh.creation.cylinder(radius=10, height=35, sections=32)
    mouth.apply_translation((0, 0, 107.5))
    base = trimesh.creation.box(extents=(24, 24, 10))
    base.apply_translation((0, 0, -95))
    mesh = trimesh.util.concatenate([body, mouth, base])
    data = mesh.export(file_type='stl')
    return data if isinstance(data, (bytes, bytearray)) else str(data).encode('utf-8')


def build_fallback_stl_bytes(
    prompt: str = "",
    summary_text: str = "",
    materials_text: str = "",
    plot_code: str = ""
):
    """
    Build deterministic but prompt-sensitive STL fallback.
    Returns: (stl_bytes, stl_code, profile)
    """
    stl_code, profile = _build_contextual_fallback_stl_code(prompt, summary_text, materials_text, plot_code)
    try:
        stl_bytes = execute_trimesh_code(stl_code)
        return stl_bytes, stl_code, profile
    except Exception as e:
        logging.error(f"Contextual STL fallback failed ({profile}): {e}")
        stl_bytes = _build_legacy_fallback_stl_bytes()
        legacy_code = (
            "import trimesh\n"
            "# legacy fallback profile used after contextual fallback failure\n"
            "body = trimesh.creation.cylinder(radius=18, height=180, sections=48)\n"
            "mouth = trimesh.creation.cylinder(radius=10, height=35, sections=32)\n"
            "mouth.apply_translation((0, 0, 107.5))\n"
            "base = trimesh.creation.box(extents=(24, 24, 10))\n"
            "base.apply_translation((0, 0, -95))\n"
            "mesh = trimesh.util.concatenate([body, mouth, base])\n"
        )
        return stl_bytes, legacy_code, "legacy"


def execute_trimesh_code(code: str) -> bytes:
    """Execute Python code that builds a trimesh Trimesh/Scene and return STL bytes."""
    if trimesh is None:
        raise RuntimeError("trimesh is not installed; install to enable STL mode")

    cleaned = code.strip()
    cleaned = re.sub(r'^```(?:python|py)?', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
    if ShapelyPolygon is None:
        cleaned = re.sub(r'^\s*(?:from|import)\s+shapely[^\n]*\n?', '', cleaned, flags=re.MULTILINE)
        if re.search(r'\b(LineString|Point|Polygon|unary_union)\b', cleaned):
            raise RuntimeError("shapely is not installed; generate trimesh code using pure trimesh primitives")
    if not cleaned.endswith('\n'):
        cleaned += '\n'
    cleaned = _compile_with_truncation(cleaned, '<stl>')

    # Compat helper: some models call creation.extrude_prism although many trimesh builds expose extrude_polygon.
    def _extrude_prism_compat(polygon, height):
        return trimesh.creation.extrude_polygon(polygon, height)

    # Constrained globals for execution
    g = {
        '__builtins__': __builtins__,
        'trimesh': trimesh,
        'np': np,
    }
    if ShapelyPolygon is not None:
        g['Polygon'] = ShapelyPolygon
    if ShapelyPoint is not None:
        g['Point'] = ShapelyPoint
    if ShapelyLineString is not None:
        g['LineString'] = ShapelyLineString
    if shapely_unary_union is not None:
        g['unary_union'] = shapely_unary_union
    if not hasattr(trimesh.creation, 'extrude_prism') and hasattr(trimesh.creation, 'extrude_polygon'):
        g['extrude_prism'] = _extrude_prism_compat

    l = {}
    exec(cleaned, g, l)

    mesh = l.get('mesh') or g.get('mesh')
    scene = l.get('scene') or g.get('scene')
    if mesh is None and scene is None:
        # try to find any Trimesh instance
        for v in list(l.values()) + list(g.values()):
            if hasattr(trimesh, 'Trimesh') and isinstance(v, trimesh.Trimesh):
                mesh = v
                break
        if mesh is None and hasattr(trimesh, 'Scene'):
            for v in list(l.values()) + list(g.values()):
                if isinstance(v, trimesh.Scene):
                    scene = v
                    break

    if scene is not None and hasattr(scene, 'dump'):
        geoms = []
        for _, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                geoms.append(geom)
        if geoms:
            mesh = trimesh.util.concatenate(geoms)

    if mesh is None:
        if scene is not None and hasattr(scene, 'export'):
            data = scene.export(file_type='stl')
            if isinstance(data, (bytes, bytearray)) and len(data) > 84:
                return data
            if isinstance(data, str) and len(data.strip()) > 0:
                return data.encode('utf-8')
        raise RuntimeError("No `mesh` or exportable `scene` found after executing trimesh code")

    face_count = int(getattr(mesh, 'faces', np.empty((0, 3))).shape[0]) if hasattr(mesh, 'faces') else 0
    if face_count <= 0:
        raise RuntimeError("Generated mesh has zero faces")

    data = mesh.export(file_type='stl')
    if not data:
        raise RuntimeError("Mesh export produced empty STL output")
    return data if isinstance(data, (bytes, bytearray)) else str(data).encode('utf-8')

def _extract_first_line(text: str) -> str:
    for line in (text or "").splitlines():
        cleaned = line.strip().lstrip('#').strip()
        if cleaned:
            return cleaned
    return ""


def _descriptive_slug(primary_text: str, fallback_text: str = "") -> str:
    """
    Build a descriptive title slug using one adjective + at least two nouns.
    Priority source is the first line of primary_text (generation output).
    """
    stop = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'into', 'over', 'under', 'using',
        'your', 'a', 'an', 'of', 'to', 'in', 'on', 'by', 'or', 'as', 'is', 'are', 'be', 'it',
        'its', 'their', 'our', 'you', 'then', 'first', 'finally', 'section', 'materials',
        'conceptual', 'summary'
    }
    adjective_bank = {
        'acoustic', 'resonant', 'hybrid', 'modular', 'speculative', 'dynamic', 'adaptive',
        'harmonic', 'industrial', 'organological', 'performative', 'digital', 'analog',
        'virtual', 'structural', 'aerodynamic', 'geometric', 'mechanical'
    }

    first_line = _extract_first_line(primary_text) or _extract_first_line(fallback_text)
    source = f"{first_line} {fallback_text or ''}".strip()
    words = [w for w in re.findall(r"[a-zA-Z]{3,}", source.lower()) if w not in stop]

    adjective = None
    nouns = []
    for w in words:
        if adjective is None and (
            w in adjective_bank or w.endswith(('ive', 'ous', 'al', 'ic', 'ary', 'ory', 'ful', 'less', 'ant', 'ent'))
        ):
            adjective = w
            continue
        nouns.append(w)

    # Ensure we get at least two nouns
    unique_nouns = []
    for noun in nouns:
        if noun not in unique_nouns:
            unique_nouns.append(noun)
        if len(unique_nouns) >= 2:
            break

    if len(unique_nouns) < 2:
        for w in words:
            if w not in unique_nouns and w != adjective:
                unique_nouns.append(w)
            if len(unique_nouns) >= 2:
                break

    if adjective is None:
        adjective = 'hybrid'
    while len(unique_nouns) < 2:
        unique_nouns.append('organogram')

    parts = [adjective, unique_nouns[0], unique_nouns[1]]
    return "_".join(parts)


def _save_gallery_item(
    prompt: str,
    answer: str,
    code: str,
    image_bytes: bytes,
    stl_bytes: bytes = None,
    llm_model: str = None,
    elapsed_ms: int = None,
    plot_code: str = None,
    stl_code: str = None,
    materials_text: str = None,
    summary_text: str = None,
    refact_meta: dict = None
) -> dict:
    refact_meta = refact_meta or {}
    is_refact = bool(refact_meta.get('is_refact'))
    source_basename = (refact_meta.get('source') or refact_meta.get('base') or '').strip()

    ts = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    requested_title = (refact_meta.get('title') or '').strip()
    descriptor = requested_title or _descriptive_slug(summary_text or answer, fallback_text=prompt)
    title_slug = _safe_slug(descriptor, fallback='hybrid_organogram_system')
    title = title_slug.replace('_', '-')

    if is_refact:
        group_candidate = _safe_slug(
            (refact_meta.get('group') or refact_meta.get('group_id') or '').strip(),
            fallback=''
        )
        group_id = group_candidate or _safe_slug(source_basename or title_slug, fallback=f"{title_slug}_group")
        defaults = _group_defaults(group_id)
        if defaults.get('title_slug'):
            title_slug = _safe_slug(defaults.get('title_slug'), fallback=title_slug)
            title = (defaults.get('title') or title).strip() or title
        seed_version_index = _version_index_from_value(
            refact_meta.get('version_index') or refact_meta.get('version'),
            default=1
        )
        version_index = _next_group_version_index(group_id, seed_version_index=seed_version_index)
    else:
        group_id = f"{title_slug}_{ts}"
        version_index = 1

    version = _version_label_from_index(version_index)
    version_token = version
    base = f"{ts}_{title_slug}_{version_token}"
    png_path = os.path.join(GALLERY_DIR, f"{base}.png")
    txt_path = os.path.join(GALLERY_DIR, f"{base}.txt")
    json_path = os.path.join(GALLERY_DIR, f"{base}.json")
    stl_path = os.path.join(GALLERY_DIR, f"{base}.stl") if stl_bytes else None

    # Resolve collisions (rare but possible with concurrent writes on same second).
    suffix = 1
    while os.path.exists(json_path):
        base = f"{ts}_{title_slug}_{version_token}_{suffix}"
        png_path = os.path.join(GALLERY_DIR, f"{base}.png")
        txt_path = os.path.join(GALLERY_DIR, f"{base}.txt")
        json_path = os.path.join(GALLERY_DIR, f"{base}.json")
        stl_path = os.path.join(GALLERY_DIR, f"{base}.stl") if stl_bytes else None
        suffix += 1

    plot_code = (plot_code or code or '').strip()
    stl_code = (stl_code or '').strip()
    summary_stored = (summary_text or answer or '').strip()
    materials_stored = (materials_text or '').strip() or None
    combined_code = plot_code or stl_code or ''

    # write image only if provided
    has_image = bool(image_bytes) and len(image_bytes) > 0
    if has_image:
        try:
            with open(png_path, 'wb') as f:
                f.write(image_bytes)
        except Exception as e:
            logging.error(f"Error writing PNG file: {e}")
            has_image = False

    # write text
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("# Prompt\n\n")
            f.write(prompt.strip() + "\n\n")
            if summary_stored:
                f.write("# Summary\n\n")
                f.write(summary_stored + "\n")
            if materials_stored:
                f.write("\n# Materials\n\n")
                f.write(materials_stored + "\n")
            if llm_model or elapsed_ms is not None:
                f.write("\n# Generation Metadata\n\n")
                if llm_model:
                    f.write(f"- model: {llm_model}\n")
                if elapsed_ms is not None:
                    f.write(f"- elapsed_ms: {int(elapsed_ms)}\n")
                f.write(f"- group_id: {group_id}\n")
                f.write(f"- version: {version}\n")
                if source_basename:
                    f.write(f"- source: {source_basename}\n")
            if plot_code:
                f.write("\n# Plot Code (matplotlib)\n\n")
                f.write(plot_code + "\n")
            if stl_code:
                f.write("\n# Geometry Code (trimesh)\n\n")
                f.write(stl_code + "\n")
    except Exception:
        pass

    # write STL if provided
    if stl_bytes:
        try:
            with open(stl_path, 'wb') as f:
                f.write(stl_bytes)
        except Exception as e:
            logging.error(f"Error writing STL file: {e}")
            stl_path = None

    meta = {
        'basename': base,
        'timestamp': ts,
        'title': title,
        'title_slug': title_slug,
        'group_id': group_id,
        'version': version,
        'version_index': version_index,
        'source_basename': source_basename or None,
        'prompt': prompt,
        'answer': answer,
        'summary': summary_stored or None,
        'materials_text': materials_stored,
        'code': combined_code,
        'plot_code': plot_code,
        'stl_code': stl_code,
        'llm_model': llm_model,
        'elapsed_ms': int(elapsed_ms) if elapsed_ms is not None else None,
        'image_url': f"/api/gallery/image/{base}.png" if has_image else None,
        'stl_url': f"/api/gallery/file/{base}.stl" if stl_bytes and stl_path else None,
        'modes': [m for m in (['plot'] if has_image else []) + (['stl'] if stl_bytes and stl_path else [])]
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f)
    return meta

def init_models():
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=HF_CACHE_DIR)
        bert_model = AutoModel.from_pretrained("distilbert-base-uncased", cache_dir=HF_CACHE_DIR).to(device)
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir=HF_CACHE_DIR).to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir=HF_CACHE_DIR)

        class MultimodalAttentionModel(nn.Module):
            def __init__(self, text_hidden_size=768, image_hidden_size=512, combined_hidden_size=256):
                super().__init__()
                self.text_fc = nn.Linear(text_hidden_size, combined_hidden_size)
                self.image_fc = nn.Linear(image_hidden_size, combined_hidden_size)
                self.attention = nn.MultiheadAttention(embed_dim=combined_hidden_size, num_heads=4)
                self.classifier = nn.Linear(combined_hidden_size, 10)

            def forward(self, text_features, image_features):
                text_out = self.text_fc(text_features)
                image_out = self.image_fc(image_features)
                attention_out, _ = self.attention(text_out.unsqueeze(1), image_out.unsqueeze(1), image_out.unsqueeze(1))
                return self.classifier(attention_out.squeeze(1))

        multimodal_model = MultimodalAttentionModel().to(device)
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modeltrainer/outputModel/multimodal_model_final.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            multimodal_model.load_state_dict(checkpoint['model_state_dict'])
            multimodal_model.eval()
        return {
            'bert_tokenizer': bert_tokenizer,
            'bert_model': bert_model,
            'clip_model': clip_model,
            'clip_processor': clip_processor,
            'multimodal_model': multimodal_model
        }
    except Exception as e:
        logging.error(f"Error initializing models: {e}")
        raise

models = {}

def load_models_if_needed():
    global models
    if not models:
        models.update(init_models())

def cleanup_models():
    global models
    if models:
        for name, model in models.items():
            if hasattr(model, 'to'):
                model.to('cpu')
                del model
        models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

@app.route('/log', methods=['GET'])
def view_logs():
    try:
        logs = get_logs(limit=100)
        return Response(format_logs_html(logs), content_type='text/html')
    except Exception as e:
        logging.error(f"Error viewing logs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
@log_activity('generate')
def generate():
    data = request.json
    prompt_input = data.get('prompt', '').strip()
    prompt, refact_meta = parse_refact_marker(prompt_input)
    if not prompt and prompt_input:
        prompt = prompt_input
    if not prompt:
        return jsonify({'error': 'No prompt provided.'}), 400
    try:
        limits = get_generation_limits()
        prompt_content = get_prompt_content()
        if "ERROR" in prompt_content:
            return jsonify({'error': prompt_content}), 500
        soopub_context = get_soopub_context(prompt)

        # Keep prompt.txt as primary instruction source; add contextual notes as secondary reference.
        system_prompt = prompt_content
        if soopub_context:
            system_prompt = (
                f"{prompt_content}\n\n"
                "Additional Organological Context (secondary reference, do not override core prompt rules):\n"
                f"{soopub_context}"
            )
        refactor_clause = ""
        if refact_meta.get('is_refact'):
            refactor_clause = (
                "\n\nRefactor mode (lineage-aware):"
                "\n- The prompt includes BASE context from a previous generation."
                "\n- Preserve instrument identity and iterate on the existing design; do not restart from scratch."
                "\n- Reuse and modify provided matplotlib/trimesh code when available."
                "\n- Return full updated code blocks (matplotlib first, trimesh second), not only patches/diffs."
            )
        # Ask for a conceptual summary, then matplotlib code, then a trimesh STL code block
        user_instruction = (
            prompt
            + "\n\nWrite first a concise 120-180 word conceptual summary describing the organogram's design decisions: shapes/schematics, arrows/flows, colors and their meanings, acoustical rationale, organological relations, and performative interactions."
            + "\n\nThen, add a section titled 'Materials' as an enumerated shopping list with quantities and dimensions for building the instrument (e.g., '1 PVC tube, 30 cm length, 2.5 cm diameter'). If part of the instrument is virtual or hybrid, include 'Virtual Materials' items for assets (e.g., textures, shaders, samples)."
            + "\n\nThen, provide the executable Python matplotlib code in a single fenced code block (```python ... ```)."
            + "\n\nFinally, provide a second fenced Python code block that uses trimesh to build a simple, printable 3D representation of the instrument (units in millimeters), assigning the final geometry to a variable named 'mesh' (trimesh.Trimesh). Do not save files; do not display; just build the mesh object."
            + "\n\nStrict generation contract:"
            + "\n- You are a Python and matplotlib assistant specialized in Mantle Hood organograms."
            + "\n- The first Python block must produce a valid matplotlib organogram figure (no placeholders, no pseudo-code, no blank figure)."
            + "\n- Always include all required imports in each code block."
            + "\n- For arrows and geometric patches use matplotlib.patches and ax.annotate/FancyArrowPatch; avoid plt.Arrow."
            + "\n- The second Python block must define `mesh` as a valid trimesh.Trimesh with faces > 0."
            + "\n- The second Python block must be explicitly related to the organogram: map each major organogram component to a corresponding 3D primitive."
            + "\n- Add 3-6 comment lines in the trimesh block with format: '# map: <organogram element> -> <geometry primitive> -> <acoustic function>'."
            + "\n- Vary topology and dimensions according to the prompt, materials, and acoustic target. Do not reuse a generic template."
            + "\n- Prefer pure trimesh primitives (creation.cylinder/box/cone/extrude_polygon). Avoid shapely-dependent pipelines."
            + "\n- Never include the phrase '# Evaluate the text again to render...'."
            + "\n- Keep conceptual summary and materials outside code blocks."
            + "\n\nOptional structured blueprint (use when dimensional pipe/component data is available):"
            + "\n- Define dataclass PipeElement with: name, length_mm, diameter_mm, wall_thickness_mm, material, category, acoustic_target."
            + "\n- Implement plot_geometry_vs_frequency(elements) with a 2x2 figure: length vs fundamental_frequency_hz (log-log), diameter vs fundamental_frequency_hz, histogram of wall_thickness_mm, scatter length vs diameter sized by wall_thickness_mm."
            + "\n- Use seaborn-like aesthetics with standard matplotlib only."
            + "\n- Include a __main__ example with 3-4 dummy elements and ensure plotting code is executed so the backend can render the image."
            + refactor_clause
            + "\n\nDo not include any additional commentary after the code blocks."
        )
        request_start = time.perf_counter()
        deadline_at = (
            request_start + float(limits['deadline_sec'])
            if float(limits['deadline_sec']) > 0
            else None
        )
        try:
            (
                raw_response,
                summary_text,
                materials_text,
                plot_code,
                stl_code,
                image_base64,
                stl_bytes,
                llm_meta
            ) = generate_with_image_required(
                system_prompt,
                user_instruction,
                max_attempts=limits['max_attempts'],
                llm_timeout_sec=limits['llm_timeout_sec'],
                deadline_at=deadline_at
            )
        except Exception as first_error:
            can_retry_without_context = bool(soopub_context) and limits.get('retry_without_context', False)
            has_time_budget = deadline_at is None or (deadline_at - time.perf_counter()) > 25
            if can_retry_without_context and has_time_budget:
                logging.error(
                    f"Primary generation with extended context failed: {first_error}. "
                    "Retrying once with prompt.txt only."
                )
                retry_instruction = (
                    user_instruction
                    + f"\n\nPrevious generation failed with: {first_error}"
                    + "\nRetry with shorter output, but keep full validity of matplotlib and trimesh code blocks."
                )
                (
                    raw_response,
                    summary_text,
                    materials_text,
                    plot_code,
                    stl_code,
                    image_base64,
                    stl_bytes,
                    llm_meta
                ) = generate_with_image_required(
                    prompt_content,
                    retry_instruction,
                    max_attempts=limits['max_attempts'],
                    llm_timeout_sec=limits['llm_timeout_sec'],
                    deadline_at=deadline_at
                )
            else:
                raise
        elapsed_ms = int((time.perf_counter() - request_start) * 1000)

        stl_profile = None
        if not stl_bytes:
            try:
                stl_bytes, fallback_code, stl_profile = build_fallback_stl_bytes(
                    prompt=prompt,
                    summary_text=summary_text or raw_response,
                    materials_text=materials_text or "",
                    plot_code=plot_code or ""
                )
                stl_code = fallback_code
                logging.warning(
                    f"STL fallback mesh was used because generated trimesh output was unavailable (profile={stl_profile})."
                )
            except Exception as stl_fallback_error:
                logging.error(f"Fallback STL generation failed: {stl_fallback_error}")

        image_bytes = base64.b64decode(image_base64)
        meta = None
        try:
            meta = _save_gallery_item(
                prompt,
                summary_text or raw_response,
                plot_code or stl_code or '',
                image_bytes,
                stl_bytes=stl_bytes,
                llm_model=llm_meta.get('model'),
                elapsed_ms=elapsed_ms,
                plot_code=plot_code,
                stl_code=stl_code,
                materials_text=materials_text,
                summary_text=summary_text,
                refact_meta=refact_meta
            )
            if summary_text:
                meta['summary'] = summary_text
            if materials_text:
                meta['materials_text'] = materials_text
        except Exception as e:
            logging.error(f"Error saving gallery item: {e}")

        return jsonify({
            "type": "plot",
            "content": (plot_code or '').strip(),
            "plot_code": (plot_code or '').strip(),
            "stl_code": (stl_code or '').strip(),
            "image": image_base64,
            "gallery": meta,
            "summary": summary_text,
            "materials": materials_text,
            "llm_endpoint": llm_meta.get('endpoint'),
            "llm_model": llm_meta.get('model'),
            "elapsed_ms": elapsed_ms,
            "stl_profile": stl_profile
        })
    except Exception as e:
        logging.error(f"Error in /api/generate: {e}")
        return jsonify({
            'error': f'Failed to generate organogram image: {str(e)}. '
                     f'Ensure Ollama model outputs valid matplotlib code.'
        }), 500

@app.route('/api/predict', methods=['POST'])
@log_activity('predict')
def predict():
    data = request.json
    text = data.get('text', '').strip()
    image_data = data.get('image', '')
    if not text or not image_data:
        return jsonify({'error': 'Both text and image are required.'}), 400
    try:
        load_models_if_needed()
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        text_inputs = models['bert_tokenizer'](text, return_tensors="pt", padding=True, truncation=True)
        image_inputs = models['clip_processor'](images=image, return_tensors="pt")
        with torch.no_grad():
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            image_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in image_inputs.items()}
            text_features = models['bert_model'](**text_inputs).pooler_output
            image_features = models['clip_model'].get_image_features(**image_inputs)
            outputs = models['multimodal_model'](text_features, image_features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        cleanup_models()
        return jsonify({'prediction': predicted_class, 'confidence': confidence})
    except Exception as e:
        logging.error(f"Error in /api/predict: {e}")
        cleanup_models()
        return jsonify({'error': str(e)}), 500

@app.route('/api/version', methods=['GET'])
@log_activity('version')
def version():
    try:
        v = get_version()
        response = jsonify({'version': v})
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        logging.error(f"Error in /api/version: {e}")
        error_response = jsonify({'version': '0.0.0', 'error': str(e)})
        error_response.headers['Content-Type'] = 'application/json'
        return error_response, 500


@app.route('/api/health', methods=['GET'])
@log_activity('health')
def health():
    """Simple health check with version and device info"""
    try:
        base_url, model_name, _ = get_ollama_config()
        return jsonify({
            'status': 'ok',
            'version': get_version(),
            'cuda': torch.cuda.is_available(),
            'device': str(device),
            'llm_provider': 'ollama',
            'ollama': {
                'configured': bool(model_name),
                'base_url': base_url,
                'model': model_name,
                'override_active': bool(OLLAMA_MODEL_OVERRIDE)
            }
        })
    except Exception as e:
        logging.error(f"Error in /api/health: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/somap/graph', methods=['GET'])
@log_activity('somap-graph')
def somap_graph():
    """
    Build a knowledge graph from markdown notes in backend/soopub:
    folders, notes, tags, and Obsidian wikilinks.
    """
    try:
        graph = _build_soopub_graph()
        return jsonify(graph)
    except Exception as e:
        logging.error(f"Error in /api/somap/graph: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ollama/verify', methods=['GET'])
@app.route('/api/deepseek/verify', methods=['GET'])
@log_activity('ollama-verify')
def ollama_verify():
    """Verify Ollama connectivity by listing available models."""
    try:
        res = fetch_ollama_models()
        if res.get('ok'):
            models = res.get('models', [])
            configured = res.get('configured_model')
            return jsonify({
                'ok': True,
                'endpoint': res.get('endpoint'),
                'configured_model': configured,
                'model_available': configured in models,
                'model_count': len(models),
                'models': models[:50]
            })
        return jsonify({
            'ok': False,
            'status': res.get('status', 500),
            'error': res.get('error', 'Unknown Ollama error')
        }), 502
    except Exception as e:
        logging.error(f"Error in /api/ollama/verify: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/ollama/models', methods=['GET'])
@log_activity('ollama-models')
def ollama_models():
    """List installed Ollama models and current active model."""
    try:
        res = fetch_ollama_models()
        if not res.get('ok'):
            return jsonify({
                'ok': False,
                'error': res.get('error', 'Could not list Ollama models'),
                'status': res.get('status', 500)
            }), 502

        models = res.get('models', [])
        current = res.get('configured_model')
        return jsonify({
            'ok': True,
            'models': models,
            'current_model': current,
            'model_available': current in models,
            'override_active': bool(OLLAMA_MODEL_OVERRIDE)
        })
    except Exception as e:
        logging.error(f"Error in /api/ollama/models: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/ollama/model', methods=['POST'])
@log_activity('ollama-model-set')
def ollama_set_model():
    """Set current Ollama model at runtime for this backend process."""
    global OLLAMA_MODEL_OVERRIDE
    try:
        body = request.get_json(force=True) or {}
        requested = (body.get('model') or '').strip()
        if not requested:
            return jsonify({'ok': False, 'error': 'Missing model field'}), 400

        res = fetch_ollama_models()
        if not res.get('ok'):
            return jsonify({
                'ok': False,
                'error': res.get('error', 'Could not list Ollama models')
            }), 502

        models = res.get('models', [])
        if requested not in models:
            return jsonify({
                'ok': False,
                'error': f"Model '{requested}' is not installed",
                'models': models[:50]
            }), 400

        OLLAMA_MODEL_OVERRIDE = requested
        return jsonify({
            'ok': True,
            'model': requested,
            'override_active': True
        })
    except Exception as e:
        logging.error(f"Error in /api/ollama/model: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/gallery/list', methods=['GET'])
def gallery_list():
    try:
        items = []
        if not os.path.isdir(GALLERY_DIR):
            return jsonify({'items': []})
        for name in os.listdir(GALLERY_DIR):
            if name.endswith('.json'):
                path = os.path.join(GALLERY_DIR, name)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        title = (meta.get('title') or '').strip() or _title_from_basename(meta.get('basename', ''))
                        meta['title'] = title or 'untitled'
                        meta['title_slug'] = (meta.get('title_slug') or '').strip() or _safe_slug(meta['title'], fallback='untitled')
                        meta['group_id'] = _meta_group_id(meta) or meta.get('basename', '')
                        version_index = _version_index_from_value(
                            meta.get('version_index') or meta.get('version'),
                            default=1
                        )
                        meta['version_index'] = version_index
                        meta['version'] = (meta.get('version') or '').strip() or _version_label_from_index(version_index)
                        items.append(meta)
                except Exception:
                    continue
        # sort by timestamp desc
        items.sort(key=lambda m: m.get('timestamp', ''), reverse=True)
        return jsonify({'items': items})
    except Exception as e:
        logging.error(f"Error listing gallery: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/gallery/image/<path:filename>', methods=['GET'])
def gallery_image(filename):
    try:
        return send_from_directory(GALLERY_DIR, filename)
    except Exception as e:
        logging.error(f"Error serving gallery image {filename}: {e}")
        return jsonify({'error': 'Image not found'}), 404


@app.route('/api/gallery/file/<path:filename>', methods=['GET'])
def gallery_file(filename):
    """Serve arbitrary gallery files (e.g., STL)."""
    try:
        return send_from_directory(GALLERY_DIR, filename, as_attachment=True)
    except Exception as e:
        logging.error(f"Error serving gallery file {filename}: {e}")
        return jsonify({'error': 'File not found'}), 404


@app.route('/api/gallery/item/<string:basename>', methods=['DELETE'])
def gallery_delete_item(basename):
    """Delete a gallery item by its basename."""
    try:
        # Basic validation on basename
        if not re.match(r'^[a-zA-Z0-9_\-]+$', basename):
            return jsonify({'error': 'Invalid basename format'}), 400

        files_to_delete = []
        for ext in ['.json', '.png', '.txt', '.stl']:
            filename = f"{basename}{ext}"
            path = os.path.join(GALLERY_DIR, filename)
            if os.path.exists(path):
                files_to_delete.append(path)

        if not files_to_delete:
            return jsonify({'error': 'Item not found'}), 404

        for path in files_to_delete:
            try:
                os.remove(path)
            except OSError as e:
                logging.warning(f"Could not delete file {path}: {e}")
                # Decide if you want to fail the whole operation or just log
                # For robustness, we log and continue

        return jsonify({'ok': True, 'message': f'Item {basename} deleted'})
    except Exception as e:
        logging.error(f"Error deleting gallery item {basename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/gallery/item/<string:basename>/rename', methods=['POST'])
def gallery_rename_item(basename):
    """Rename a gallery item."""
    try:
        data = request.json
        new_name = data.get('newName', '').strip()

        if not new_name or not re.match(r'^[a-zA-Z0-9_\-]+$', new_name):
            return jsonify({'error': 'Invalid new name format'}), 400
        
        if not re.match(r'^[a-zA-Z0-9_\-]+$', basename):
            return jsonify({'error': 'Invalid basename format'}), 400

        timestamp = basename.split('_')[0]
        new_basename = f"{timestamp}_{new_name}"

        # Check if new name already exists
        if os.path.exists(os.path.join(GALLERY_DIR, f"{new_basename}.json")):
            return jsonify({'error': 'New name already exists'}), 409

        # Rename files
        renamed_files = []
        for ext in ['.json', '.png', '.txt', '.stl']:
            old_path = os.path.join(GALLERY_DIR, f"{basename}{ext}")
            if os.path.exists(old_path):
                new_path = os.path.join(GALLERY_DIR, f"{new_basename}{ext}")
                os.rename(old_path, new_path)
                renamed_files.append(new_path)
        
        if not renamed_files:
            return jsonify({'error': 'Item not found'}), 404

        # Update JSON metadata
        json_path = os.path.join(GALLERY_DIR, f"{new_basename}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r+') as f:
                meta = json.load(f)
                meta['basename'] = new_basename
                if meta.get('image_url'):
                    meta['image_url'] = f"/api/gallery/image/{new_basename}.png"
                if meta.get('stl_url'):
                    meta['stl_url'] = f"/api/gallery/file/{new_basename}.stl"
                
                f.seek(0)
                json.dump(meta, f)
                f.truncate()

        return jsonify({'ok': True, 'message': f'Item {basename} renamed to {new_basename}', 'newBasename': new_basename})

    except Exception as e:
        logging.error(f"Error renaming gallery item {basename}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/gallery/group/<string:group_id>/rename', methods=['POST'])
def gallery_rename_group(group_id):
    """Rename all versions in the same gallery group."""
    try:
        if not re.match(r'^[a-zA-Z0-9_\-]+$', group_id):
            return jsonify({'error': 'Invalid group format'}), 400

        data = request.get_json(force=True) or {}
        new_name = (data.get('newName') or '').strip()
        if not new_name:
            return jsonify({'error': 'New name is required'}), 400

        new_slug = _safe_slug(new_name, fallback='untitled')
        new_title = new_slug.replace('_', '-')

        selected = []
        for name in os.listdir(GALLERY_DIR):
            if not name.endswith('.json'):
                continue
            path = os.path.join(GALLERY_DIR, name)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                continue
            if _meta_group_id(meta) != group_id:
                continue
            selected.append(meta)

        if not selected:
            return jsonify({'error': 'Group not found'}), 404

        selected.sort(key=lambda meta: meta.get('timestamp', ''))
        used_basenames = {meta.get('basename', '') for meta in selected}
        existing_json_files = {
            os.path.splitext(name)[0]
            for name in os.listdir(GALLERY_DIR)
            if name.endswith('.json')
        }

        updated = []
        for meta in selected:
            old_base = meta.get('basename', '')
            if not old_base:
                continue
            timestamp = str(meta.get('timestamp') or old_base.split('_')[0])
            version_token = _safe_slug(str(meta.get('version') or 'v1'), fallback='v1')
            candidate = f"{timestamp}_{new_slug}_{version_token}"
            suffix = 1
            while candidate in existing_json_files and candidate not in used_basenames:
                candidate = f"{timestamp}_{new_slug}_{version_token}_{suffix}"
                suffix += 1

            new_base = candidate
            existing_json_files.add(new_base)
            used_basenames.add(new_base)

            # Rename physical files
            for ext in ['.json', '.png', '.txt', '.stl']:
                old_path = os.path.join(GALLERY_DIR, f"{old_base}{ext}")
                if not os.path.exists(old_path):
                    continue
                new_path = os.path.join(GALLERY_DIR, f"{new_base}{ext}")
                os.rename(old_path, new_path)

            json_path = os.path.join(GALLERY_DIR, f"{new_base}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r+', encoding='utf-8') as f:
                    current = json.load(f)
                    current['basename'] = new_base
                    current['title'] = new_title
                    current['title_slug'] = new_slug
                    if current.get('image_url'):
                        current['image_url'] = f"/api/gallery/image/{new_base}.png"
                    if current.get('stl_url'):
                        current['stl_url'] = f"/api/gallery/file/{new_base}.stl"
                    f.seek(0)
                    json.dump(current, f)
                    f.truncate()

            updated.append({'old': old_base, 'new': new_base})

        return jsonify({
            'ok': True,
            'group_id': group_id,
            'title': new_title,
            'title_slug': new_slug,
            'updated': updated
        })
    except Exception as e:
        logging.error(f"Error renaming gallery group {group_id}: {e}")
        return jsonify({'error': str(e)}), 500




@app.route('/api/dev/save', methods=['POST'])
def dev_save():
    # Restricted in production unless explicitly enabled
    if not app.debug and os.getenv('ENABLE_DEV_SAVE', '') != '1':
        return jsonify({'error': 'dev_save disabled'}), 403
    try:
        data = request.get_json(force=True) or {}
        prompt = (data.get('prompt') or 'dev test').strip()
        summary = (data.get('summary') or '').strip()
        matplotlib_code = data.get('matplotlib_code') or ''
        trimesh_code = data.get('trimesh_code') or ''

        image_bytes = b''
        stl_bytes = None
        plot_code = None
        stl_code = None

        if matplotlib_code.strip():
            try:
                image_b64 = execute_matplotlib_code(matplotlib_code)
                image_bytes = base64.b64decode(image_b64)
                plot_code = matplotlib_code
            except Exception as e:
                logging.error(f"Dev matplotlib exec failed: {e}")
        if trimesh_code.strip():
            try:
                stl_bytes = execute_trimesh_code(trimesh_code)
                stl_code = trimesh_code
            except Exception as e:
                logging.error(f"Dev trimesh exec failed: {e}")

        if not image_bytes and not stl_bytes:
            return jsonify({'error': 'No outputs produced'}), 400

        code_to_save = plot_code or stl_code or ''
        meta = _save_gallery_item(prompt, summary, code_to_save, image_bytes, stl_bytes=stl_bytes)
        return jsonify({'ok': True, 'gallery': meta})
    except Exception as e:
        logging.error(f"Error in /api/dev/save: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    debug_mode = os.getenv("FLASK_DEBUG", "1").strip().lower() in ("1", "true", "yes", "on")
    use_reloader = os.getenv("FLASK_RELOAD", "1").strip().lower() in ("1", "true", "yes", "on")
    logging.info(
        f"Starting Flask app on port {port} "
        f"(debug={debug_mode}, reloader={use_reloader})"
    )
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=use_reloader)
