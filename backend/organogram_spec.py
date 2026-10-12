"""Declarative organogram specs and a deterministic Mantle Hood renderer.

The LLM describes the instrument as a small JSON spec (parts, classes, materials, counts, flows).
This module turns that spec into matplotlib code that follows the Mantle Hood primitives and the
material colour coding of prompt.txt, so small models no longer need to write valid drawing code.
"""
import math
import re

HOOD_CLASSES = ('idiophone', 'membranophone', 'chordophone', 'aerophone', 'electronophone', 'microphone', 'speaker')
MATERIALS = ('wood', 'bamboo', 'metal', 'stone', 'glass', 'water', 'gourd', 'skin', 'bone', 'plastic', 'other')
FLOW_KINDS = ('excitation', 'airflow', 'signal', 'radiation')

MATERIAL_COLORS = {
    'wood': '#ff9a2f', 'bamboo': '#ffff00', 'metal': '#c0c0c0', 'stone': '#ffffff', 'glass': '#00ff00',
    'water': '#0000ff', 'gourd': '#d2b48c', 'skin': '#ffc0cb', 'bone': '#e0e0e0', 'plastic': '#9aa5b1',
    'other': '#8f8fff',
}
FLOW_STYLES = {
    'excitation': ('#ff9a2f', '->'),
    'airflow': ('#7fd4ff', '->'),
    'signal': ('#ffffff', '-|>'),
    'radiation': ('#c77dff', '->'),
}

CLASS_SYNONYMS = {
    'percussion': 'idiophone', 'bar': 'idiophone', 'bell': 'idiophone', 'plate': 'idiophone',
    'drum': 'membranophone', 'membrane': 'membranophone',
    'string': 'chordophone', 'strings': 'chordophone',
    'wind': 'aerophone', 'air': 'aerophone', 'pipe': 'aerophone', 'reed': 'aerophone',
    'electronic': 'electronophone', 'electrophone': 'electronophone', 'electric': 'electronophone',
    'mic': 'microphone', 'loudspeaker': 'speaker',
}
MATERIAL_SYNONYMS = {
    'steel': 'metal', 'copper': 'metal', 'brass': 'metal', 'bronze': 'metal', 'aluminum': 'metal',
    'aluminium': 'metal', 'iron': 'metal', 'silver': 'metal',
    'leather': 'skin', 'hide': 'skin', 'membrane': 'skin',
    'pvc': 'plastic', 'resin': 'plastic', 'acrylic': 'plastic', 'polymer': 'plastic',
    'ceramic': 'stone', 'clay': 'stone', 'marble': 'stone', 'earth': 'gourd',
    'reed': 'bamboo', 'cane': 'bamboo',
    'carbon': 'plastic', 'silicon': 'other', 'electronics': 'other', 'organic': 'other',
}


def build_spec_schema():
    return {
        'type': 'object',
        'properties': {
            'summary': {'type': 'string'},
            'materials': {'type': 'string'},
            'parts': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'class': {'type': 'string', 'enum': list(HOOD_CLASSES)},
                        'material': {'type': 'string', 'enum': list(MATERIALS)},
                        'count': {'type': 'integer'},
                        'role': {'type': 'string'},
                    },
                    'required': ['name', 'class', 'material', 'count'],
                },
            },
            'flows': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'source': {'type': 'integer'},
                        'target': {'type': 'integer'},
                        'kind': {'type': 'string', 'enum': list(FLOW_KINDS)},
                    },
                    'required': ['source', 'target', 'kind'],
                },
            },
        },
        'required': ['summary', 'materials', 'parts', 'flows'],
    }


def build_spec_instruction(prompt):
    return (
        f"{prompt}\n\n"
        "Describe this instrument as a structured Mantle Hood organogram. Return one JSON object with:\n"
        "- summary: 120-180 word conceptual summary of the physics, excitation, resonance, radiation and performance.\n"
        "- materials: enumerated shopping list (plain text lines) with quantities and dimensions; add virtual items if hybrid.\n"
        "- parts: 3 to 8 distinct components that exist in THIS instrument. For each: name (short), class "
        "(idiophone=square, membranophone=horizontal rectangle, chordophone=vertical rectangle, aerophone=circle, "
        "electronophone=rhombus, microphone, speaker), material (wood, bamboo, metal, stone, glass, water, gourd, skin, "
        "bone, plastic, other), count (number of identical elements such as holes, strings, keys), role.\n"
        "- flows: arrows between parts using zero-based part indexes (source, target) with kind "
        "(excitation, airflow, signal, radiation).\n"
        "Choose classes and materials from the description itself; different instruments must yield different parts."
    )


def _clean_word(value):
    return re.sub(r'[^a-z]', '', str(value or '').lower())


def _norm_class(value):
    word = _clean_word(value)
    if word in HOOD_CLASSES:
        return word
    if word in CLASS_SYNONYMS:
        return CLASS_SYNONYMS[word]
    if len(word) >= 5:
        for cls in HOOD_CLASSES:
            if cls.startswith(word[:5]):
                return cls
    return 'idiophone'


def _norm_material(value):
    word = _clean_word(value)
    if word in MATERIALS:
        return word
    return MATERIAL_SYNONYMS.get(word, 'other')


def normalize_spec(spec):
    """Return a clean spec or None when it has no usable parts."""
    if not isinstance(spec, dict):
        return None
    raw_parts = spec.get('parts')
    if not isinstance(raw_parts, list):
        return None
    parts = []
    for item in raw_parts[:9]:
        if not isinstance(item, dict):
            continue
        name = re.sub(r'\s+', ' ', str(item.get('name') or '')).strip()[:28] or f'part {len(parts) + 1}'
        try:
            count = int(item.get('count') or 1)
        except (TypeError, ValueError):
            count = 1
        parts.append({
            'name': name,
            'class': _norm_class(item.get('class')),
            'material': _norm_material(item.get('material')),
            'count': max(1, min(count, 99)),
        })
    if not parts:
        return None
    flows = []
    for item in spec.get('flows') or []:
        if not isinstance(item, dict):
            continue
        try:
            src, dst = int(item.get('source')), int(item.get('target'))
        except (TypeError, ValueError):
            continue
        kind = _clean_word(item.get('kind'))
        if 0 <= src < len(parts) and 0 <= dst < len(parts) and src != dst:
            flows.append({'source': src, 'target': dst, 'kind': kind if kind in FLOW_KINDS else 'signal'})
    if not flows and len(parts) > 1:
        flows = [{'source': i, 'target': i + 1, 'kind': 'excitation' if i == 0 else 'radiation'} for i in range(len(parts) - 1)]
    return {'parts': parts, 'flows': flows[:16]}


def _centers(n):
    if n == 1:
        return [(0.5, 0.5)], 0.30
    cols = 2 if n <= 4 else 3
    rows = math.ceil(n / cols)
    unit = 0.62 / max(cols, rows) * 0.62
    points = []
    for i in range(n):
        r, c = divmod(i, cols)
        in_row = min(cols, n - r * cols)
        x = (c + 0.5) / cols if in_row == cols else 0.5 + (c - (in_row - 1) / 2.0) / cols
        y = 1.0 - (r + 0.5) / rows
        points.append((0.08 + 0.84 * x, 0.08 + 0.84 * y))
    return points, unit


def _shape_code(cls, cx, cy, unit, color):
    u = unit
    face = f"facecolor='{color}', edgecolor='white', alpha=0.75"
    if cls == 'idiophone':
        return f"ax.add_patch(Rectangle(({cx - u / 2:.4f}, {cy - u / 2:.4f}), {u:.4f}, {u:.4f}, {face}))"
    if cls == 'membranophone':
        w, h = u * 1.5, u * 0.7
        return f"ax.add_patch(Rectangle(({cx - w / 2:.4f}, {cy - h / 2:.4f}), {w:.4f}, {h:.4f}, {face}))"
    if cls == 'chordophone':
        w, h = u * 0.55, u * 1.5
        return f"ax.add_patch(Rectangle(({cx - w / 2:.4f}, {cy - h / 2:.4f}), {w:.4f}, {h:.4f}, {face}))"
    if cls == 'aerophone':
        return f"ax.add_patch(Circle(({cx:.4f}, {cy:.4f}), {u * 0.55:.4f}, {face}))"
    if cls in ('electronophone', 'microphone'):
        s = u * (0.62 if cls == 'electronophone' else 0.32)
        pts = [[round(cx, 4), round(cy + s, 4)], [round(cx + s, 4), round(cy, 4)],
               [round(cx, 4), round(cy - s, 4)], [round(cx - s, 4), round(cy, 4)]]
        return f"ax.add_patch(Polygon({pts}, closed=True, {face}))"
    w, h = u * 0.8, u * 0.7  # speaker: small horizontal cone
    pts = [[round(cx - w / 2, 4), round(cy, 4)], [round(cx + w / 2, 4), round(cy + h / 2, 4)],
           [round(cx + w / 2, 4), round(cy - h / 2, 4)]]
    return f"ax.add_patch(Polygon({pts}, closed=True, {face}))"


def render_spec_code(spec):
    """Matplotlib code for a normalized spec, following prompt.txt (no axes, equal aspect, dark)."""
    parts, flows = spec['parts'], spec['flows']
    centers, unit = _centers(len(parts))
    lines = [
        'import matplotlib.pyplot as plt',
        'from matplotlib.patches import Circle, Rectangle, Polygon, FancyArrowPatch',
        "plt.style.use('dark_background')",
        'fig, ax = plt.subplots(figsize=(10, 6))',
        '# organogram generated from a structured spec (Mantle Hood primitives, material colours)',
    ]
    for idx, (part, (cx, cy)) in enumerate(zip(parts, centers)):
        color = MATERIAL_COLORS[part['material']]
        label = re.sub(r'[\r\n]+', ' ', part['name'])
        lines.append(f"# part {idx}: {label} -> class {part['class']}, material {part['material']}, count {part['count']}")
        lines.append(_shape_code(part['class'], cx, cy, unit, color))
        lines.append(f"ax.text({cx + unit * 0.62:.4f}, {cy + unit * 0.62:.4f}, '{part['count']}', color='#4aa3ff', fontsize=11, ha='left', va='bottom')")
        lines.append(f"ax.text({cx:.4f}, {cy - unit * 1.0:.4f}, {label!r}, color='#9aa0a6', fontsize=7, ha='center', va='top')")
    for flow in flows:
        (sx, sy), (tx, ty) = centers[flow['source']], centers[flow['target']]
        dx, dy = tx - sx, ty - sy
        dist = math.hypot(dx, dy) or 1.0
        pad = unit * 0.95
        pa = (sx + dx / dist * pad, sy + dy / dist * pad)
        pb = (tx - dx / dist * pad, ty - dy / dist * pad)
        color, style = FLOW_STYLES[flow['kind']]
        lines.append(
            f"ax.add_patch(FancyArrowPatch(posA=({pa[0]:.4f}, {pa[1]:.4f}), posB=({pb[0]:.4f}, {pb[1]:.4f}), "
            f"connectionstyle='arc3,rad=0.15', color='{color}', arrowstyle='{style}', mutation_scale=14, linewidth=1.4))"
        )
    lines += ['ax.set_xlim(0, 1)', 'ax.set_ylim(0, 1)', "ax.set_aspect('equal')", 'ax.set_axis_off()', 'fig.patch.set_alpha(0.0)']
    return '\n'.join(lines) + '\n'


_CLASS_KEYWORDS = (
    ('electronophone', ('electr', 'signal', 'circuit', 'sensor', 'synth', 'digital', 'microcontroller', 'bio', 'plant', 'neural', 'software', 'midi')),
    ('chordophone', ('string', 'wire', 'cord', 'bow', 'pluck', 'harp', 'lute', 'zither')),
    ('membranophone', ('drum', 'skin', 'membrane', 'head', 'tympan')),
    ('aerophone', ('air', 'breath', 'flute', 'pipe', 'tube', 'reed', 'wind', 'whistle', 'horn', 'organ', 'bellows')),
    ('idiophone', ('bar', 'bell', 'plate', 'gong', 'chime', 'rattle', 'glass', 'stone', 'hammer', 'percuss', 'marimba')),
)
_MATERIAL_KEYWORDS = (
    ('bamboo', ('bamboo', 'cane')), ('wood', ('wood', 'spruce', 'maple', 'oak', 'timber')),
    ('metal', ('metal', 'steel', 'copper', 'brass', 'bronze', 'aluminum', 'aluminium', 'iron')),
    ('glass', ('glass',)), ('stone', ('stone', 'marble', 'ceramic', 'clay')), ('water', ('water', 'liquid')),
    ('gourd', ('gourd', 'earth', 'seed')), ('skin', ('skin', 'leather', 'hide', 'membrane')),
    ('bone', ('bone', 'ivory')), ('plastic', ('pvc', 'plastic', 'resin', 'acrylic', 'polymer', 'printed')),
)


def spec_from_text(*texts):
    """Deterministic spec from free text, used when the model returns nothing usable."""
    text = ' '.join(str(t or '') for t in texts).lower()
    classes = [cls for cls, words in _CLASS_KEYWORDS if any(w in text for w in words)][:4] or ['idiophone']
    materials = [m for m, words in _MATERIAL_KEYWORDS if any(w in text for w in words)] or ['other']
    parts = [
        {'name': f'{cls} body', 'class': cls, 'material': materials[i % len(materials)], 'count': 1}
        for i, cls in enumerate(classes)
    ]
    if 'microphone' not in classes and any(w in text for w in ('amplif', 'pickup', 'transduc')):
        parts.append({'name': 'pickup', 'class': 'microphone', 'material': 'other', 'count': 1})
    parts.append({'name': 'radiation', 'class': 'speaker', 'material': 'other', 'count': 1})
    return normalize_spec({'parts': parts, 'flows': []})
