import json
import math
import re
import sys
from typing import Iterable

import numpy as np

SPEED_OF_SOUND = 343.0
GRID_SIZE = 84
PRIMITIVES = {"circle", "square", "triangle", "hexagon"}


def _safe_float(value, default):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _point_pair(params: dict, prefix: str, fallback_x: float, fallback_y: float):
    return [
        _clamp(_safe_float(params.get(f"{prefix}_x"), fallback_x), -0.95, 0.95),
        _clamp(_safe_float(params.get(f"{prefix}_y"), fallback_y), -0.95, 0.95),
    ]


def _extract_frequencies(prompt: str, params: dict):
    explicit = _safe_float(params.get("freq"), 440.0)
    found = [float(token) for token in re.findall(r"(\d+(?:\.\d+)?)\s*(?:hz|khz|freq)", prompt or "", re.I)]
    normalized = []
    for value in found[:5]:
        normalized.append(value * 1000.0 if value < 10 and "khz" in (prompt or "").lower() else value)
    if not normalized:
        normalized = [explicit]
    cleaned = []
    seen = set()
    for freq in normalized:
        bounded = round(_clamp(freq, 20.0, 4000.0), 2)
        if bounded not in seen:
            seen.add(bounded)
            cleaned.append(bounded)
    return cleaned[:4]


def _regular_polygon(sides: int, radius: float, rotation: float = math.pi / 2):
    return [
        (
            radius * math.cos(rotation + idx * (2 * math.pi / sides)),
            radius * math.sin(rotation + idx * (2 * math.pi / sides)),
        )
        for idx in range(sides)
    ]


def _point_in_polygon(x: float, y: float, vertices: Iterable[tuple[float, float]]) -> bool:
    inside = False
    vertices = list(vertices)
    j = len(vertices) - 1
    for i, (xi, yi) in enumerate(vertices):
        xj, yj = vertices[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _distance_to_segment(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = _clamp(t, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def _polygon_signed_distance(x: float, y: float, vertices: list[tuple[float, float]]) -> float:
    distances = []
    for idx, (ax, ay) in enumerate(vertices):
        bx, by = vertices[(idx + 1) % len(vertices)]
        distances.append(_distance_to_segment(x, y, ax, ay, bx, by))
    distance = min(distances) if distances else 0.0
    return distance if _point_in_polygon(x, y, vertices) else -distance


def _signed_distance_field(primitive: str, x_grid, y_grid):
    field = np.zeros_like(x_grid, dtype=float)
    if primitive == "circle":
        radius = 0.86
        field = radius - np.sqrt(x_grid * x_grid + y_grid * y_grid)
        return field

    if primitive == "square":
        half = 0.84
        field = np.minimum(half - np.abs(x_grid), half - np.abs(y_grid))
        return field

    sides = 3 if primitive == "triangle" else 6
    vertices = _regular_polygon(sides, 0.9, math.pi / 2 if primitive == "triangle" else math.pi / 6)
    for idx in np.ndindex(x_grid.shape):
        field[idx] = _polygon_signed_distance(float(x_grid[idx]), float(y_grid[idx]), vertices)
    return field


def _mode_field(primitive: str, x_grid, y_grid, freq: float):
    normalized = _clamp(freq / 900.0, 0.25, 3.2)
    if primitive == "circle":
        radius = np.sqrt(x_grid * x_grid + y_grid * y_grid)
        theta = np.arctan2(y_grid, x_grid)
        return (
            np.cos((2.2 + normalized) * math.pi * radius)
            + 0.45 * np.cos(3 * theta) * np.cos((1.2 + normalized * 0.5) * math.pi * radius)
        )

    if primitive == "square":
        xn = (x_grid + 1.0) * 0.5
        yn = (y_grid + 1.0) * 0.5
        return (
            np.cos((1 + normalized) * math.pi * xn) * np.cos((2 + normalized * 0.5) * math.pi * yn)
            + 0.35 * np.cos((3 + normalized * 0.3) * math.pi * xn) * np.cos((1 + normalized) * math.pi * yn)
        )

    if primitive == "triangle":
        dirs = np.array([
            [1.0, 0.0],
            [-0.5, math.sqrt(3) * 0.5],
            [-0.5, -math.sqrt(3) * 0.5],
        ])
    else:
        dirs = np.array([
            [1.0, 0.0],
            [0.5, math.sqrt(3) * 0.5],
            [-0.5, math.sqrt(3) * 0.5],
            [-1.0, 0.0],
            [-0.5, -math.sqrt(3) * 0.5],
            [0.5, -math.sqrt(3) * 0.5],
        ])

    field = np.zeros_like(x_grid, dtype=float)
    scale = 1.2 + normalized * 0.45
    for direction in dirs:
        projection = x_grid * direction[0] + y_grid * direction[1]
        field += np.cos(math.pi * scale * projection)
    return field / max(len(dirs), 1)


def _estimate_resonance_peaks(primitive: str, freqs: list[float]):
    base = min(freqs) if freqs else 440.0
    ratios = {
        "circle": [0.82, 1.0, 1.34, 1.68, 2.08, 2.56],
        "square": [0.71, 1.0, 1.41, 1.58, 2.0, 2.24],
        "triangle": [0.76, 1.0, 1.27, 1.61, 1.96, 2.31],
        "hexagon": [0.79, 1.0, 1.29, 1.52, 1.88, 2.18],
    }
    return [round(base * ratio, 2) for ratio in ratios.get(primitive, ratios["circle"])]


def run_simulation(params):
    prompt = str(params.get("prompt") or "")
    primitive = str(params.get("primitive") or "circle").strip().lower()
    if primitive not in PRIMITIVES:
        primitive = "circle"

    freqs = _extract_frequencies(prompt, params)
    source = _point_pair(params, "source", -0.55, 0.0)
    probe = _point_pair(params, "probe", 0.55, 0.0)
    obstacle = _point_pair(params, "obs", 0.15, 0.0)

    x_range = np.linspace(-1, 1, GRID_SIZE)
    y_range = np.linspace(-1, 1, GRID_SIZE)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    signed_distance = _signed_distance_field(primitive, x_grid, y_grid)
    cavity_mask = signed_distance >= 0
    boundary_weight = np.clip(signed_distance / 0.22, 0.0, 1.0)

    pressure = np.zeros_like(x_grid, dtype=float)
    source_records = []

    for idx, freq in enumerate(freqs):
        k = 2.0 * math.pi * freq / SPEED_OF_SOUND
        phase = idx * (math.pi / 5.0)
        sx, sy = source
        dx = x_grid - sx
        dy = y_grid - sy
        dist = np.sqrt(dx * dx + dy * dy)
        direct = np.cos(k * dist + phase) * np.exp(-0.95 * dist) / np.sqrt(dist + 0.06)

        mode_shape = _mode_field(primitive, x_grid, y_grid, freq)
        modal_weight = math.exp(-abs(freq - min(freqs)) / 1200.0)
        standing = mode_shape * modal_weight * np.cos(phase)

        ox, oy = obstacle
        obs_dist = np.sqrt((x_grid - ox) ** 2 + (y_grid - oy) ** 2)
        obstacle_core = np.exp(-(obs_dist ** 2) / 0.012)
        shadow_axis = np.array([ox - sx, oy - sy], dtype=float)
        shadow_norm = float(np.dot(shadow_axis, shadow_axis))
        shadow = np.ones_like(x_grid, dtype=float)
        if shadow_norm > 1e-6:
            ahead = ((x_grid - sx) * shadow_axis[0] + (y_grid - sy) * shadow_axis[1]) / shadow_norm
            lateral = np.abs((x_grid - ox) * shadow_axis[1] - (y_grid - oy) * shadow_axis[0]) / math.sqrt(shadow_norm)
            behind = ahead > 1.0
            shadow = np.where(behind, 1.0 - 0.48 * np.exp(-(lateral ** 2) / 0.03), 1.0)
        attenuation = np.clip(1.0 - 0.4 * obstacle_core, 0.18, 1.0)

        component = (0.72 * direct + 0.38 * standing) * attenuation * shadow
        pressure += component
        source_records.append({"pos": [round(sx, 3), round(sy, 3)], "freq": round(freq, 2)})

    pressure *= cavity_mask * (0.34 + 0.66 * boundary_weight)

    if np.any(cavity_mask):
        cavity_values = pressure[cavity_mask]
        mean = float(np.mean(cavity_values))
        std = float(np.std(cavity_values)) or 1.0
        pressure = (pressure - mean) / std
    pressure = np.where(cavity_mask, pressure, 0.0)

    max_abs = float(np.max(np.abs(pressure))) or 1.0
    pressure /= max_abs

    probe_x_idx = int(np.argmin(np.abs(x_range - probe[0])))
    probe_y_idx = int(np.argmin(np.abs(y_range - probe[1])))
    probe_response = float(pressure[probe_y_idx, probe_x_idx])

    result = {
        "status": "success",
        "method": "Primitive cavity acoustic surrogate",
        "params": {
            "primitive": primitive,
            "frequencies_hz": freqs,
            "freq": float(freqs[0]),
            "sources": source_records,
            "probe": [round(probe[0], 3), round(probe[1], 3)],
            "obstacle": [round(obstacle[0], 3), round(obstacle[1], 3)],
            "grid_size": GRID_SIZE,
        },
        "results": {
            "pressure_map": pressure.tolist(),
            "probe_response": probe_response,
            "mic_response": probe_response,
            "max_p": float(np.max(pressure)),
            "min_p": float(np.min(pressure)),
            "cavity_fill_ratio": round(float(np.mean(cavity_mask)), 4),
            "resonance_peaks_hz": _estimate_resonance_peaks(primitive, freqs),
        },
    }
    return result


if __name__ == "__main__":
    input_params = {}
    if len(sys.argv) > 1:
        try:
            input_params = json.loads(sys.argv[1])
        except Exception:
            input_params = {}
    print(json.dumps(run_simulation(input_params)))
