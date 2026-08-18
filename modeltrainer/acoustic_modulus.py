import json
import math
import re
import sys

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SPEED_OF_SOUND = 343.0
GRID_SIZE_2D = 60
GRID_SIZE_3D = 22
EIGENMODE_COUNT_2D = 8
EIGENMODE_COUNT_3D = 6
PLANE_PRIMITIVES = {"circle", "square", "triangle", "hexagon"}
VOLUME_PRIMITIVES = {"sphere", "cube", "cylinder"}


def _safe_float(value, default):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _normalize_mode(raw_mode, primitive):
    mode = str(raw_mode or "").strip().lower()
    if mode in {"3d", "volume", "volumetric"}:
        return "3d"
    if str(primitive or "").strip().lower() in VOLUME_PRIMITIVES:
        return "3d"
    return "2d"


def _normalize_solver(raw_solver, prompt_text=""):
    solver = str(raw_solver or "").strip().lower()
    if solver in {"surrogate", "preview", "fast"}:
        return "surrogate"
    if solver in {"fd", "finite_difference", "finite-difference", "helmholtz", "fem", "phase2"}:
        return "fd"
    lowered = str(prompt_text or "").lower()
    if any(keyword in lowered for keyword in ("surrogate", "preview-only", "fast preview")):
        return "surrogate"
    return "fd"


def _normalize_primitive(mode: str, raw_primitive: str):
    primitive = str(raw_primitive or "").strip().lower()
    aliases_2d = {
        "round": "circle",
        "circular": "circle",
        "box": "square",
        "boxed": "square",
        "triangular": "triangle",
        "hexagonal": "hexagon",
        "honeycomb": "hexagon",
    }
    aliases_3d = {
        "round": "sphere",
        "circular": "sphere",
        "ball": "sphere",
        "box": "cube",
        "boxed": "cube",
        "pipe": "cylinder",
        "tube": "cylinder",
        "tubular": "cylinder",
    }

    if mode == "3d":
        resolved = aliases_3d.get(primitive, primitive)
        return resolved if resolved in VOLUME_PRIMITIVES else "sphere"

    resolved = aliases_2d.get(primitive, primitive)
    return resolved if resolved in PLANE_PRIMITIVES else "circle"


def _point_tuple(params: dict, prefix: str, fallbacks: list[float]):
    coords = []
    for axis, fallback in zip(("x", "y", "z"), fallbacks):
        coords.append(_clamp(_safe_float(params.get(f"{prefix}_{axis}"), fallback), -0.95, 0.95))
    return coords


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


def _point_in_polygon(x: float, y: float, vertices: list[tuple[float, float]]) -> bool:
    inside = False
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


def _signed_distance_field_2d(primitive: str, x_grid, y_grid):
    field = np.zeros_like(x_grid, dtype=float)
    if primitive == "circle":
        radius = 0.86
        return radius - np.sqrt(x_grid * x_grid + y_grid * y_grid)

    if primitive == "square":
        half = 0.84
        return np.minimum(half - np.abs(x_grid), half - np.abs(y_grid))

    sides = 3 if primitive == "triangle" else 6
    vertices = _regular_polygon(sides, 0.9, math.pi / 2 if primitive == "triangle" else math.pi / 6)
    for idx in np.ndindex(x_grid.shape):
        field[idx] = _polygon_signed_distance(float(x_grid[idx]), float(y_grid[idx]), vertices)
    return field


def _signed_distance_field_3d(primitive: str, x_grid, y_grid, z_grid):
    if primitive == "sphere":
        radius = 0.84
        return radius - np.sqrt(x_grid * x_grid + y_grid * y_grid + z_grid * z_grid)

    if primitive == "cube":
        half = 0.8
        return np.minimum.reduce((half - np.abs(x_grid), half - np.abs(y_grid), half - np.abs(z_grid)))

    radius = 0.72
    half_height = 0.88
    radial = radius - np.sqrt(x_grid * x_grid + y_grid * y_grid)
    axial = half_height - np.abs(z_grid)
    return np.minimum(radial, axial)


def _mode_field_2d(primitive: str, x_grid, y_grid, freq: float):
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


def _mode_field_3d(primitive: str, x_grid, y_grid, z_grid, freq: float):
    normalized = _clamp(freq / 900.0, 0.25, 3.2)

    if primitive == "sphere":
        radius = np.sqrt(x_grid * x_grid + y_grid * y_grid + z_grid * z_grid)
        theta = np.arctan2(y_grid, x_grid)
        phi = np.arctan2(z_grid, np.sqrt(x_grid * x_grid + y_grid * y_grid) + 1e-9)
        return (
            np.cos((2.0 + normalized) * math.pi * radius)
            + 0.28 * np.cos(2 * theta) * np.cos((1.2 + normalized * 0.35) * math.pi * radius)
            + 0.22 * np.sin(2 * phi) * np.cos((1.1 + normalized * 0.22) * math.pi * radius)
        )

    if primitive == "cube":
        xn = (x_grid + 1.0) * 0.5
        yn = (y_grid + 1.0) * 0.5
        zn = (z_grid + 1.0) * 0.5
        return (
            np.cos((1.0 + normalized) * math.pi * xn)
            * np.cos((1.55 + normalized * 0.35) * math.pi * yn)
            * np.cos((2.1 + normalized * 0.28) * math.pi * zn)
            + 0.26 * np.cos((3.0 + normalized * 0.2) * math.pi * (xn - zn))
        )

    radial = np.sqrt(x_grid * x_grid + y_grid * y_grid)
    theta = np.arctan2(y_grid, x_grid)
    axial = (z_grid + 1.0) * 0.5
    return (
        np.cos((1.9 + normalized * 0.45) * math.pi * radial)
        * np.cos((1.05 + normalized * 0.3) * math.pi * axial)
        + 0.3 * np.cos(3 * theta) * np.cos((1.35 + normalized * 0.22) * math.pi * radial)
    )


def _estimate_resonance_peaks(mode: str, primitive: str, freqs: list[float]):
    base = min(freqs) if freqs else 440.0
    ratios_2d = {
        "circle": [0.82, 1.0, 1.34, 1.68, 2.08, 2.56],
        "square": [0.71, 1.0, 1.41, 1.58, 2.0, 2.24],
        "triangle": [0.76, 1.0, 1.27, 1.61, 1.96, 2.31],
        "hexagon": [0.79, 1.0, 1.29, 1.52, 1.88, 2.18],
    }
    ratios_3d = {
        "sphere": [0.74, 1.0, 1.31, 1.67, 1.94, 2.28],
        "cube": [0.69, 1.0, 1.26, 1.55, 1.83, 2.12],
        "cylinder": [0.72, 1.0, 1.22, 1.49, 1.78, 2.06],
    }
    source = ratios_3d if mode == "3d" else ratios_2d
    fallback = "sphere" if mode == "3d" else "circle"
    return [round(base * ratio, 2) for ratio in source.get(primitive, source[fallback])]


def _infer_reference_span_m(mode: str, primitive: str, freqs: list[float]):
    base_freq = max(60.0, min(freqs) if freqs else 440.0)
    base_span = SPEED_OF_SOUND / (2.0 * base_freq)
    factors_2d = {
        "circle": 1.08,
        "square": 1.0,
        "triangle": 0.94,
        "hexagon": 1.02,
    }
    factors_3d = {
        "sphere": 1.12,
        "cube": 1.0,
        "cylinder": 1.06,
    }
    factor = (factors_3d if mode == "3d" else factors_2d).get(primitive, 1.0)
    upper = 1.0 if mode == "3d" else 1.35
    return _clamp(base_span * factor, 0.18, upper)


def _obstacle_radius_norm(mode: str):
    return 0.14 if mode == "3d" else 0.11


def _carve_obstacle_2d(cavity_mask, x_grid, y_grid, obstacle):
    ox, oy = obstacle[:2]
    radius_norm = _obstacle_radius_norm("2d")
    obstacle_mask = ((x_grid - ox) ** 2 + (y_grid - oy) ** 2) <= radius_norm ** 2
    carved = cavity_mask & ~obstacle_mask
    if np.any(carved):
        return carved, radius_norm
    return cavity_mask, 0.0


def _carve_obstacle_3d(cavity_mask, x_grid, y_grid, z_grid, obstacle):
    ox, oy, oz = obstacle[:3]
    radius_norm = _obstacle_radius_norm("3d")
    obstacle_mask = ((x_grid - ox) ** 2 + (y_grid - oy) ** 2 + (z_grid - oz) ** 2) <= radius_norm ** 2
    carved = cavity_mask & ~obstacle_mask
    if np.any(carved):
        return carved, radius_norm
    return cavity_mask, 0.0


def _build_active_cloud_2d(mask, x_range, y_range):
    coords = np.argwhere(mask)
    index_map = np.full(mask.shape, -1, dtype=int)
    if coords.size == 0:
        return coords, index_map, np.zeros((0, 2), dtype=float)
    index_map[mask] = np.arange(coords.shape[0])
    points = np.column_stack((x_range[coords[:, 1]], y_range[coords[:, 0]])).astype(float)
    return coords, index_map, points


def _build_active_cloud_3d(mask, x_range, y_range, z_range):
    coords = np.argwhere(mask)
    index_map = np.full(mask.shape, -1, dtype=int)
    if coords.size == 0:
        return coords, index_map, np.zeros((0, 3), dtype=float)
    index_map[mask] = np.arange(coords.shape[0])
    points = np.column_stack((x_range[coords[:, 1]], y_range[coords[:, 0]], z_range[coords[:, 2]])).astype(float)
    return coords, index_map, points


def _build_stiffness_matrix(mask, index_map, step_m):
    dims = mask.ndim
    if dims == 2:
        deltas = ((0, 1), (0, -1), (1, 0), (-1, 0))
    else:
        deltas = ((0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1))

    rows = []
    cols = []
    data = []
    inv_h2 = 1.0 / (step_m * step_m)

    for coord in np.argwhere(mask):
        index = index_map[tuple(coord)]
        degree = 0.0
        for delta in deltas:
            neighbor = tuple(coord[axis] + delta[axis] for axis in range(dims))
            if any(value < 0 or value >= mask.shape[axis] for axis, value in enumerate(neighbor)):
                continue
            if not mask[neighbor]:
                continue
            degree += 1.0
            rows.append(index)
            cols.append(index_map[neighbor])
            data.append(-inv_h2)
        rows.append(index)
        cols.append(index)
        data.append(max(degree, 1.0) * inv_h2)

    size = int(np.max(index_map)) + 1 if np.any(index_map >= 0) else 0
    return sp.csr_matrix((data, (rows, cols)), shape=(size, size), dtype=float)


def _gaussian_excitation(points, target_point, grid_size):
    if points.size == 0:
        return np.zeros((0,), dtype=float), -1, np.zeros((0,), dtype=float)

    dims = points.shape[1]
    target = np.asarray(target_point[:dims], dtype=float)
    deltas = points - target
    dist2 = np.sum(deltas * deltas, axis=1)
    sigma_norm = max(0.08, 2.8 / max(grid_size - 1, 1))
    weights = np.exp(-(dist2 / (2.0 * sigma_norm * sigma_norm)))
    weights[dist2 > (sigma_norm * 3.2) ** 2] *= 0.22
    if not np.any(weights > 1e-12):
        nearest = int(np.argmin(dist2))
        weights = np.zeros(points.shape[0], dtype=float)
        weights[nearest] = 1.0
        return weights, nearest, points[nearest]

    weights_sum = float(np.sum(weights)) or 1.0
    weights /= weights_sum
    nearest = int(np.argmax(weights))
    return weights, nearest, points[nearest]


def _solve_sparse_response(system_matrix, rhs):
    try:
        solution = spla.spsolve(system_matrix.tocsc(), rhs)
        if np.all(np.isfinite(solution)):
            return solution
    except Exception:
        pass

    solution, info = spla.bicgstab(system_matrix, rhs, rtol=1e-6, atol=0.0, maxiter=max(800, rhs.shape[0] * 2))
    if info != 0 or not np.all(np.isfinite(solution)):
        raise RuntimeError(f"finite-difference solve failed (info={info})")
    return solution


def _estimate_resonance_peaks_fd(stiffness, count):
    size = stiffness.shape[0]
    if size <= 2:
        return []
    eigen_count = min(max(count + 1, 3), size - 1)
    try:
        eigenvalues = spla.eigsh(stiffness, k=eigen_count, which="SM", return_eigenvectors=False)
    except Exception:
        return []
    peaks = []
    for value in sorted(float(max(0.0, ev)) for ev in eigenvalues):
        if value <= 1e-9:
            continue
        freq = (SPEED_OF_SOUND / (2.0 * math.pi)) * math.sqrt(value)
        if 20.0 <= freq <= 8000.0:
            rounded = round(freq, 2)
            if rounded not in peaks:
                peaks.append(rounded)
    return peaks[:count]


def _render_fd_field(solution, mask, taper):
    if solution.size == 0:
        return np.zeros(mask.shape, dtype=float)
    field = np.zeros(mask.shape, dtype=float)
    field[mask] = np.real(solution)
    field *= mask * (0.28 + 0.72 * taper)
    if np.any(mask):
        cavity_values = field[mask]
        mean = float(np.mean(cavity_values))
        std = float(np.std(cavity_values)) or 1.0
        field = np.where(mask, (field - mean) / std, 0.0)
        max_abs = float(np.max(np.abs(field[mask]))) or 1.0
        field = np.where(mask, field / max_abs, 0.0)
    return field


def _as_float_list(values):
    return [float(round(value, 6)) for value in values]


def _fd_summary_payload(mode, primitive, freqs, span_m, step_m, cavity_mask, obstacle_radius_m, source_point, probe_point, obstacle_point, active_nodes, resonance_peaks):
    return {
        "mode": mode,
        "primitive": primitive,
        "frequencies_hz": [float(freq) for freq in freqs],
        "freq": float(freqs[0]),
        "span_m": round(float(span_m), 4),
        "cell_pitch_m": round(float(step_m), 5),
        "cavity_fill_ratio": round(float(np.mean(cavity_mask)), 4),
        "active_nodes": int(active_nodes),
        "obstacle_radius_m": round(float(obstacle_radius_m), 5),
        "probe": _as_float_list(probe_point),
        "source": _as_float_list(source_point),
        "obstacle": _as_float_list(obstacle_point),
        "resonance_peaks_hz": resonance_peaks,
    }


def _run_fd_2d_simulation(params, prompt, primitive, freqs):
    source = _point_tuple(params, "source", [-0.55, 0.0])
    probe = _point_tuple(params, "probe", [0.55, 0.0])
    obstacle = _point_tuple(params, "obs", [0.15, 0.0])

    x_range = np.linspace(-1, 1, GRID_SIZE_2D)
    y_range = np.linspace(-1, 1, GRID_SIZE_2D)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    signed_distance = _signed_distance_field_2d(primitive, x_grid, y_grid)
    cavity_mask = signed_distance >= 0
    cavity_mask, obstacle_radius_norm = _carve_obstacle_2d(cavity_mask, x_grid, y_grid, obstacle)
    if not np.any(cavity_mask):
        raise RuntimeError("2d cavity mask is empty after obstacle carving")

    taper_cells = np.clip(ndi.distance_transform_edt(cavity_mask) / 2.4, 0.0, 1.0)
    span_m = _infer_reference_span_m("2d", primitive, freqs)
    step_m = span_m / max(GRID_SIZE_2D - 1, 1)
    obstacle_radius_m = obstacle_radius_norm * span_m * 0.5

    coords, index_map, points = _build_active_cloud_2d(cavity_mask, x_range, y_range)
    stiffness = _build_stiffness_matrix(cavity_mask, index_map, step_m)
    if stiffness.shape[0] == 0:
        raise RuntimeError("2d stiffness matrix is empty")

    source_weights, source_idx, snapped_source = _gaussian_excitation(points, source, GRID_SIZE_2D)
    _, probe_idx, snapped_probe = _gaussian_excitation(points, probe, GRID_SIZE_2D)
    _, obstacle_idx, snapped_obstacle = _gaussian_excitation(points, obstacle, GRID_SIZE_2D)

    base_freq = min(freqs) if freqs else 440.0
    identity = sp.identity(stiffness.shape[0], format="csr", dtype=np.complex128)
    response = np.zeros(stiffness.shape[0], dtype=np.complex128)
    frequency_response = []

    for idx, freq in enumerate(freqs):
        k = 2.0 * math.pi * freq / SPEED_OF_SOUND
        damping = 0.055 + 0.012 * idx
        system = stiffness.astype(np.complex128) - ((k * k) * (1.0 - 1j * damping)) * identity
        rhs = source_weights.astype(np.complex128) * np.exp(1j * idx * (math.pi / 7.0))
        solved = _solve_sparse_response(system, rhs)
        weight = math.exp(-abs(freq - base_freq) / max(base_freq * 0.72, 150.0))
        response += weight * solved
        frequency_response.append({
            "freq_hz": round(float(freq), 2),
            "probe_amplitude": round(float(abs(solved[probe_idx])), 6),
        })

    field = _render_fd_field(response, cavity_mask, taper_cells)
    probe_coord = tuple(coords[probe_idx])
    probe_response = float(field[probe_coord])
    resonance_peaks = _estimate_resonance_peaks_fd(stiffness, EIGENMODE_COUNT_2D)
    max_p = float(np.max(field))
    min_p = float(np.min(field))

    result = {
        "status": "success",
        "solver": "fd_helmholtz_phase2",
        "method": "Finite-difference cavity solver (Phase 2)",
        "params": {
            "mode": "2d",
            "primitive": primitive,
            "frequencies_hz": freqs,
            "freq": float(freqs[0]),
            "sources": [{"pos": _as_float_list(snapped_source), "freq": round(float(freq), 2)} for freq in freqs],
            "probe": _as_float_list(snapped_probe),
            "obstacle": _as_float_list(snapped_obstacle),
            "grid_size": GRID_SIZE_2D,
            "active_nodes": int(stiffness.shape[0]),
            "cell_pitch_m": round(float(step_m), 5),
            "span_m": round(float(span_m), 4),
            "boundary": "voxel_neumann",
        },
        "results": {
            "pressure_map": np.round(field, 6).tolist(),
            "probe_response": probe_response,
            "probe_amplitude": round(float(abs(response[probe_idx])), 6),
            "probe_phase_rad": round(float(np.angle(response[probe_idx])), 6),
            "mic_response": probe_response,
            "max_p": max_p,
            "min_p": min_p,
            "cavity_fill_ratio": round(float(np.mean(cavity_mask)), 4),
            "resonance_peaks_hz": resonance_peaks or _estimate_resonance_peaks("2d", primitive, freqs),
            "frequency_response_probe": frequency_response,
            "solver_summary": _fd_summary_payload(
                "2d", primitive, freqs, span_m, step_m, cavity_mask, obstacle_radius_m,
                snapped_source, snapped_probe, snapped_obstacle, stiffness.shape[0],
                resonance_peaks or _estimate_resonance_peaks("2d", primitive, freqs)
            ),
        },
    }
    if obstacle_idx >= 0:
        result["params"]["obstacle_index"] = int(obstacle_idx)
    if source_idx >= 0:
        result["params"]["source_index"] = int(source_idx)
    return result


def _run_fd_3d_simulation(params, prompt, primitive, freqs):
    source = _point_tuple(params, "source", [-0.45, 0.0, -0.28])
    probe = _point_tuple(params, "probe", [0.42, 0.08, 0.32])
    obstacle = _point_tuple(params, "obs", [0.08, 0.12, 0.0])

    x_range = np.linspace(-1, 1, GRID_SIZE_3D)
    y_range = np.linspace(-1, 1, GRID_SIZE_3D)
    z_range = np.linspace(-1, 1, GRID_SIZE_3D)
    x_grid, y_grid, z_grid = np.meshgrid(x_range, y_range, z_range, indexing="xy")

    signed_distance = _signed_distance_field_3d(primitive, x_grid, y_grid, z_grid)
    cavity_mask = signed_distance >= 0
    cavity_mask, obstacle_radius_norm = _carve_obstacle_3d(cavity_mask, x_grid, y_grid, z_grid, obstacle)
    if not np.any(cavity_mask):
        raise RuntimeError("3d cavity mask is empty after obstacle carving")

    taper_cells = np.clip(ndi.distance_transform_edt(cavity_mask) / 2.2, 0.0, 1.0)
    span_m = _infer_reference_span_m("3d", primitive, freqs)
    step_m = span_m / max(GRID_SIZE_3D - 1, 1)
    obstacle_radius_m = obstacle_radius_norm * span_m * 0.5

    coords, index_map, points = _build_active_cloud_3d(cavity_mask, x_range, y_range, z_range)
    stiffness = _build_stiffness_matrix(cavity_mask, index_map, step_m)
    if stiffness.shape[0] == 0:
        raise RuntimeError("3d stiffness matrix is empty")

    source_weights, source_idx, snapped_source = _gaussian_excitation(points, source, GRID_SIZE_3D)
    _, probe_idx, snapped_probe = _gaussian_excitation(points, probe, GRID_SIZE_3D)
    _, obstacle_idx, snapped_obstacle = _gaussian_excitation(points, obstacle, GRID_SIZE_3D)

    base_freq = min(freqs) if freqs else 440.0
    identity = sp.identity(stiffness.shape[0], format="csr", dtype=np.complex128)
    response = np.zeros(stiffness.shape[0], dtype=np.complex128)
    frequency_response = []

    for idx, freq in enumerate(freqs):
        k = 2.0 * math.pi * freq / SPEED_OF_SOUND
        damping = 0.07 + 0.014 * idx
        system = stiffness.astype(np.complex128) - ((k * k) * (1.0 - 1j * damping)) * identity
        rhs = source_weights.astype(np.complex128) * np.exp(1j * idx * (math.pi / 8.0))
        solved = _solve_sparse_response(system, rhs)
        weight = math.exp(-abs(freq - base_freq) / max(base_freq * 0.85, 170.0))
        response += weight * solved
        frequency_response.append({
            "freq_hz": round(float(freq), 2),
            "probe_amplitude": round(float(abs(solved[probe_idx])), 6),
        })

    field = _render_fd_field(response, cavity_mask, taper_cells)
    probe_coord = tuple(coords[probe_idx])
    probe_response = float(field[probe_coord])
    resonance_peaks = _estimate_resonance_peaks_fd(stiffness, EIGENMODE_COUNT_3D)

    probe_y_idx = int(coords[probe_idx][0])
    probe_x_idx = int(coords[probe_idx][1])
    probe_z_idx = int(coords[probe_idx][2])
    slice_xy = np.round(field[:, :, probe_z_idx], 6).tolist()
    slice_xz = np.round(np.transpose(field[probe_y_idx, :, :]), 6).tolist()
    slice_yz = np.round(np.transpose(field[:, probe_x_idx, :]), 6).tolist()
    pressure_volume = np.round(np.transpose(field, (2, 0, 1)), 6).tolist()

    result = {
        "status": "success",
        "solver": "fd_helmholtz_phase2",
        "method": "Volumetric finite-difference cavity solver (Phase 2)",
        "params": {
            "mode": "3d",
            "primitive": primitive,
            "frequencies_hz": freqs,
            "freq": float(freqs[0]),
            "sources": [{"pos": _as_float_list(snapped_source), "freq": round(float(freq), 2)} for freq in freqs],
            "probe": _as_float_list(snapped_probe),
            "obstacle": _as_float_list(snapped_obstacle),
            "grid_size": GRID_SIZE_3D,
            "grid_shape": [GRID_SIZE_3D, GRID_SIZE_3D, GRID_SIZE_3D],
            "slice_indices": {"x": probe_x_idx, "y": probe_y_idx, "z": probe_z_idx},
            "active_nodes": int(stiffness.shape[0]),
            "cell_pitch_m": round(float(step_m), 5),
            "span_m": round(float(span_m), 4),
            "boundary": "voxel_neumann",
        },
        "results": {
            "pressure_map": slice_xy,
            "pressure_volume": pressure_volume,
            "slice_xy": slice_xy,
            "slice_xz": slice_xz,
            "slice_yz": slice_yz,
            "probe_response": probe_response,
            "probe_amplitude": round(float(abs(response[probe_idx])), 6),
            "probe_phase_rad": round(float(np.angle(response[probe_idx])), 6),
            "mic_response": probe_response,
            "max_p": float(np.max(field)),
            "min_p": float(np.min(field)),
            "cavity_fill_ratio": round(float(np.mean(cavity_mask)), 4),
            "resonance_peaks_hz": resonance_peaks or _estimate_resonance_peaks("3d", primitive, freqs),
            "frequency_response_probe": frequency_response,
            "solver_summary": _fd_summary_payload(
                "3d", primitive, freqs, span_m, step_m, cavity_mask, obstacle_radius_m,
                snapped_source, snapped_probe, snapped_obstacle, stiffness.shape[0],
                resonance_peaks or _estimate_resonance_peaks("3d", primitive, freqs)
            ),
        },
    }
    if obstacle_idx >= 0:
        result["params"]["obstacle_index"] = int(obstacle_idx)
    if source_idx >= 0:
        result["params"]["source_index"] = int(source_idx)
    return result


def _run_surrogate_2d_simulation(params, prompt, primitive, freqs):
    source = _point_tuple(params, "source", [-0.55, 0.0])
    probe = _point_tuple(params, "probe", [0.55, 0.0])
    obstacle = _point_tuple(params, "obs", [0.15, 0.0])

    x_range = np.linspace(-1, 1, GRID_SIZE_2D)
    y_range = np.linspace(-1, 1, GRID_SIZE_2D)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    signed_distance = _signed_distance_field_2d(primitive, x_grid, y_grid)
    cavity_mask = signed_distance >= 0
    boundary_weight = np.clip(signed_distance / 0.22, 0.0, 1.0)

    pressure = np.zeros_like(x_grid, dtype=float)
    source_records = []
    min_freq = min(freqs) if freqs else 440.0

    for idx, freq in enumerate(freqs):
        k = 2.0 * math.pi * freq / SPEED_OF_SOUND
        phase = idx * (math.pi / 5.0)
        sx, sy = source[:2]
        dx = x_grid - sx
        dy = y_grid - sy
        dist = np.sqrt(dx * dx + dy * dy)
        direct = np.cos(k * dist + phase) * np.exp(-0.95 * dist) / np.sqrt(dist + 0.06)

        mode_shape = _mode_field_2d(primitive, x_grid, y_grid, freq)
        modal_weight = math.exp(-abs(freq - min_freq) / 1200.0)
        standing = mode_shape * modal_weight * np.cos(phase)

        ox, oy = obstacle[:2]
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

    return {
        "status": "success",
        "solver": "surrogate_preview",
        "method": "Live preview acoustic surrogate",
        "params": {
            "mode": "2d",
            "primitive": primitive,
            "frequencies_hz": freqs,
            "freq": float(freqs[0]),
            "sources": source_records,
            "probe": [round(probe[0], 3), round(probe[1], 3)],
            "obstacle": [round(obstacle[0], 3), round(obstacle[1], 3)],
            "grid_size": GRID_SIZE_2D,
        },
        "results": {
            "pressure_map": np.round(pressure, 6).tolist(),
            "probe_response": probe_response,
            "mic_response": probe_response,
            "max_p": float(np.max(pressure)),
            "min_p": float(np.min(pressure)),
            "cavity_fill_ratio": round(float(np.mean(cavity_mask)), 4),
            "resonance_peaks_hz": _estimate_resonance_peaks("2d", primitive, freqs),
        },
    }


def _run_surrogate_3d_simulation(params, prompt, primitive, freqs):
    source = _point_tuple(params, "source", [-0.45, 0.0, -0.28])
    probe = _point_tuple(params, "probe", [0.42, 0.08, 0.32])
    obstacle = _point_tuple(params, "obs", [0.08, 0.12, 0.0])

    x_range = np.linspace(-1, 1, GRID_SIZE_3D)
    y_range = np.linspace(-1, 1, GRID_SIZE_3D)
    z_range = np.linspace(-1, 1, GRID_SIZE_3D)
    x_grid, y_grid, z_grid = np.meshgrid(x_range, y_range, z_range, indexing="xy")

    signed_distance = _signed_distance_field_3d(primitive, x_grid, y_grid, z_grid)
    cavity_mask = signed_distance >= 0
    boundary_weight = np.clip(signed_distance / 0.18, 0.0, 1.0)

    pressure = np.zeros_like(x_grid, dtype=float)
    source_records = []
    min_freq = min(freqs) if freqs else 440.0

    for idx, freq in enumerate(freqs):
        k = 2.0 * math.pi * freq / SPEED_OF_SOUND
        phase = idx * (math.pi / 6.0)
        sx, sy, sz = source
        dx = x_grid - sx
        dy = y_grid - sy
        dz = z_grid - sz
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        direct = np.cos(k * dist + phase) * np.exp(-1.08 * dist) / np.power(dist + 0.05, 0.68)

        mode_shape = _mode_field_3d(primitive, x_grid, y_grid, z_grid, freq)
        modal_weight = math.exp(-abs(freq - min_freq) / 1350.0)
        standing = mode_shape * modal_weight * np.cos(phase)

        ox, oy, oz = obstacle
        obs_dist = np.sqrt((x_grid - ox) ** 2 + (y_grid - oy) ** 2 + (z_grid - oz) ** 2)
        obstacle_core = np.exp(-(obs_dist ** 2) / 0.022)
        shadow_axis = np.array([ox - sx, oy - sy, oz - sz], dtype=float)
        shadow_norm = float(np.dot(shadow_axis, shadow_axis))
        shadow = np.ones_like(x_grid, dtype=float)
        if shadow_norm > 1e-6:
            ahead = ((x_grid - sx) * shadow_axis[0] + (y_grid - sy) * shadow_axis[1] + (z_grid - sz) * shadow_axis[2]) / shadow_norm
            rel_to_obstacle = np.stack((x_grid - ox, y_grid - oy, z_grid - oz), axis=-1)
            cross = np.cross(rel_to_obstacle, shadow_axis)
            lateral = np.linalg.norm(cross, axis=-1) / math.sqrt(shadow_norm)
            behind = ahead > 1.0
            shadow = np.where(behind, 1.0 - 0.52 * np.exp(-(lateral ** 2) / 0.05), 1.0)
        attenuation = np.clip(1.0 - 0.44 * obstacle_core, 0.16, 1.0)

        component = (0.7 * direct + 0.44 * standing) * attenuation * shadow
        pressure += component
        source_records.append({"pos": [round(sx, 3), round(sy, 3), round(sz, 3)], "freq": round(freq, 2)})

    pressure *= cavity_mask * (0.3 + 0.7 * boundary_weight)

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
    probe_z_idx = int(np.argmin(np.abs(z_range - probe[2])))
    probe_response = float(pressure[probe_y_idx, probe_x_idx, probe_z_idx])

    slice_xy = np.round(pressure[:, :, probe_z_idx], 6).tolist()
    slice_xz = np.round(np.transpose(pressure[probe_y_idx, :, :]), 6).tolist()
    slice_yz = np.round(np.transpose(pressure[:, probe_x_idx, :]), 6).tolist()
    pressure_volume = np.round(np.transpose(pressure, (2, 0, 1)), 6).tolist()

    return {
        "status": "success",
        "solver": "surrogate_preview",
        "method": "Live preview volumetric acoustic surrogate",
        "params": {
            "mode": "3d",
            "primitive": primitive,
            "frequencies_hz": freqs,
            "freq": float(freqs[0]),
            "sources": source_records,
            "probe": [round(probe[0], 3), round(probe[1], 3), round(probe[2], 3)],
            "obstacle": [round(obstacle[0], 3), round(obstacle[1], 3), round(obstacle[2], 3)],
            "grid_size": GRID_SIZE_3D,
            "grid_shape": [GRID_SIZE_3D, GRID_SIZE_3D, GRID_SIZE_3D],
            "slice_indices": {"x": probe_x_idx, "y": probe_y_idx, "z": probe_z_idx},
        },
        "results": {
            "pressure_map": slice_xy,
            "pressure_volume": pressure_volume,
            "slice_xy": slice_xy,
            "slice_xz": slice_xz,
            "slice_yz": slice_yz,
            "probe_response": probe_response,
            "mic_response": probe_response,
            "max_p": float(np.max(pressure)),
            "min_p": float(np.min(pressure)),
            "cavity_fill_ratio": round(float(np.mean(cavity_mask)), 4),
            "resonance_peaks_hz": _estimate_resonance_peaks("3d", primitive, freqs),
        },
    }


def run_simulation(params):
    prompt = str(params.get("prompt") or "")
    mode = _normalize_mode(params.get("mode"), params.get("primitive"))
    primitive = _normalize_primitive(mode, params.get("primitive") or ("sphere" if mode == "3d" else "circle"))
    freqs = _extract_frequencies(prompt, params)
    solver = _normalize_solver(params.get("solver"), prompt)

    if solver == "surrogate":
        return _run_surrogate_3d_simulation(params, prompt, primitive, freqs) if mode == "3d" else _run_surrogate_2d_simulation(params, prompt, primitive, freqs)

    try:
        return _run_fd_3d_simulation(params, prompt, primitive, freqs) if mode == "3d" else _run_fd_2d_simulation(params, prompt, primitive, freqs)
    except Exception as error:
        fallback = _run_surrogate_3d_simulation(params, prompt, primitive, freqs) if mode == "3d" else _run_surrogate_2d_simulation(params, prompt, primitive, freqs)
        fallback["status"] = "degraded"
        fallback["requested_solver"] = "fd"
        fallback["fallback_reason"] = str(error)
        fallback["method"] = f"{fallback['method']} (fallback from phase 2 solver)"
        return fallback


if __name__ == "__main__":
    input_params = {}
    if len(sys.argv) > 1:
        try:
            input_params = json.loads(sys.argv[1])
        except Exception:
            input_params = {}
    print(json.dumps(run_simulation(input_params)))
