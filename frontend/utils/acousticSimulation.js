const SPEED_OF_SOUND = 343.0;
const GRID_SIZE_2D = 84;
const GRID_SIZE_3D = 28;
const PLANE_PRIMITIVES = new Set(['circle', 'square', 'triangle', 'hexagon']);
const VOLUME_PRIMITIVES = new Set(['sphere', 'cube', 'cylinder']);

function safeFloat(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeMode(rawMode, primitive) {
  const mode = String(rawMode || '').trim().toLowerCase();
  if (mode === '3d' || mode === 'volume' || mode === 'volumetric') return '3d';
  if (VOLUME_PRIMITIVES.has(String(primitive || '').toLowerCase())) return '3d';
  return '2d';
}

function normalizePrimitive(mode, rawPrimitive) {
  const primitive = String(rawPrimitive || '').trim().toLowerCase();
  const aliases2d = {
    round: 'circle',
    circular: 'circle',
    box: 'square',
    boxed: 'square',
    triangular: 'triangle',
    hexagonal: 'hexagon',
    honeycomb: 'hexagon',
  };
  const aliases3d = {
    round: 'sphere',
    circular: 'sphere',
    ball: 'sphere',
    box: 'cube',
    boxed: 'cube',
    pipe: 'cylinder',
    tube: 'cylinder',
    tubular: 'cylinder',
  };

  if (mode === '3d') {
    const resolved = aliases3d[primitive] || primitive;
    return VOLUME_PRIMITIVES.has(resolved) ? resolved : 'sphere';
  }

  const resolved = aliases2d[primitive] || primitive;
  return PLANE_PRIMITIVES.has(resolved) ? resolved : 'circle';
}

function buildRange(size) {
  return Array.from({ length: size }, (_, index) => -1 + ((2 * index) / (size - 1)));
}

function nearestIndex(range, target) {
  return range.reduce((best, value, index) => (
    Math.abs(value - target) < Math.abs(range[best] - target) ? index : best
  ), 0);
}

function pointTuple(params, prefix, fallbacks) {
  return fallbacks.map((fallback, index) => {
    const axis = ['x', 'y', 'z'][index];
    return clamp(safeFloat(params[`${prefix}_${axis}`], fallback), -0.95, 0.95);
  });
}

function extractFrequencies(prompt, params) {
  const explicit = safeFloat(params.freq, 440.0);
  const matches = String(prompt || '').match(/(\d+(?:\.\d+)?)\s*(?:hz|khz|freq)/gi) || [];
  const values = matches
    .slice(0, 5)
    .map((chunk) => {
      const numeric = safeFloat(chunk, explicit);
      return /khz/i.test(chunk) && numeric < 10 ? numeric * 1000 : numeric;
    })
    .filter((numeric) => Number.isFinite(numeric));

  const source = values.length ? values : [explicit];
  const cleaned = [];
  const seen = new Set();

  for (const freq of source) {
    const bounded = Math.round(clamp(freq, 20, 4000) * 100) / 100;
    if (!seen.has(bounded)) {
      seen.add(bounded);
      cleaned.push(bounded);
    }
  }

  return cleaned.slice(0, 4);
}

function roundFieldValue(value) {
  return Math.round(value * 1000000) / 1000000;
}

function regularPolygon(sides, radius, rotation = Math.PI / 2) {
  return Array.from({ length: sides }, (_, index) => ([
    radius * Math.cos(rotation + index * (2 * Math.PI / sides)),
    radius * Math.sin(rotation + index * (2 * Math.PI / sides)),
  ]));
}

function pointInPolygon(x, y, vertices) {
  let inside = false;
  let j = vertices.length - 1;
  for (let i = 0; i < vertices.length; i += 1) {
    const [xi, yi] = vertices[i];
    const [xj, yj] = vertices[j];
    const intersects = ((yi > y) !== (yj > y))
      && (x < (((xj - xi) * (y - yi)) / ((yj - yi) || 1e-9)) + xi);
    if (intersects) inside = !inside;
    j = i;
  }
  return inside;
}

function distanceToSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const denom = (abx * abx) + (aby * aby);
  if (denom <= 1e-12) return Math.hypot(px - ax, py - ay);

  const t = clamp((((px - ax) * abx) + ((py - ay) * aby)) / denom, 0, 1);
  const cx = ax + (t * abx);
  const cy = ay + (t * aby);
  return Math.hypot(px - cx, py - cy);
}

function polygonSignedDistance(x, y, vertices) {
  let distance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < vertices.length; index += 1) {
    const [ax, ay] = vertices[index];
    const [bx, by] = vertices[(index + 1) % vertices.length];
    distance = Math.min(distance, distanceToSegment(x, y, ax, ay, bx, by));
  }
  return pointInPolygon(x, y, vertices) ? distance : -distance;
}

function buildSignedDistanceField2D(primitive, xGrid, yGrid) {
  const rows = yGrid.length;
  const cols = xGrid.length;
  const field = Array.from({ length: rows }, () => Array(cols).fill(0));

  if (primitive === 'circle') {
    const radius = 0.86;
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const x = xGrid[col];
        const y = yGrid[row];
        field[row][col] = radius - Math.sqrt((x * x) + (y * y));
      }
    }
    return field;
  }

  if (primitive === 'square') {
    const half = 0.84;
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const x = xGrid[col];
        const y = yGrid[row];
        field[row][col] = Math.min(half - Math.abs(x), half - Math.abs(y));
      }
    }
    return field;
  }

  const sides = primitive === 'triangle' ? 3 : 6;
  const rotation = primitive === 'triangle' ? Math.PI / 2 : Math.PI / 6;
  const vertices = regularPolygon(sides, 0.9, rotation);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      field[row][col] = polygonSignedDistance(xGrid[col], yGrid[row], vertices);
    }
  }
  return field;
}

function buildSignedDistanceField3D(primitive, xGrid, yGrid, zGrid) {
  const depth = zGrid.length;
  const rows = yGrid.length;
  const cols = xGrid.length;
  const field = Array.from({ length: depth }, () => Array.from({ length: rows }, () => Array(cols).fill(0)));

  for (let zIndex = 0; zIndex < depth; zIndex += 1) {
    const z = zGrid[zIndex];
    for (let row = 0; row < rows; row += 1) {
      const y = yGrid[row];
      for (let col = 0; col < cols; col += 1) {
        const x = xGrid[col];
        let signed = -1;

        if (primitive === 'sphere') {
          const radius = 0.84;
          signed = radius - Math.sqrt((x * x) + (y * y) + (z * z));
        } else if (primitive === 'cube') {
          const half = 0.8;
          signed = Math.min(half - Math.abs(x), half - Math.abs(y), half - Math.abs(z));
        } else {
          const radius = 0.72;
          const halfHeight = 0.88;
          const radial = radius - Math.sqrt((x * x) + (y * y));
          const axial = halfHeight - Math.abs(z);
          signed = Math.min(radial, axial);
        }

        field[zIndex][row][col] = signed;
      }
    }
  }

  return field;
}

function modeField2D(primitive, x, y, freq) {
  const normalized = clamp(freq / 900, 0.25, 3.2);

  if (primitive === 'circle') {
    const radius = Math.sqrt((x * x) + (y * y));
    const theta = Math.atan2(y, x);
    return (
      Math.cos((2.2 + normalized) * Math.PI * radius)
      + (0.45 * Math.cos(3 * theta) * Math.cos((1.2 + (normalized * 0.5)) * Math.PI * radius))
    );
  }

  if (primitive === 'square') {
    const xn = (x + 1) * 0.5;
    const yn = (y + 1) * 0.5;
    return (
      Math.cos((1 + normalized) * Math.PI * xn) * Math.cos((2 + (normalized * 0.5)) * Math.PI * yn)
      + (0.35 * Math.cos((3 + (normalized * 0.3)) * Math.PI * xn) * Math.cos((1 + normalized) * Math.PI * yn))
    );
  }

  const dirs = primitive === 'triangle'
    ? [
        [1, 0],
        [-0.5, Math.sqrt(3) * 0.5],
        [-0.5, -Math.sqrt(3) * 0.5],
      ]
    : [
        [1, 0],
        [0.5, Math.sqrt(3) * 0.5],
        [-0.5, Math.sqrt(3) * 0.5],
        [-1, 0],
        [-0.5, -Math.sqrt(3) * 0.5],
        [0.5, -Math.sqrt(3) * 0.5],
      ];

  const scale = 1.2 + (normalized * 0.45);
  let field = 0;
  for (const [dx, dy] of dirs) {
    field += Math.cos(Math.PI * scale * ((x * dx) + (y * dy)));
  }
  return field / Math.max(dirs.length, 1);
}

function modeField3D(primitive, x, y, z, freq) {
  const normalized = clamp(freq / 900, 0.25, 3.2);

  if (primitive === 'sphere') {
    const radius = Math.sqrt((x * x) + (y * y) + (z * z));
    const theta = Math.atan2(y, x);
    const phi = Math.atan2(z, Math.sqrt((x * x) + (y * y)) + 1e-9);
    return (
      Math.cos((2.0 + normalized) * Math.PI * radius)
      + (0.28 * Math.cos(2 * theta) * Math.cos((1.2 + (normalized * 0.35)) * Math.PI * radius))
      + (0.22 * Math.sin(2 * phi) * Math.cos((1.1 + (normalized * 0.22)) * Math.PI * radius))
    );
  }

  if (primitive === 'cube') {
    const xn = (x + 1) * 0.5;
    const yn = (y + 1) * 0.5;
    const zn = (z + 1) * 0.5;
    return (
      Math.cos((1.0 + normalized) * Math.PI * xn)
      * Math.cos((1.55 + (normalized * 0.35)) * Math.PI * yn)
      * Math.cos((2.1 + (normalized * 0.28)) * Math.PI * zn)
      + (0.26 * Math.cos((3.0 + (normalized * 0.2)) * Math.PI * (xn - zn)))
    );
  }

  const radial = Math.sqrt((x * x) + (y * y));
  const theta = Math.atan2(y, x);
  const axial = (z + 1) * 0.5;
  return (
    Math.cos((1.9 + (normalized * 0.45)) * Math.PI * radial)
    * Math.cos((1.05 + (normalized * 0.3)) * Math.PI * axial)
    + (0.3 * Math.cos(3 * theta) * Math.cos((1.35 + (normalized * 0.22)) * Math.PI * radial))
  );
}

function estimateResonancePeaks(mode, primitive, freqs) {
  const base = freqs.length ? Math.min(...freqs) : 440;
  const ratios2d = {
    circle: [0.82, 1.0, 1.34, 1.68, 2.08, 2.56],
    square: [0.71, 1.0, 1.41, 1.58, 2.0, 2.24],
    triangle: [0.76, 1.0, 1.27, 1.61, 1.96, 2.31],
    hexagon: [0.79, 1.0, 1.29, 1.52, 1.88, 2.18],
  };
  const ratios3d = {
    sphere: [0.74, 1.0, 1.31, 1.67, 1.94, 2.28],
    cube: [0.69, 1.0, 1.26, 1.55, 1.83, 2.12],
    cylinder: [0.72, 1.0, 1.22, 1.49, 1.78, 2.06],
  };
  const source = mode === '3d' ? ratios3d : ratios2d;
  return (source[primitive] || source[mode === '3d' ? 'sphere' : 'circle'])
    .map((ratio) => Math.round(base * ratio * 100) / 100);
}

function simulate2D(params, prompt, primitive, freqs) {
  const source = pointTuple(params, 'source', [-0.55, 0.0]);
  const probe = pointTuple(params, 'probe', [0.55, 0.0]);
  const obstacle = pointTuple(params, 'obs', [0.15, 0.0]);

  const xRange = buildRange(GRID_SIZE_2D);
  const yRange = buildRange(GRID_SIZE_2D);
  const signedDistance = buildSignedDistanceField2D(primitive, xRange, yRange);
  const pressure = Array.from({ length: GRID_SIZE_2D }, () => Array(GRID_SIZE_2D).fill(0));
  const cavityMask = Array.from({ length: GRID_SIZE_2D }, () => Array(GRID_SIZE_2D).fill(false));
  const boundaryWeight = Array.from({ length: GRID_SIZE_2D }, () => Array(GRID_SIZE_2D).fill(0));
  const sourceRecords = [];

  for (let row = 0; row < GRID_SIZE_2D; row += 1) {
    for (let col = 0; col < GRID_SIZE_2D; col += 1) {
      const signed = signedDistance[row][col];
      cavityMask[row][col] = signed >= 0;
      boundaryWeight[row][col] = clamp(signed / 0.22, 0, 1);
    }
  }

  freqs.forEach((freq, index) => {
    const k = (2 * Math.PI * freq) / SPEED_OF_SOUND;
    const phase = index * (Math.PI / 5);
    const [sx, sy] = source;
    const [ox, oy] = obstacle;
    const shadowAxisX = ox - sx;
    const shadowAxisY = oy - sy;
    const shadowNorm = (shadowAxisX * shadowAxisX) + (shadowAxisY * shadowAxisY);

    for (let row = 0; row < GRID_SIZE_2D; row += 1) {
      const y = yRange[row];
      for (let col = 0; col < GRID_SIZE_2D; col += 1) {
        const x = xRange[col];
        const dx = x - sx;
        const dy = y - sy;
        const dist = Math.sqrt((dx * dx) + (dy * dy));
        const direct = (Math.cos((k * dist) + phase) * Math.exp(-0.95 * dist)) / Math.sqrt(dist + 0.06);

        const modalWeight = Math.exp(-Math.abs(freq - Math.min(...freqs)) / 1200);
        const standing = modeField2D(primitive, x, y, freq) * modalWeight * Math.cos(phase);

        const obsDist = Math.sqrt(((x - ox) ** 2) + ((y - oy) ** 2));
        const obstacleCore = Math.exp(-((obsDist ** 2) / 0.012));

        let shadow = 1;
        if (shadowNorm > 1e-6) {
          const ahead = (((x - sx) * shadowAxisX) + ((y - sy) * shadowAxisY)) / shadowNorm;
          const lateral = Math.abs(((x - ox) * shadowAxisY) - ((y - oy) * shadowAxisX)) / Math.sqrt(shadowNorm);
          if (ahead > 1.0) {
            shadow = 1.0 - (0.48 * Math.exp(-((lateral ** 2) / 0.03)));
          }
        }

        const attenuation = clamp(1.0 - (0.4 * obstacleCore), 0.18, 1.0);
        const component = ((0.72 * direct) + (0.38 * standing)) * attenuation * shadow;
        pressure[row][col] += component;
      }
    }

    sourceRecords.push({
      pos: [Math.round(source[0] * 1000) / 1000, Math.round(source[1] * 1000) / 1000],
      freq: Math.round(freq * 100) / 100,
    });
  });

  const cavityValues = [];
  for (let row = 0; row < GRID_SIZE_2D; row += 1) {
    for (let col = 0; col < GRID_SIZE_2D; col += 1) {
      pressure[row][col] *= (cavityMask[row][col] ? 1 : 0) * (0.34 + (0.66 * boundaryWeight[row][col]));
      if (cavityMask[row][col]) cavityValues.push(pressure[row][col]);
    }
  }

  const mean = cavityValues.length
    ? cavityValues.reduce((sum, value) => sum + value, 0) / cavityValues.length
    : 0;
  const variance = cavityValues.length
    ? cavityValues.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / cavityValues.length
    : 1;
  const std = Math.sqrt(variance) || 1;

  let maxAbs = 0;
  let maxP = -Infinity;
  let minP = Infinity;
  let cavityFill = 0;
  for (let row = 0; row < GRID_SIZE_2D; row += 1) {
    for (let col = 0; col < GRID_SIZE_2D; col += 1) {
      const next = cavityMask[row][col] ? ((pressure[row][col] - mean) / std) : 0;
      pressure[row][col] = next;
      maxAbs = Math.max(maxAbs, Math.abs(next));
    }
  }
  maxAbs = maxAbs || 1;
  for (let row = 0; row < GRID_SIZE_2D; row += 1) {
    for (let col = 0; col < GRID_SIZE_2D; col += 1) {
      pressure[row][col] = pressure[row][col] / maxAbs;
      maxP = Math.max(maxP, pressure[row][col]);
      minP = Math.min(minP, pressure[row][col]);
      if (cavityMask[row][col]) cavityFill += 1;
      pressure[row][col] = roundFieldValue(pressure[row][col]);
    }
  }

  const probeXIdx = nearestIndex(xRange, probe[0]);
  const probeYIdx = nearestIndex(yRange, probe[1]);
  const probeResponse = pressure[probeYIdx][probeXIdx];

  return {
    status: 'success',
    solver: 'surrogate_preview',
    method: 'Live preview acoustic surrogate',
    params: {
      mode: '2d',
      primitive,
      frequencies_hz: freqs,
      freq: Number(freqs[0] || 440),
      sources: sourceRecords,
      probe: [Math.round(probe[0] * 1000) / 1000, Math.round(probe[1] * 1000) / 1000],
      obstacle: [Math.round(obstacle[0] * 1000) / 1000, Math.round(obstacle[1] * 1000) / 1000],
      grid_size: GRID_SIZE_2D,
    },
    results: {
      pressure_map: pressure,
      probe_response: probeResponse,
      mic_response: probeResponse,
      max_p: maxP,
      min_p: minP,
      cavity_fill_ratio: Math.round((cavityFill / (GRID_SIZE_2D * GRID_SIZE_2D)) * 10000) / 10000,
      resonance_peaks_hz: estimateResonancePeaks('2d', primitive, freqs),
    },
  };
}

function simulate3D(params, prompt, primitive, freqs) {
  const source = pointTuple(params, 'source', [-0.45, 0.0, -0.28]);
  const probe = pointTuple(params, 'probe', [0.42, 0.08, 0.32]);
  const obstacle = pointTuple(params, 'obs', [0.08, 0.12, 0.0]);

  const xRange = buildRange(GRID_SIZE_3D);
  const yRange = buildRange(GRID_SIZE_3D);
  const zRange = buildRange(GRID_SIZE_3D);
  const signedDistance = buildSignedDistanceField3D(primitive, xRange, yRange, zRange);
  const pressure = Array.from({ length: GRID_SIZE_3D }, () => Array.from({ length: GRID_SIZE_3D }, () => Array(GRID_SIZE_3D).fill(0)));
  const cavityMask = Array.from({ length: GRID_SIZE_3D }, () => Array.from({ length: GRID_SIZE_3D }, () => Array(GRID_SIZE_3D).fill(false)));
  const boundaryWeight = Array.from({ length: GRID_SIZE_3D }, () => Array.from({ length: GRID_SIZE_3D }, () => Array(GRID_SIZE_3D).fill(0)));
  const sourceRecords = [];

  for (let zIndex = 0; zIndex < GRID_SIZE_3D; zIndex += 1) {
    for (let row = 0; row < GRID_SIZE_3D; row += 1) {
      for (let col = 0; col < GRID_SIZE_3D; col += 1) {
        const signed = signedDistance[zIndex][row][col];
        cavityMask[zIndex][row][col] = signed >= 0;
        boundaryWeight[zIndex][row][col] = clamp(signed / 0.18, 0, 1);
      }
    }
  }

  freqs.forEach((freq, index) => {
    const k = (2 * Math.PI * freq) / SPEED_OF_SOUND;
    const phase = index * (Math.PI / 6);
    const [sx, sy, sz] = source;
    const [ox, oy, oz] = obstacle;
    const shadowAxisX = ox - sx;
    const shadowAxisY = oy - sy;
    const shadowAxisZ = oz - sz;
    const shadowNorm = (shadowAxisX * shadowAxisX) + (shadowAxisY * shadowAxisY) + (shadowAxisZ * shadowAxisZ);
    const minFreq = Math.min(...freqs);

    for (let zIndex = 0; zIndex < GRID_SIZE_3D; zIndex += 1) {
      const z = zRange[zIndex];
      for (let row = 0; row < GRID_SIZE_3D; row += 1) {
        const y = yRange[row];
        for (let col = 0; col < GRID_SIZE_3D; col += 1) {
          const x = xRange[col];
          const dx = x - sx;
          const dy = y - sy;
          const dz = z - sz;
          const dist = Math.sqrt((dx * dx) + (dy * dy) + (dz * dz));
          const direct = (Math.cos((k * dist) + phase) * Math.exp(-1.08 * dist)) / ((dist + 0.05) ** 0.68);

          const modalWeight = Math.exp(-Math.abs(freq - minFreq) / 1350);
          const standing = modeField3D(primitive, x, y, z, freq) * modalWeight * Math.cos(phase);

          const obsDx = x - ox;
          const obsDy = y - oy;
          const obsDz = z - oz;
          const obsDist = Math.sqrt((obsDx * obsDx) + (obsDy * obsDy) + (obsDz * obsDz));
          const obstacleCore = Math.exp(-((obsDist ** 2) / 0.022));

          let shadow = 1;
          if (shadowNorm > 1e-6) {
            const ahead = (((x - sx) * shadowAxisX) + ((y - sy) * shadowAxisY) + ((z - sz) * shadowAxisZ)) / shadowNorm;
            const crossX = (obsDy * shadowAxisZ) - (obsDz * shadowAxisY);
            const crossY = (obsDz * shadowAxisX) - (obsDx * shadowAxisZ);
            const crossZ = (obsDx * shadowAxisY) - (obsDy * shadowAxisX);
            const lateral = Math.sqrt((crossX * crossX) + (crossY * crossY) + (crossZ * crossZ)) / Math.sqrt(shadowNorm);
            if (ahead > 1.0) {
              shadow = 1.0 - (0.52 * Math.exp(-((lateral ** 2) / 0.05)));
            }
          }

          const attenuation = clamp(1.0 - (0.44 * obstacleCore), 0.16, 1.0);
          const component = ((0.7 * direct) + (0.44 * standing)) * attenuation * shadow;
          pressure[zIndex][row][col] += component;
        }
      }
    }

    sourceRecords.push({
      pos: [
        Math.round(source[0] * 1000) / 1000,
        Math.round(source[1] * 1000) / 1000,
        Math.round(source[2] * 1000) / 1000,
      ],
      freq: Math.round(freq * 100) / 100,
    });
  });

  const cavityValues = [];
  for (let zIndex = 0; zIndex < GRID_SIZE_3D; zIndex += 1) {
    for (let row = 0; row < GRID_SIZE_3D; row += 1) {
      for (let col = 0; col < GRID_SIZE_3D; col += 1) {
        pressure[zIndex][row][col] *= (cavityMask[zIndex][row][col] ? 1 : 0) * (0.3 + (0.7 * boundaryWeight[zIndex][row][col]));
        if (cavityMask[zIndex][row][col]) cavityValues.push(pressure[zIndex][row][col]);
      }
    }
  }

  const mean = cavityValues.length
    ? cavityValues.reduce((sum, value) => sum + value, 0) / cavityValues.length
    : 0;
  const variance = cavityValues.length
    ? cavityValues.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / cavityValues.length
    : 1;
  const std = Math.sqrt(variance) || 1;

  let maxAbs = 0;
  let maxP = -Infinity;
  let minP = Infinity;
  let cavityFill = 0;
  for (let zIndex = 0; zIndex < GRID_SIZE_3D; zIndex += 1) {
    for (let row = 0; row < GRID_SIZE_3D; row += 1) {
      for (let col = 0; col < GRID_SIZE_3D; col += 1) {
        const next = cavityMask[zIndex][row][col] ? ((pressure[zIndex][row][col] - mean) / std) : 0;
        pressure[zIndex][row][col] = next;
        maxAbs = Math.max(maxAbs, Math.abs(next));
      }
    }
  }
  maxAbs = maxAbs || 1;
  for (let zIndex = 0; zIndex < GRID_SIZE_3D; zIndex += 1) {
    for (let row = 0; row < GRID_SIZE_3D; row += 1) {
      for (let col = 0; col < GRID_SIZE_3D; col += 1) {
        pressure[zIndex][row][col] = roundFieldValue(pressure[zIndex][row][col] / maxAbs);
        maxP = Math.max(maxP, pressure[zIndex][row][col]);
        minP = Math.min(minP, pressure[zIndex][row][col]);
        if (cavityMask[zIndex][row][col]) cavityFill += 1;
      }
    }
  }

  const probeXIdx = nearestIndex(xRange, probe[0]);
  const probeYIdx = nearestIndex(yRange, probe[1]);
  const probeZIdx = nearestIndex(zRange, probe[2]);
  const probeResponse = pressure[probeZIdx][probeYIdx][probeXIdx];

  const sliceXY = pressure[probeZIdx].map((row) => row.slice());
  const sliceXZ = zRange.map((_, zIndex) => xRange.map((__, xIndex) => pressure[zIndex][probeYIdx][xIndex]));
  const sliceYZ = zRange.map((_, zIndex) => yRange.map((__, yIndex) => pressure[zIndex][yIndex][probeXIdx]));

  return {
    status: 'success',
    solver: 'surrogate_preview',
    method: 'Live preview volumetric acoustic surrogate',
    params: {
      mode: '3d',
      primitive,
      frequencies_hz: freqs,
      freq: Number(freqs[0] || 440),
      sources: sourceRecords,
      probe: [
        Math.round(probe[0] * 1000) / 1000,
        Math.round(probe[1] * 1000) / 1000,
        Math.round(probe[2] * 1000) / 1000,
      ],
      obstacle: [
        Math.round(obstacle[0] * 1000) / 1000,
        Math.round(obstacle[1] * 1000) / 1000,
        Math.round(obstacle[2] * 1000) / 1000,
      ],
      grid_size: GRID_SIZE_3D,
      grid_shape: [GRID_SIZE_3D, GRID_SIZE_3D, GRID_SIZE_3D],
      slice_indices: { x: probeXIdx, y: probeYIdx, z: probeZIdx },
    },
    results: {
      pressure_map: sliceXY,
      pressure_volume: pressure,
      slice_xy: sliceXY,
      slice_xz: sliceXZ,
      slice_yz: sliceYZ,
      probe_response: probeResponse,
      mic_response: probeResponse,
      max_p: maxP,
      min_p: minP,
      cavity_fill_ratio: Math.round((cavityFill / (GRID_SIZE_3D ** 3)) * 10000) / 10000,
      resonance_peaks_hz: estimateResonancePeaks('3d', primitive, freqs),
    },
  };
}

export function runAcousticSimulation(params = {}) {
  const prompt = String(params.prompt || '');
  const mode = normalizeMode(params.mode, params.primitive);
  const primitive = normalizePrimitive(mode, params.primitive);
  const freqs = extractFrequencies(prompt, params);

  if (mode === '3d') {
    return simulate3D(params, prompt, primitive, freqs);
  }
  return simulate2D(params, prompt, primitive, freqs);
}
