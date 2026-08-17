const SPEED_OF_SOUND = 343.0;
const GRID_SIZE = 84;
const PRIMITIVES = new Set(['circle', 'square', 'triangle', 'hexagon']);

function safeFloat(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function pointPair(params, prefix, fallbackX, fallbackY) {
  return [
    clamp(safeFloat(params[`${prefix}_x`], fallbackX), -0.95, 0.95),
    clamp(safeFloat(params[`${prefix}_y`], fallbackY), -0.95, 0.95),
  ];
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

function buildSignedDistanceField(primitive, xGrid, yGrid) {
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

function modeField(primitive, x, y, freq) {
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

function estimateResonancePeaks(primitive, freqs) {
  const base = freqs.length ? Math.min(...freqs) : 440;
  const ratios = {
    circle: [0.82, 1.0, 1.34, 1.68, 2.08, 2.56],
    square: [0.71, 1.0, 1.41, 1.58, 2.0, 2.24],
    triangle: [0.76, 1.0, 1.27, 1.61, 1.96, 2.31],
    hexagon: [0.79, 1.0, 1.29, 1.52, 1.88, 2.18],
  };
  return (ratios[primitive] || ratios.circle).map((ratio) => Math.round(base * ratio * 100) / 100);
}

export function runAcousticSimulation(params = {}) {
  const prompt = String(params.prompt || '');
  const primitive = PRIMITIVES.has(String(params.primitive || 'circle').toLowerCase())
    ? String(params.primitive || 'circle').toLowerCase()
    : 'circle';

  const freqs = extractFrequencies(prompt, params);
  const source = pointPair(params, 'source', -0.55, 0.0);
  const probe = pointPair(params, 'probe', 0.55, 0.0);
  const obstacle = pointPair(params, 'obs', 0.15, 0.0);

  const xRange = Array.from({ length: GRID_SIZE }, (_, index) => -1 + ((2 * index) / (GRID_SIZE - 1)));
  const yRange = Array.from({ length: GRID_SIZE }, (_, index) => -1 + ((2 * index) / (GRID_SIZE - 1)));
  const signedDistance = buildSignedDistanceField(primitive, xRange, yRange);
  const pressure = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
  const cavityMask = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(false));
  const boundaryWeight = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
  const sourceRecords = [];

  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
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

    for (let row = 0; row < GRID_SIZE; row += 1) {
      const y = yRange[row];
      for (let col = 0; col < GRID_SIZE; col += 1) {
        const x = xRange[col];
        const dx = x - sx;
        const dy = y - sy;
        const dist = Math.sqrt((dx * dx) + (dy * dy));
        const direct = (Math.cos((k * dist) + phase) * Math.exp(-0.95 * dist)) / Math.sqrt(dist + 0.06);

        const modalWeight = Math.exp(-Math.abs(freq - Math.min(...freqs)) / 1200);
        const standing = modeField(primitive, x, y, freq) * modalWeight * Math.cos(phase);

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
  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
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
  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      const next = cavityMask[row][col] ? ((pressure[row][col] - mean) / std) : 0;
      pressure[row][col] = next;
      maxAbs = Math.max(maxAbs, Math.abs(next));
    }
  }
  maxAbs = maxAbs || 1;

  let maxP = -Infinity;
  let minP = Infinity;
  let cavityFill = 0;
  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      pressure[row][col] = pressure[row][col] / maxAbs;
      maxP = Math.max(maxP, pressure[row][col]);
      minP = Math.min(minP, pressure[row][col]);
      if (cavityMask[row][col]) cavityFill += 1;
    }
  }

  const probeXIdx = xRange.reduce((best, value, index) => (
    Math.abs(value - probe[0]) < Math.abs(xRange[best] - probe[0]) ? index : best
  ), 0);
  const probeYIdx = yRange.reduce((best, value, index) => (
    Math.abs(value - probe[1]) < Math.abs(yRange[best] - probe[1]) ? index : best
  ), 0);
  const probeResponse = pressure[probeYIdx][probeXIdx];

  return {
    status: 'success',
    method: 'Primitive cavity acoustic surrogate',
    params: {
      primitive,
      frequencies_hz: freqs,
      freq: Number(freqs[0] || 440),
      sources: sourceRecords,
      probe: [Math.round(probe[0] * 1000) / 1000, Math.round(probe[1] * 1000) / 1000],
      obstacle: [Math.round(obstacle[0] * 1000) / 1000, Math.round(obstacle[1] * 1000) / 1000],
      grid_size: GRID_SIZE,
    },
    results: {
      pressure_map: pressure.map((row) => row.map((value) => Math.round(value * 1000000) / 1000000)),
      probe_response: probeResponse,
      mic_response: probeResponse,
      max_p: maxP,
      min_p: minP,
      cavity_fill_ratio: Math.round((cavityFill / (GRID_SIZE * GRID_SIZE)) * 10000) / 10000,
      resonance_peaks_hz: estimateResonancePeaks(primitive, freqs),
    },
  };
}
