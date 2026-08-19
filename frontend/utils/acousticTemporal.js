const TAU = Math.PI * 2;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function normalizePlaneKey(plane = 'xy') {
  const value = String(plane || 'xy').trim().toLowerCase();
  return value === 'xz' || value === 'yz' ? value : 'xy';
}

export function planeAxes(plane = 'xy') {
  const key = normalizePlaneKey(plane);
  if (key === 'xz') return ['x', 'z'];
  if (key === 'yz') return ['y', 'z'];
  return ['x', 'y'];
}

export function markerAxisValue(marker, axis) {
  const numeric = Number(marker?.[axis]);
  return Number.isFinite(numeric) ? clamp(numeric, -1, 1) : 0;
}

export function projectMarkerToPlane(marker, plane = 'xy') {
  const [uAxis, vAxis] = planeAxes(plane);
  return {
    u: markerAxisValue(marker, uAxis),
    v: markerAxisValue(marker, vAxis),
  };
}

export function findSourceMarker(markers = []) {
  if (!Array.isArray(markers)) return null;
  return markers.find((marker) => marker?.type === 'source') || markers[0] || null;
}

export function normalizedCol(col, cols) {
  return ((col / Math.max(cols - 1, 1)) * 2) - 1;
}

export function normalizedRow(row, rows) {
  return 1 - ((row / Math.max(rows - 1, 1)) * 2);
}

export function phaseDistance2d(row, col, rows, cols, markers = [], plane = 'xy') {
  const source = projectMarkerToPlane(findSourceMarker(markers), plane);
  const x = normalizedCol(col, cols);
  const y = normalizedRow(row, rows);
  return Math.hypot(x - source.u, y - source.v);
}

export function phaseDistance3d(x, y, z, markers = []) {
  const source = findSourceMarker(markers);
  const sx = markerAxisValue(source, 'x');
  const sy = markerAxisValue(source, 'y');
  const sz = markerAxisValue(source, 'z');
  return Math.hypot(x - sx, y - sy, z - sz);
}

export function temporalFieldValue(value, phase = 0, distance = 0) {
  const numeric = Number(value) || 0;
  if (!numeric) return 0;
  const phaseShift = Number(phase || 0) - distance * 7.35;
  const carrier = 0.72 + 0.28 * Math.sin(phaseShift);
  const shimmer = 0.08 * Math.cos((Number(phase || 0) * 0.68) + distance * 4.15);
  return numeric * (carrier + shimmer);
}

export function temporalBandWeight(value, phase = 0, distance = 0) {
  const shifted = temporalFieldValue(value, phase, distance);
  return Math.abs(shifted);
}

export function wrapPhase(value) {
  const numeric = Number(value) || 0;
  const wrapped = numeric % TAU;
  return wrapped < 0 ? wrapped + TAU : wrapped;
}

export function phaseTurnsLabel(value) {
  return `${(wrapPhase(value) / Math.PI).toFixed(2)}pi`;
}
