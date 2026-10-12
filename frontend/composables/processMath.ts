// Pure helpers behind the process dock: progress estimate, clock text, history medians.

export type ProcessKind = 'render' | 'sound' | 'sketch' | 'mesh' | 'acoustics';
export type ProcessStatus = 'running' | 'cancelling' | 'completed' | 'error' | 'cancelled';

export const DEFAULT_ESTIMATE_MS: Record<ProcessKind, number> = {
  render: 180000,
  sound: 240000,
  sketch: 90000,
  mesh: 120000,
  acoustics: 45000,
};

export const KIND_LABEL: Record<ProcessKind, string> = {
  render: 'RENDER',
  sound: 'SOUND',
  sketch: 'SKETCH',
  mesh: '3D',
  acoustics: 'ACOUSTICS',
};

export function median(values: number[]): number {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Typical duration from past runs of the same kind and model, else the default for the kind. */
export function estimateMs(kind: ProcessKind, history: number[]): number {
  const typical = median(history || []);
  return Number.isFinite(typical) && typical > 0 ? typical : DEFAULT_ESTIMATE_MS[kind];
}

/** Estimated progress in percent: reaches 92% at the typical duration, then creeps towards 98%. */
export function estimateProgress(elapsedMs: number, typicalMs: number): number {
  const avg = Math.max(3000, typicalMs || 0);
  const ratio = Math.max(0, elapsedMs) / avg;
  if (ratio <= 1) return Math.min(92, ratio * 92);
  return Math.min(98, 92 + Math.log1p(ratio - 1) * 6);
}

/** 7s, 1m05s, 12m00s */
export function formatClock(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '—';
  const total = Math.round(ms / 1000);
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  return `${minutes}m${String(total % 60).padStart(2, '0')}s`;
}

export function isActive(status: ProcessStatus): boolean {
  return status === 'running' || status === 'cancelling';
}

export function shortLabel(text: string, max = 42): string {
  const flat = String(text || '').replace(/\s+/g, ' ').trim();
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
}
