// Pure helpers behind the SESSION panel: stage rows from backend events, and server load facts.

export interface StageEvent {
  stage: string;
  status?: string;
  at?: number;
  t_ms: number;
  cpu_pct?: number | null;
}

export interface ProgressInfo {
  status: string;
  started_at: number;
  updated_at: number;
}

export interface StageRow {
  key: string;
  label: string;
  running: boolean;
  duration: string;
  cpu: number | null;
}

export interface ServerMetrics {
  cpu_pct?: number | null;
  cpus?: number;
  load1?: number;
  load5?: number;
  load15?: number;
  mem_total_mb?: number;
  mem_used_mb?: number;
  swap_total_mb?: number;
  swap_used_mb?: number;
  process_rss_mb?: number;
}

export function formatStageLabel(stage: string): string {
  return String(stage || '')
    .replace(/[_-]+/g, ' ')
    .trim()
    .replace(/\b\w/g, (ch) => ch.toUpperCase())
    .replace(/\b(Llm|Stl|Ogg|Lrm|Sot)\b/g, (word) => word.toUpperCase());
}

export function formatSeconds(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '—';
  return ms < 10000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms / 1000)}s`;
}

/** One row per stage: duration is the gap to the next stage; the last stage runs until the request ends. */
export function buildStageRows(events: StageEvent[], info: ProgressInfo): StageRow[] {
  const spanMs = Math.max(0, (Number(info.updated_at) - Number(info.started_at)) * 1000);
  const finished = ['completed', 'error'].includes(info.status);
  return (events || []).map((event, index, all) => {
    const next = all[index + 1];
    const running = index === all.length - 1 && !finished;
    const cpu = event.cpu_pct === null || event.cpu_pct === undefined ? NaN : Number(event.cpu_pct);
    return {
      key: `${event.stage}-${event.t_ms}`,
      label: formatStageLabel(event.stage),
      running,
      duration: running ? '…' : formatSeconds((next ? next.t_ms : spanMs) - event.t_ms),
      cpu: Number.isFinite(cpu) ? Math.max(0, Math.min(100, cpu)) : null,
    };
  });
}

const toGb = (mb: number, digits = 1) => (mb / 1024).toFixed(digits);

export function buildServerFacts(m: ServerMetrics | null): { label: string; value: string }[] {
  if (!m) return [];
  const facts: { label: string; value: string }[] = [];
  if (m.cpu_pct != null) facts.push({ label: 'CPU', value: `${m.cpu_pct}% of ${m.cpus} cores` });
  if (m.load1 != null) facts.push({ label: 'Load 1/5/15', value: `${m.load1} / ${m.load5} / ${m.load15}` });
  if (m.mem_total_mb) facts.push({ label: 'Memory', value: `${toGb(m.mem_used_mb || 0)} / ${toGb(m.mem_total_mb)} GB` });
  if (m.swap_total_mb) facts.push({ label: 'Swap', value: `${toGb(m.swap_used_mb || 0)} / ${toGb(m.swap_total_mb)} GB` });
  if (m.process_rss_mb != null) facts.push({ label: 'Backend RSS', value: `${toGb(m.process_rss_mb, 2)} GB` });
  return facts;
}
