import { computed, ref } from 'vue';
import { useApi } from '~/composables/useApi';
import {
  estimateMs, isActive, shortLabel,
  type ProcessKind, type ProcessStatus,
} from '~/composables/processMath';

// One list of running and recent processes (render, sound, sketch, 3D, acoustics) for the whole
// app. Flows register a process under their backend request id; one poller reads the backend
// stage and reasoning line of every active process; the dock renders the list.

export interface ProcessRow {
  id: string;
  kind: ProcessKind;
  label: string;
  model: string;
  startedAt: number;
  finishedAt: number | null;
  status: ProcessStatus;
  stage: string;
  preview: string;
  /** A message from the dock itself (a failed cancel), shown over the backend's reasoning line. */
  notice: string;
  estimateMs: number;
  error: string;
  cancellable: boolean;
}

export interface StartProcess {
  id: string;
  kind: ProcessKind;
  label: string;
  model?: string;
  estimateMs?: number;
  /** False for jobs the backend cannot stop (no request id in its progress store). */
  cancellable?: boolean;
}

const HISTORY_KEY = 'soog.process-times.v1';
const HISTORY_LIMIT = 20;
const LINGER_MS = 6000;
const POLL_MS = 800;

function readHistory(): Record<string, number[]> {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '{}') || {};
  } catch {
    return {};
  }
}

function writeHistory(history: Record<string, number[]>) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {
    // storage can be blocked; estimates then fall back to the defaults
  }
}

function createProcesses() {
  const { apiBase } = useApi();
  const { authHeaders } = useSoogAuth();

  const processes = ref<ProcessRow[]>([]);
  const now = ref(Date.now());
  let pollTimer: ReturnType<typeof setInterval> | null = null;
  const lingerTimers = new Map<string, ReturnType<typeof setTimeout>>();

  const activeCount = computed(() => processes.value.filter((row) => isActive(row.status)).length);

  function patch(id: string, changes: Partial<ProcessRow>) {
    processes.value = processes.value.map((row) => (row.id === id ? { ...row, ...changes } : row));
  }

  function find(id: string) {
    return processes.value.find((row) => row.id === id);
  }

  async function pollOne(row: ProcessRow) {
    try {
      const res = await fetch(`${apiBase.value}/generate/progress/${row.id}`, { headers: { Accept: 'application/json' } });
      if (!res.ok) return;
      const payload = await res.json();
      if (!payload?.ok) return;
      const current = find(row.id);
      if (!current || !isActive(current.status)) return;
      patch(row.id, {
        stage: String(payload.stage || '').trim(),
        preview: String(payload.reasoning_preview || '').trim(),
        model: current.model || String(payload.model || '').trim(),
      });
    } catch {
      // keep polling quiet; the owning flow reports real failures
    }
  }

  function tick() {
    now.value = Date.now();
    const active = processes.value.filter((row) => isActive(row.status));
    if (!active.length) {
      stopTimer();
      return;
    }
    active.forEach((row) => { void pollOne(row); });
  }

  function stopTimer() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
  }

  function ensureTimer() {
    if (typeof window === 'undefined' || pollTimer) return;
    pollTimer = setInterval(tick, POLL_MS);
  }

  function historyKey(kind: ProcessKind, model: string) {
    return `${kind}|${model || '-'}`;
  }

  function startProcess(input: StartProcess) {
    if (typeof window === 'undefined' || !input.id) return;
    const model = input.model || '';
    const typical = input.estimateMs && input.estimateMs > 0
      ? input.estimateMs
      : estimateMs(input.kind, readHistory()[historyKey(input.kind, model)] || []);
    const row: ProcessRow = {
      id: input.id,
      kind: input.kind,
      label: shortLabel(input.label),
      model,
      startedAt: Date.now(),
      finishedAt: null,
      status: 'running',
      stage: '',
      preview: '',
      notice: '',
      estimateMs: typical,
      error: '',
      cancellable: input.cancellable !== false,
    };
    processes.value = [...processes.value.filter((existing) => existing.id !== row.id), row];
    now.value = Date.now();
    ensureTimer();
  }

  function dismissProcess(id: string) {
    const timer = lingerTimers.get(id);
    if (timer) clearTimeout(timer);
    lingerTimers.delete(id);
    processes.value = processes.value.filter((row) => row.id !== id);
  }

  function finishProcess(id: string, status: Exclude<ProcessStatus, 'running' | 'cancelling'>, error = '') {
    const row = find(id);
    if (!row) return;
    const finishedAt = Date.now();
    patch(id, { status, finishedAt, error, preview: status === 'error' ? error : '' });
    if (status === 'completed') {
      const history = readHistory();
      const key = historyKey(row.kind, row.model);
      history[key] = [...(history[key] || []), finishedAt - row.startedAt].slice(-HISTORY_LIMIT);
      writeHistory(history);
    }
    if (status !== 'error') {
      lingerTimers.set(id, setTimeout(() => dismissProcess(id), LINGER_MS));
    }
  }

  /** Ask the backend to stop the job; it stops at its next progress update. */
  async function cancelProcess(id: string) {
    const row = find(id);
    if (!row || row.status !== 'running' || !row.cancellable) return;
    patch(id, { status: 'cancelling', notice: '' });
    try {
      const res = await fetch(`${apiBase.value}/generate/cancel/${id}`, {
        method: 'POST',
        headers: { Accept: 'application/json', ...(await authHeaders()) },
      });
      if (!res.ok) throw new Error(res.status === 404 ? 'The server does not know this job yet' : `Cancel failed (${res.status})`);
    } catch (error) {
      patch(id, { status: 'running', notice: `Cancel failed: ${error instanceof Error ? error.message : String(error)}` });
    }
  }

  return { processes, now, activeCount, startProcess, finishProcess, cancelProcess, dismissProcess };
}

let shared: ReturnType<typeof createProcesses> | null = null;

/** One process list for the whole app; call it from component setup. */
export function useProcesses() {
  if (!shared) shared = createProcesses();
  return shared;
}
