import { computed, ref } from 'vue';
import { useApi } from '~/composables/useApi';
import { useProcesses } from '~/composables/useProcesses';
import type { ProcessKind } from '~/composables/processMath';

// Per-item tools (regenerate sketch, build 3D, acoustics, sound samples) shared by the gallery
// dock and the workspace pods. Each tool works on a gallery basename, so it applies to whatever
// item the pods are showing whether or not the gallery is open.

export interface ToolItem {
  basename: string;
  prompt?: string;
}

type ItemListener = (item: Record<string, any>) => void;

async function readJsonSafe(res: Response) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    return { error: `Server returned ${res.status} ${res.statusText || ''} (not JSON). The request may have timed out at the proxy.`.trim() };
  }
}

class CancelledError extends Error {
  constructor() {
    super('Cancelled');
  }
}

function createItemTools() {
  const { apiBase } = useApi();
  const { authHeaders } = useSoogAuth();

  const remakingSketch = ref(false);
  const generatingSound = ref(false);
  const remakingModulus = ref(false);
  const remakingLRM = ref(false);
  const toolError = ref('');
  const { startProcess, finishProcess } = useProcesses();

  const soundModalOpen = ref(false);
  const soundPrompt = ref('');
  let soundTarget: ToolItem | null = null;

  const playingIdx = ref<number | null>(null);
  const currentTime = ref(0);
  const duration = ref(0);
  let audioObj: HTMLAudioElement | null = null;

  const listeners = new Set<ItemListener>();
  const isBusy = computed(() => (
    generatingSound.value || remakingSketch.value || remakingModulus.value || remakingLRM.value
  ));

  function onItemUpdated(listener: ItemListener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  function assetHref(url: string) {
    if (!url) return '';
    if (url.startsWith('http')) return url;
    const base = apiBase.value;
    const offloadApiBase = base.endsWith('/api') ? base.slice(0, -4) : base;
    if (url.startsWith('/offload')) return offloadApiBase + url;
    if (base.endsWith('/api') && url.startsWith('/api/')) return base + url.substring(4);
    return base + url;
  }

  // Reload the item from the gallery list and tell every listener (the pods, the dock).
  async function refreshItem(basename: string) {
    const res = await fetch(`${apiBase.value}/gallery/list`);
    const data = await res.json();
    const item = (data.items || []).find((entry: Record<string, any>) => entry.basename === basename);
    if (item) listeners.forEach((listener) => listener({ ...item }));
    return item || null;
  }

  const newRequestId = () => `gallery-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

  interface ToolMeta { kind: ProcessKind; model: string; what: string }

  async function runTool(flag: { value: boolean }, item: ToolItem | null, meta: ToolMeta, job: (id: string) => Promise<void>) {
    if (!item?.basename || flag.value) return;
    flag.value = true;
    toolError.value = '';
    const id = newRequestId();
    startProcess({ id, kind: meta.kind, label: `${meta.what} ${item.basename}`, model: meta.model });
    try {
      await job(id);
      finishProcess(id, 'completed');
      await refreshItem(item.basename);
    } catch (error) {
      if (error instanceof CancelledError) {
        finishProcess(id, 'cancelled');
      } else {
        const message = error instanceof Error ? error.message : String(error);
        toolError.value = message;
        finishProcess(id, 'error', message);
      }
    } finally {
      flag.value = false;
    }
  }

  const post = async (path: string, body: Record<string, unknown>) => fetch(`${apiBase.value}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(await authHeaders()) },
    body: JSON.stringify(body),
  });

  const remakeSketch = (item: ToolItem | null) => runTool(remakingSketch, item, { kind: 'sketch', model: '', what: 'sketch' }, async (id) => {
    const res = await post(`/gallery/item/${item!.basename}/remake_sketch`, { request_id: id });
    if (res.status === 499) throw new CancelledError();
    if (!res.ok) throw new Error('Failed to remake sketch');
  });

  const buildMesh = (item: ToolItem | null) => runTool(remakingLRM, item, { kind: 'mesh', model: 'TripoSR', what: '3D' }, async (id) => {
    const res = await post(`/gallery/item/${item!.basename}/generate_lrm`, { request_id: id });
    if (res.status === 499) throw new CancelledError();
    if (!res.ok) {
      const payload = await readJsonSafe(res);
      throw new Error(payload.error || 'Failed to generate high-res mesh');
    }
  });

  const calcAcoustics = (item: ToolItem | null) => runTool(remakingModulus, item, { kind: 'acoustics', model: 'SOT-A solver', what: 'acoustics' }, async (id) => {
    const res = await post('/generate', {
      prompt: `[MODULUS] ${item!.prompt || ''}`,
      request_id: id,
      gallery_update: item!.basename,
    });
    if (res.status === 499) throw new CancelledError();
    if (!res.ok) throw new Error('Failed to run simulation');
  });

  async function waitForSoundJob(id: string, timeoutMs = 40 * 60 * 1000) {
    const started = Date.now();
    while (Date.now() - started < timeoutMs) {
      await new Promise((resolve) => setTimeout(resolve, 3000));
      try {
        const res = await fetch(`${apiBase.value}/generate/progress/${id}`, { headers: { Accept: 'application/json' } });
        if (!res.ok) continue;
        const payload = await readJsonSafe(res);
        if (payload?.status === 'completed') return;
        if (payload?.status === 'cancelled') throw new CancelledError();
        if (payload?.status === 'error') throw new Error(payload.reasoning_preview || 'Sound generation failed');
      } catch (error) {
        if (error instanceof CancelledError) throw error;
        if (error instanceof Error && error.message && !/fetch|network|Failed/i.test(error.message)) throw error;
      }
    }
    throw new Error('Sound generation timed out');
  }

  function openSoundGenerator(item: ToolItem | null) {
    if (!item?.basename) return;
    soundTarget = item;
    soundPrompt.value = item.prompt || '';
    soundModalOpen.value = true;
  }

  function closeSoundGenerator() {
    soundModalOpen.value = false;
  }

  async function runSoundGenerator() {
    const item = soundTarget;
    soundModalOpen.value = false;
    await runTool(generatingSound, item, { kind: 'sound', model: '', what: 'sound' }, async (id) => {
      const res = await post(`/gallery/item/${item!.basename}/generate_sound`, { request_id: id, prompt: soundPrompt.value });
      const payload = await readJsonSafe(res);
      if (!res.ok) throw new Error(payload.error || 'Failed to generate sound');
      if (res.status === 202) await waitForSoundJob(id);
    });
  }

  function stopAudio() {
    audioObj?.pause();
    playingIdx.value = null;
  }

  function toggleAudio(sample: Record<string, any>, idx: number) {
    if (playingIdx.value === idx) {
      stopAudio();
      return;
    }
    if (audioObj) audioObj.pause();
    playingIdx.value = idx;
    const audio = new Audio(assetHref(sample.ogg_url || sample.url));
    audioObj = audio;
    audio.addEventListener('timeupdate', () => { currentTime.value = audio.currentTime; });
    audio.addEventListener('loadedmetadata', () => { duration.value = audio.duration; });
    audio.addEventListener('ended', () => { playingIdx.value = null; });
    audio.play();
  }

  return {
    remakingSketch, generatingSound, remakingModulus, remakingLRM, isBusy,
    toolError,
    soundModalOpen, soundPrompt, openSoundGenerator, closeSoundGenerator, runSoundGenerator,
    playingIdx, currentTime, duration, toggleAudio, stopAudio,
    remakeSketch, buildMesh, calcAcoustics, onItemUpdated, assetHref,
  };
}

let shared: ReturnType<typeof createItemTools> | null = null;

/** One set of tool state for the whole app; call it from component setup. */
export function useItemTools() {
  if (!shared) shared = createItemTools();
  return shared;
}
