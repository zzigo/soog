import { computed, reactive, ref } from 'vue';

export interface WorkspaceView {
  key: string;
  label: string;
}

export interface WorkspacePreset {
  label: string;
  summary: string;
  areas: string[];
  defaults: string[];
}

export const WORKSPACE_PRESETS: Record<string, WorkspacePreset> = {
  fullscreen: {
    label: 'Fullscreen',
    summary: 'Single immersive viewport with routed render focus.',
    areas: ['a'],
    defaults: ['auto'],
  },
  splitVertical: {
    label: 'Split Vertical',
    summary: 'Two equal panes for paired visual inspection.',
    areas: ['a', 'b'],
    defaults: ['organogram', 'geometry'],
  },
  splitThree: {
    label: 'Split 3',
    summary: 'Main left viewport with two stacked analytical panes.',
    areas: ['a', 'b', 'c'],
    defaults: ['organogram', 'sound', 'acoustic'],
  },
  splitFour: {
    label: 'Split 4',
    summary: 'Four-way dock for organogram, text, object, and field.',
    areas: ['a', 'b', 'c', 'd'],
    defaults: ['organogram', 'sound', 'geometry', 'acoustic'],
  },
};

const CORE_VIEWS: WorkspaceView[] = [
  { key: 'auto', label: 'Auto Route' },
  { key: 'organogram', label: 'Organogram' },
  { key: 'sound', label: 'Sound' },
  { key: 'geometry', label: '3D Object' },
  { key: 'acoustic', label: 'Acoustic' },
  { key: 'sketch', label: 'Sketch' },
];

export const PANE_SPLIT_MIN = 15;
export const PANE_SPLIT_MAX = 85;

/**
 * Workspace state: layout preset, which view each pane shows, and the pane divider position.
 * The view catalog is a registry: new pane types (gallery item, stage, SOT-A, ...) call
 * registerView and are then selectable in every pane and reachable through showView.
 */
export function createWorkspace() {
  const preset = ref('fullscreen');
  const layout = ref('background'); // 'background' (fullscreen) or 'split'
  const presetMenuOpen = ref(false);
  const presetMenuRef = ref<HTMLElement | null>(null);
  const autoView = ref('organogram');
  const paneViews = ref<string[]>(['auto', 'organogram', 'sound', 'acoustic']);
  const views = ref<WorkspaceView[]>([...CORE_VIEWS]);
  const stageRef = ref<HTMLElement | null>(null);
  const paneSplit = reactive({ x: 50, y: 50 });

  const presetOptions = computed(() => Object.entries(WORKSPACE_PRESETS).map(([key, value]) => ({
    key,
    label: value.label,
    summary: value.summary,
  })));
  const viewOptions = computed(() => views.value);
  const activePreset = computed(() => WORKSPACE_PRESETS[preset.value] || WORKSPACE_PRESETS.fullscreen);
  const modeLabel = computed(() => activePreset.value.label);
  const panes = computed(() => activePreset.value.areas.map((area, index) => {
    const assigned = paneViews.value[index] || activePreset.value.defaults[index] || 'auto';
    const view = assigned === 'auto' ? autoView.value : assigned;
    return { id: `${preset.value}-${area}-${index}`, area, index, assigned, view };
  }));
  const showHandleX = computed(() => ['splitVertical', 'splitThree', 'splitFour'].includes(preset.value));
  const showHandleY = computed(() => ['splitThree', 'splitFour'].includes(preset.value));
  const stageStyle = computed(() => (
    preset.value === 'fullscreen'
      ? {}
      : { '--pane-x': `${paneSplit.x}%`, '--pane-y': `${paneSplit.y}%` }
  ));

  function viewLabel(key: string) {
    return views.value.find((item) => item.key === key)?.label || String(key || 'Pane');
  }

  function registerView(view: WorkspaceView) {
    if (!view?.key || views.value.some((item) => item.key === view.key)) return;
    views.value = [...views.value, view];
  }

  function resetSplit() {
    paneSplit.x = 50;
    paneSplit.y = 50;
  }

  function toggleMenu() {
    presetMenuOpen.value = !presetMenuOpen.value;
  }

  function setPreset(presetKey: string) {
    const next = WORKSPACE_PRESETS[presetKey];
    if (!next) return;
    preset.value = presetKey;
    resetSplit();
    paneViews.value = [...next.defaults];
    layout.value = presetKey === 'fullscreen' ? 'background' : 'split';
    presetMenuOpen.value = false;
  }

  function setPaneView(index: number, viewKey: string) {
    const next = [...paneViews.value];
    next[index] = viewKey;
    paneViews.value = next;
  }

  /**
   * Make a registered view visible: keep it when a pane already shows it, otherwise
   * put it in the pane that follows the auto route, or in the last pane.
   */
  function showView(viewKey: string) {
    if (!views.value.some((item) => item.key === viewKey)) return false;
    if (panes.value.some((pane) => pane.view === viewKey)) return true;
    const auto = panes.value.find((pane) => pane.assigned === 'auto');
    const target = auto ? auto.index : panes.value.length - 1;
    setPaneView(target, viewKey);
    return true;
  }

  function startDrag(axis: 'x' | 'y', event: PointerEvent) {
    const stage = stageRef.value;
    if (!stage) return;
    const handle = event.currentTarget as HTMLElement;
    handle.setPointerCapture?.(event.pointerId);
    const move = (e: PointerEvent) => {
      const rect = stage.getBoundingClientRect();
      if (axis === 'x') {
        paneSplit.x = Math.min(PANE_SPLIT_MAX, Math.max(PANE_SPLIT_MIN, ((e.clientX - rect.left) / rect.width) * 100));
      } else {
        paneSplit.y = Math.min(PANE_SPLIT_MAX, Math.max(PANE_SPLIT_MIN, ((e.clientY - rect.top) / rect.height) * 100));
      }
    };
    const end = (e: PointerEvent) => {
      handle.releasePointerCapture?.(e.pointerId);
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', end);
      handle.removeEventListener('pointercancel', end);
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', end);
    handle.addEventListener('pointercancel', end);
  }

  return {
    preset, layout, presetMenuOpen, presetMenuRef, autoView, paneViews, stageRef, paneSplit,
    presetOptions, viewOptions, activePreset, modeLabel, panes, showHandleX, showHandleY, stageStyle,
    viewLabel, registerView, resetSplit, toggleMenu, setPreset, setPaneView, showView, startDrag,
  };
}

let shared: ReturnType<typeof createWorkspace> | null = null;

/** Shared workspace for the whole app, so any component can call showView. */
export function useWorkspace() {
  if (!shared) shared = createWorkspace();
  return shared;
}
