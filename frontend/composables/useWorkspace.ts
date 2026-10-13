import { computed, ref, watch } from 'vue';
import {
  MAX_PANES, buildPresetTree, createIdGen, dropPreviewRect, dropZone, findLeaf, insertBeside, layoutTree, leaves,
  makeLeaf, maxIdNumber, moveLeaf, removeLeaf, setLeafView, setRatio, validateTree,
  type DropZone, type HandleRect, type TreeNode,
} from './workspaceTree.ts';

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
  { key: 'summary', label: 'Summary' },
  { key: 'geometry', label: '3D Object' },
  { key: 'acoustic', label: 'Acoustic' },
  { key: 'sketch', label: 'Sketch' },
  { key: 'sota', label: 'SOT-A' },
  { key: 'stage', label: 'Stage 3D' },
];

const CUSTOM_PRESET: WorkspacePreset = {
  label: 'Custom',
  summary: 'Panes arranged by hand.',
  areas: [],
  defaults: [],
};

const LAYOUT_KEY = 'soog.workspace.layout.v1';
const NEW_PANE_ORDER = ['organogram', 'sound', 'geometry', 'acoustic', 'sketch', 'summary', 'sota', 'stage'];

export interface PaneDrag {
  fromId: string;
  overId: string;
  zone: DropZone | null;
  x: number;
  y: number;
}

/**
 * Workspace state: a layout tree of panes (each showing a view), the shared auto view, and the
 * drag state for moving panes. The classic presets build a tree; after that the user can close,
 * split, resize and drag panes freely. The view catalog is a registry: new pane types call
 * registerView and are then selectable in every pane and reachable through showView.
 */
export function createWorkspace() {
  const newId = createIdGen();
  const preset = ref('fullscreen');
  const presetMenuOpen = ref(false);
  const presetMenuRef = ref<HTMLElement | null>(null);
  const autoView = ref('organogram');
  const tree = ref<TreeNode>(buildPresetTree(1, WORKSPACE_PRESETS.fullscreen.defaults, newId));
  const views = ref<WorkspaceView[]>([...CORE_VIEWS]);
  const stageRef = ref<HTMLElement | null>(null);
  const restoreState = ref<{ preset: string; tree: TreeNode } | null>(null);
  const drag = ref<PaneDrag | null>(null);

  const presetOptions = computed(() => Object.entries(WORKSPACE_PRESETS).map(([key, value]) => ({
    key,
    label: value.label,
    summary: value.summary,
  })));
  const viewOptions = computed(() => views.value);
  const activePreset = computed(() => WORKSPACE_PRESETS[preset.value] || CUSTOM_PRESET);
  const modeLabel = computed(() => activePreset.value.label);
  const geometry = computed(() => layoutTree(tree.value));
  const panes = computed(() => geometry.value.panes.map((rect, index) => {
    const view = rect.view === 'auto' ? autoView.value : rect.view;
    return {
      id: rect.id, area: String.fromCharCode(97 + index), index, assigned: rect.view, view,
      x: rect.x, y: rect.y, w: rect.w, h: rect.h,
    };
  }));
  const paneViews = computed(() => leaves(tree.value).map((leaf) => leaf.view));
  const handles = computed(() => geometry.value.handles);
  /** One pane is the immersive background layout; two or more are a split. */
  const layout = computed({
    get: () => (panes.value.length > 1 ? 'split' : 'background'),
    set: () => {},
  });

  function viewLabel(key: string) {
    return views.value.find((item) => item.key === key)?.label || String(key || 'Pane');
  }

  function registerView(view: WorkspaceView) {
    if (!view?.key || views.value.some((item) => item.key === view.key)) return;
    views.value = [...views.value, view];
  }

  function toggleMenu() {
    presetMenuOpen.value = !presetMenuOpen.value;
  }

  /** After a hand edit the layout is custom, unless a single pane is left. */
  function markEdited() {
    preset.value = leaves(tree.value).length === 1 ? 'fullscreen' : 'custom';
  }

  function setPreset(presetKey: string) {
    const next = WORKSPACE_PRESETS[presetKey];
    if (!next) return;
    preset.value = presetKey;
    tree.value = buildPresetTree(next.areas.length, next.defaults, newId);
    presetMenuOpen.value = false;
  }

  function leafAt(index: number) {
    return leaves(tree.value)[index] || null;
  }

  function setPaneView(index: number, viewKey: string) {
    const leaf = leafAt(index);
    if (leaf) tree.value = setLeafView(tree.value, leaf.id, viewKey);
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

  /** Close a pane: its neighbour fills the space. The last pane stays. */
  function closePane(index: number) {
    const leaf = leafAt(index);
    if (!leaf || leaves(tree.value).length <= 1) return;
    const rest = removeLeaf(tree.value, leaf.id);
    if (rest) tree.value = rest;
    markEdited();
  }

  /** Split a pane in two: row puts the new pane on the right, col below. */
  function splitPane(index: number, dir: 'row' | 'col' = 'row') {
    const leaf = leafAt(index);
    if (!leaf || leaves(tree.value).length >= MAX_PANES) return;
    const shown = new Set(panes.value.map((pane) => pane.view));
    const view = NEW_PANE_ORDER.find((key) => !shown.has(key) && views.value.some((item) => item.key === key)) || 'auto';
    tree.value = insertBeside(tree.value, leaf.id, makeLeaf(view, newId), dir === 'row' ? 'right' : 'bottom', newId);
    markEdited();
  }

  function movePane(fromId: string, toId: string, zone: DropZone) {
    tree.value = moveLeaf(tree.value, fromId, toId, zone, newId);
    markEdited();
  }

  /** Show one pane alone (fullscreen = a one-pane layout) and remember the layout to return to. */
  function maximizePane(index: number) {
    const pane = panes.value[index];
    if (!pane) return;
    if (panes.value.length > 1) restoreState.value = { preset: preset.value, tree: tree.value };
    preset.value = 'fullscreen';
    tree.value = makeLeaf(pane.assigned, newId);
  }

  function restorePanes() {
    const saved = restoreState.value;
    if (saved) {
      tree.value = saved.tree;
      preset.value = saved.preset;
    } else {
      setPreset('splitFour');
    }
    restoreState.value = null;
  }

  /** The layout as it is now, to put back later (the gallery borrows the stage while it is open). */
  function snapshotLayout() {
    return { preset: preset.value, tree: tree.value };
  }

  function restoreSnapshot(snapshot: { preset: string; tree: TreeNode }) {
    tree.value = snapshot.tree;
    preset.value = snapshot.preset;
  }

  const canRestore = computed(() => panes.value.length === 1 && !!restoreState.value);

  function startHandleDrag(handle: HandleRect, event: PointerEvent) {
    const stage = stageRef.value;
    if (!stage) return;
    const target = event.currentTarget as HTMLElement;
    target.setPointerCapture?.(event.pointerId);
    const move = (e: PointerEvent) => {
      const rect = stage.getBoundingClientRect();
      const split = handle.split;
      const ratio = handle.dir === 'row'
        ? ((((e.clientX - rect.left) / rect.width) * 100 - split.x) / split.w) * 100
        : ((((e.clientY - rect.top) / rect.height) * 100 - split.y) / split.h) * 100;
      tree.value = setRatio(tree.value, handle.id, ratio);
    };
    const end = (e: PointerEvent) => {
      target.releasePointerCapture?.(e.pointerId);
      target.removeEventListener('pointermove', move);
      target.removeEventListener('pointerup', end);
      target.removeEventListener('pointercancel', end);
    };
    target.addEventListener('pointermove', move);
    target.addEventListener('pointerup', end);
    target.addEventListener('pointercancel', end);
  }

  function resetHandle(handle: HandleRect) {
    tree.value = setRatio(tree.value, handle.id, 50);
  }

  /** Drag a pane by its header; releasing over another pane docks (edge) or swaps (centre). */
  function startPaneMove(paneId: string, event: PointerEvent) {
    const stage = stageRef.value;
    if (!stage || event.button !== 0 || panes.value.length < 2 || typeof window === 'undefined') return;
    const startX = event.clientX;
    const startY = event.clientY;
    let moving = false;
    const move = (e: PointerEvent) => {
      if (!moving && Math.hypot(e.clientX - startX, e.clientY - startY) < 6) return;
      moving = true;
      const rect = stage.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * 100;
      const py = ((e.clientY - rect.top) / rect.height) * 100;
      const over = geometry.value.panes.find((pane) => pane.id !== paneId
        && px >= pane.x && px <= pane.x + pane.w && py >= pane.y && py <= pane.y + pane.h);
      drag.value = {
        fromId: paneId,
        overId: over?.id || '',
        zone: over ? dropZone((px - over.x) / over.w, (py - over.y) / over.h) : null,
        x: e.clientX,
        y: e.clientY,
      };
    };
    const finish = (apply: boolean) => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', cancel);
      window.removeEventListener('keydown', key);
      const state = drag.value;
      drag.value = null;
      if (apply && moving && state?.overId && state.zone) movePane(state.fromId, state.overId, state.zone);
    };
    const up = () => finish(true);
    const cancel = () => finish(false);
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') finish(false); };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', cancel);
    window.addEventListener('keydown', key);
  }

  const dropPreview = computed(() => {
    const state = drag.value;
    if (!state?.overId || !state.zone) return null;
    const target = geometry.value.panes.find((pane) => pane.id === state.overId);
    if (!target) return null;
    return { ...dropPreviewRect(target, state.zone), zone: state.zone };
  });

  /** Remember a hand-made split across visits; a single pane is the default and is not stored. */
  function restoreLayout() {
    try {
      const saved = JSON.parse(localStorage.getItem(LAYOUT_KEY) || 'null');
      const restored = validateTree(saved?.tree, (key) => views.value.some((item) => item.key === key));
      if (!restored || leaves(restored).length < 2) return;
      newId.reseed?.(maxIdNumber(restored));
      tree.value = restored;
      preset.value = WORKSPACE_PRESETS[saved.preset] ? saved.preset : 'custom';
    } catch {
      // unreadable or blocked storage: keep the default layout
    }
  }

  function saveLayout() {
    if (typeof window === 'undefined') return;
    try {
      if (leaves(tree.value).length < 2) localStorage.removeItem(LAYOUT_KEY);
      else localStorage.setItem(LAYOUT_KEY, JSON.stringify({ tree: tree.value, preset: preset.value }));
    } catch {
      // storage can be blocked; the layout then lasts for this visit
    }
  }

  watch([tree, preset], saveLayout, { deep: true });

  return {
    preset, layout, presetMenuOpen, presetMenuRef, autoView, paneViews, stageRef, tree, drag, dropPreview, handles,
    presetOptions, viewOptions, activePreset, modeLabel, panes,
    viewLabel, registerView, toggleMenu, setPreset, setPaneView, showView, closePane, splitPane, movePane,
    maximizePane, restorePanes, canRestore, snapshotLayout, restoreSnapshot, startHandleDrag, resetHandle, startPaneMove, restoreLayout,
  };
}

let shared: ReturnType<typeof createWorkspace> | null = null;

/** Shared workspace for the whole app, so any component can call showView. */
export function useWorkspace() {
  if (!shared) shared = createWorkspace();
  return shared;
}
