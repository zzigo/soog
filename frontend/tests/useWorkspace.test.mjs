import test from 'node:test';
import assert from 'node:assert/strict';
import { createWorkspace } from '../composables/useWorkspace.ts';

test('fullscreen starts with one auto-routed pane', () => {
  const w = createWorkspace();
  assert.equal(w.panes.value.length, 1);
  assert.equal(w.panes.value[0].view, 'organogram');
  w.autoView.value = 'sound';
  assert.equal(w.panes.value[0].view, 'sound');
  assert.equal(w.layout.value, 'background');
});

test('presets set layout and default views', () => {
  const w = createWorkspace();
  w.setPreset('splitFour');
  assert.equal(w.layout.value, 'split');
  assert.deepEqual(w.panes.value.map((p) => p.view), ['organogram', 'sound', 'geometry', 'acoustic']);
  w.setPreset('fullscreen');
  assert.equal(w.layout.value, 'background');
  w.setPreset('unknown');
  assert.equal(w.preset.value, 'fullscreen');
});

test('registry: registerView is idempotent and showView places a view in a pane', () => {
  const w = createWorkspace();
  w.registerView({ key: 'gallery', label: 'Gallery' });
  w.registerView({ key: 'gallery', label: 'duplicate' });
  assert.equal(w.viewOptions.value.filter((v) => v.key === 'gallery').length, 1);
  assert.equal(w.viewLabel('gallery'), 'Gallery');
  assert.equal(w.showView('missing'), false);
  w.setPreset('splitFour');
  assert.equal(w.showView('geometry'), true);
  assert.deepEqual(w.panes.value.map((p) => p.view), ['organogram', 'sound', 'geometry', 'acoustic']);
  w.showView('gallery');
  assert.equal(w.panes.value[3].view, 'gallery');
  w.setPreset('fullscreen');
  w.showView('gallery');
  assert.equal(w.panes.value[0].view, 'gallery');
});

test('core views include SOT-A and the 3D stage', () => {
  const w = createWorkspace();
  const keys = w.viewOptions.value.map((v) => v.key);
  assert.ok(keys.includes('sota'));
  assert.ok(keys.includes('stage'));
  assert.equal(w.viewLabel('stage'), 'Stage 3D');
});

test('maximizePane shows one pane alone and restorePanes returns to the previous layout', () => {
  const w = createWorkspace();
  w.setPreset('splitFour');
  w.setPaneView(1, 'sota');
  w.maximizePane(1);
  assert.equal(w.preset.value, 'fullscreen');
  assert.equal(w.panes.value.length, 1);
  assert.equal(w.panes.value[0].view, 'sota');
  assert.equal(w.canRestore.value, true);
  w.restorePanes();
  assert.equal(w.preset.value, 'splitFour');
  assert.deepEqual(w.panes.value.map((p) => p.view), ['organogram', 'sota', 'geometry', 'acoustic']);
  assert.equal(w.canRestore.value, false);
});

test('closePane gives the space to the neighbour and the last pane stays', () => {
  const w = createWorkspace();
  w.setPreset('splitVertical');
  assert.equal(w.layout.value, 'split');
  w.closePane(0);
  assert.equal(w.panes.value.length, 1);
  assert.equal(w.panes.value[0].view, 'geometry');
  assert.equal(w.panes.value[0].w, 100);
  assert.equal(w.preset.value, 'fullscreen');
  assert.equal(w.layout.value, 'background');
  w.closePane(0);
  assert.equal(w.panes.value.length, 1);
});

test('closing one of four leaves a custom layout of three that still tiles the stage', () => {
  const w = createWorkspace();
  w.setPreset('splitFour');
  w.closePane(1);
  assert.equal(w.preset.value, 'custom');
  assert.equal(w.modeLabel.value, 'Custom');
  assert.equal(w.panes.value.length, 3);
  const area = w.panes.value.reduce((sum, p) => sum + p.w * p.h, 0);
  assert.ok(Math.abs(area - 10000) < 1e-6);
});

test('splitPane adds a pane with a view that is not shown yet, up to eight', () => {
  const w = createWorkspace();
  w.splitPane(0);
  assert.equal(w.panes.value.length, 2);
  assert.notEqual(w.panes.value[0].view, w.panes.value[1].view);
  for (let i = 0; i < 12; i += 1) w.splitPane(0, 'col');
  assert.equal(w.panes.value.length, 8);
});

test('movePane docks beside another pane or swaps views at the centre', () => {
  const w = createWorkspace();
  w.setPreset('splitFour');
  const ids = w.panes.value.map((p) => p.id);
  w.movePane(ids[0], ids[3], 'center');
  assert.deepEqual(w.panes.value.map((p) => p.view), ['acoustic', 'sound', 'geometry', 'organogram']);
  w.movePane(ids[1], ids[2], 'bottom');
  assert.equal(w.panes.value.length, 4);
  const moved = w.panes.value.find((p) => p.id === ids[1]);
  const target = w.panes.value.find((p) => p.id === ids[2]);
  assert.equal(moved.x, target.x);
  assert.ok(moved.y > target.y);
});

test('a hand-made layout is saved and restored', async () => {
  const store = new Map();
  globalThis.localStorage = { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, v), removeItem: (k) => store.delete(k) };
  globalThis.window = {};
  try {
    const w = createWorkspace();
    w.setPreset('splitFour');
    w.closePane(3);
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.ok(store.get('soog.workspace.layout.v1'));
    const fresh = createWorkspace();
    fresh.restoreLayout();
    assert.equal(fresh.panes.value.length, 3);
    assert.equal(fresh.preset.value, 'custom');
    fresh.splitPane(0);
    const ids = fresh.panes.value.map((p) => p.id);
    assert.equal(new Set(ids).size, ids.length);
  } finally {
    delete globalThis.localStorage;
    delete globalThis.window;
  }
});

test('snapshotLayout and restoreSnapshot put a layout back by reference', () => {
  const w = createWorkspace();
  const before = w.snapshotLayout();
  w.setPreset('splitFour');
  const applied = w.tree.value;
  assert.equal(w.panes.value.length, 4);
  w.restoreSnapshot(before);
  assert.equal(w.panes.value.length, 1);
  assert.equal(w.preset.value, 'fullscreen');
  assert.notEqual(w.tree.value, applied);
});
