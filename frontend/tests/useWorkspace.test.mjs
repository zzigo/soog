import test from 'node:test';
import assert from 'node:assert/strict';
import { createWorkspace } from '../composables/useWorkspace.ts';

test('fullscreen starts with one auto-routed pane', () => {
  const w = createWorkspace();
  assert.equal(w.panes.value.length, 1);
  assert.equal(w.panes.value[0].view, 'organogram');
  w.autoView.value = 'sound';
  assert.equal(w.panes.value[0].view, 'sound');
  assert.deepEqual(w.stageStyle.value, {});
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
