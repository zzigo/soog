import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildPresetTree, createIdGen, dropPreviewRect, dropZone, insertBeside, layoutTree, leaves, maxIdNumber,
  moveLeaf, removeLeaf, setLeafView, setRatio, swapViews, validateTree,
} from '../composables/workspaceTree.ts';

const views = ['a', 'b', 'c', 'd'];
const four = () => {
  const newId = createIdGen();
  return { tree: buildPresetTree(4, views, newId), newId };
};

test('presets build 1, 2, 3 and 4 panes in reading order', () => {
  for (const count of [1, 2, 3, 4]) {
    const tree = buildPresetTree(count, views, createIdGen());
    assert.deepEqual(leaves(tree).map((leaf) => leaf.view), views.slice(0, count));
  }
});

test('layout rectangles tile the stage with no gap', () => {
  const { tree } = four();
  const { panes, handles } = layoutTree(tree);
  assert.equal(panes.length, 4);
  assert.equal(handles.length, 3);
  const area = panes.reduce((sum, pane) => sum + pane.w * pane.h, 0);
  assert.ok(Math.abs(area - 10000) < 1e-6);
  const three = layoutTree(buildPresetTree(3, views, createIdGen()));
  assert.deepEqual(three.panes.map((p) => [p.x, p.y, p.w, p.h]), [[0, 0, 50, 100], [50, 0, 50, 50], [50, 50, 50, 50]]);
});

test('removing a pane gives its space to the sibling', () => {
  const { tree } = four();
  const [a, b] = leaves(tree);
  const rest = removeLeaf(tree, a.id);
  const { panes } = layoutTree(rest);
  assert.equal(panes.length, 3);
  assert.equal(panes.find((p) => p.id === b.id).w, 100);
  assert.equal(removeLeaf(buildPresetTree(1, views, createIdGen()), 'n1'), null);
});

test('moving to an edge docks beside the target; the centre swaps views', () => {
  const { tree, newId } = four();
  const [a, , , d] = leaves(tree);
  const docked = moveLeaf(tree, a.id, d.id, 'right', newId);
  const { panes } = layoutTree(docked);
  const pa = panes.find((p) => p.id === a.id);
  const pd = panes.find((p) => p.id === d.id);
  assert.equal(pa.y, pd.y);
  assert.ok(pa.x > pd.x);
  assert.equal(panes.length, 4);
  const swapped = moveLeaf(tree, a.id, d.id, 'center', newId);
  assert.deepEqual(leaves(swapped).map((l) => l.view), ['d', 'b', 'c', 'a']);
  assert.equal(moveLeaf(tree, a.id, a.id, 'left', newId), tree);
});

test('ratio is clamped and the id counter resumes after a restore', () => {
  const { tree } = four();
  assert.equal(setRatio(tree, tree.id, 2).ratio, 15);
  assert.equal(setRatio(tree, tree.id, 99).ratio, 85);
  assert.equal(maxIdNumber(tree), 7);
});

test('drop zones and previews', () => {
  assert.equal(dropZone(0.5, 0.5), 'center');
  assert.equal(dropZone(0.1, 0.5), 'left');
  assert.equal(dropZone(0.9, 0.4), 'right');
  assert.equal(dropZone(0.5, 0.05), 'top');
  assert.equal(dropZone(0.45, 0.95), 'bottom');
  assert.deepEqual(dropPreviewRect({ x: 10, y: 20, w: 40, h: 60 }, 'right'), { x: 30, y: 20, w: 20, h: 60 });
  assert.deepEqual(dropPreviewRect({ x: 10, y: 20, w: 40, h: 60 }, 'top'), { x: 10, y: 20, w: 40, h: 30 });
});

test('validateTree accepts a good tree, repairs unknown views and rejects garbage', () => {
  const { tree } = four();
  const json = JSON.parse(JSON.stringify(setLeafView(tree, 'n1', 'gone')));
  const ok = validateTree(json, (key) => key !== 'gone');
  assert.equal(leaves(ok)[0].view, 'auto');
  assert.equal(validateTree({ type: 'leaf' }, () => true), null);
  assert.equal(validateTree({ type: 'split', id: 'x', dir: 'diag', ratio: 50 }, () => true), null);
  assert.equal(validateTree(null, () => true), null);
  const dup = { type: 'split', id: 's', dir: 'row', ratio: 50, a: { type: 'leaf', id: 'l', view: 'a' }, b: { type: 'leaf', id: 'l', view: 'b' } };
  assert.equal(validateTree(dup, () => true), null);
});

test('insertBeside and swapViews leave other panes alone', () => {
  const newId = createIdGen(100);
  const tree = buildPresetTree(2, ['a', 'b'], createIdGen());
  const [a, b] = leaves(tree);
  const next = insertBeside(tree, b.id, { type: 'leaf', id: 'z', view: 'z' }, 'bottom', newId);
  assert.equal(leaves(next).length, 3);
  assert.deepEqual(leaves(swapViews(tree, a.id, b.id)).map((l) => l.view), ['b', 'a']);
});
