// Pure layout tree behind the workspace docking: panes are leaves, splits carry a direction and a
// ratio. Every operation returns a new tree. Rectangles are percentages of the stage.

export type SplitDir = 'row' | 'col'; // row: a left, b right. col: a on top, b below.
export type DropZone = 'left' | 'right' | 'top' | 'bottom' | 'center';

export interface Leaf {
  type: 'leaf';
  id: string;
  view: string;
}

export interface Split {
  type: 'split';
  id: string;
  dir: SplitDir;
  ratio: number;
  a: TreeNode;
  b: TreeNode;
}

export type TreeNode = Leaf | Split;

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface PaneRect extends Rect {
  id: string;
  view: string;
}

export interface HandleRect {
  id: string;
  dir: SplitDir;
  /** Position of the divider line, in percent of the stage. */
  x: number;
  y: number;
  length: number;
  /** The rectangle of the split this handle belongs to. */
  split: Rect;
}

export const RATIO_MIN = 15;
export const RATIO_MAX = 85;
export const MAX_PANES = 8;

export type IdGen = (() => string) & { reseed?: (floor: number) => void };

export function createIdGen(start = 0): IdGen {
  let counter = start;
  const gen: IdGen = () => `n${++counter}`;
  gen.reseed = (floor: number) => { counter = Math.max(counter, floor); };
  return gen;
}

/** Highest numeric suffix among node ids, so a restored tree keeps generating unique ids. */
export function maxIdNumber(node: TreeNode): number {
  const own = Number(/^n(\d+)$/.exec(node.id)?.[1] || 0);
  return node.type === 'leaf' ? own : Math.max(own, maxIdNumber(node.a), maxIdNumber(node.b));
}

const clampRatio = (value: number) => Math.min(RATIO_MAX, Math.max(RATIO_MIN, value));

export function makeLeaf(view: string, newId: IdGen): Leaf {
  return { type: 'leaf', id: newId(), view };
}

function makeSplit(dir: SplitDir, a: TreeNode, b: TreeNode, newId: IdGen, ratio = 50): Split {
  return { type: 'split', id: newId(), dir, ratio, a, b };
}

/** The classic presets as trees: 1 pane, 2 side by side, one tall plus two stacked, 2 by 2. */
export function buildPresetTree(areas: number, views: string[], newId: IdGen): TreeNode {
  const leaf = (index: number) => makeLeaf(views[index] || 'auto', newId);
  if (areas <= 1) return leaf(0);
  if (areas === 2) return makeSplit('row', leaf(0), leaf(1), newId);
  if (areas === 3) return makeSplit('row', leaf(0), makeSplit('col', leaf(1), leaf(2), newId), newId);
  return makeSplit('col', makeSplit('row', leaf(0), leaf(1), newId), makeSplit('row', leaf(2), leaf(3), newId), newId);
}

export function leaves(node: TreeNode): Leaf[] {
  return node.type === 'leaf' ? [node] : [...leaves(node.a), ...leaves(node.b)];
}

export function findLeaf(node: TreeNode, id: string): Leaf | null {
  return leaves(node).find((leaf) => leaf.id === id) || null;
}

export function setLeafView(node: TreeNode, id: string, view: string): TreeNode {
  if (node.type === 'leaf') return node.id === id ? { ...node, view } : node;
  return { ...node, a: setLeafView(node.a, id, view), b: setLeafView(node.b, id, view) };
}

/** Remove a pane; its sibling takes the space. Returns null when the tree was that single pane. */
export function removeLeaf(node: TreeNode, id: string): TreeNode | null {
  if (node.type === 'leaf') return node.id === id ? null : node;
  const a = removeLeaf(node.a, id);
  const b = removeLeaf(node.b, id);
  if (!a) return b;
  if (!b) return a;
  return a === node.a && b === node.b ? node : { ...node, a, b };
}

/** Put a leaf beside a target pane: the target is split in two and the new leaf takes one half. */
export function insertBeside(node: TreeNode, targetId: string, leaf: Leaf, zone: Exclude<DropZone, 'center'>, newId: IdGen): TreeNode {
  if (node.type === 'leaf') {
    if (node.id !== targetId) return node;
    const dir: SplitDir = zone === 'left' || zone === 'right' ? 'row' : 'col';
    const first = zone === 'left' || zone === 'top';
    return makeSplit(dir, first ? leaf : node, first ? node : leaf, newId);
  }
  return { ...node, a: insertBeside(node.a, targetId, leaf, zone, newId), b: insertBeside(node.b, targetId, leaf, zone, newId) };
}

export function swapViews(node: TreeNode, idA: string, idB: string): TreeNode {
  const first = findLeaf(node, idA);
  const second = findLeaf(node, idB);
  if (!first || !second) return node;
  return setLeafView(setLeafView(node, idA, second.view), idB, first.view);
}

/** Drag a pane onto another: an edge docks it beside the target, the centre swaps the two views. */
export function moveLeaf(node: TreeNode, fromId: string, toId: string, zone: DropZone, newId: IdGen): TreeNode {
  if (fromId === toId) return node;
  if (zone === 'center') return swapViews(node, fromId, toId);
  const source = findLeaf(node, fromId);
  if (!source || !findLeaf(node, toId)) return node;
  const rest = removeLeaf(node, fromId);
  return rest ? insertBeside(rest, toId, source, zone, newId) : node;
}

export function setRatio(node: TreeNode, splitId: string, ratio: number): TreeNode {
  if (node.type === 'leaf') return node;
  if (node.id === splitId) return { ...node, ratio: clampRatio(ratio) };
  return { ...node, a: setRatio(node.a, splitId, ratio), b: setRatio(node.b, splitId, ratio) };
}

export function layoutTree(node: TreeNode, rect: Rect = { x: 0, y: 0, w: 100, h: 100 }): { panes: PaneRect[]; handles: HandleRect[] } {
  if (node.type === 'leaf') return { panes: [{ id: node.id, view: node.view, ...rect }], handles: [] };
  const share = node.ratio / 100;
  const first: Rect = node.dir === 'row'
    ? { x: rect.x, y: rect.y, w: rect.w * share, h: rect.h }
    : { x: rect.x, y: rect.y, w: rect.w, h: rect.h * share };
  const second: Rect = node.dir === 'row'
    ? { x: rect.x + first.w, y: rect.y, w: rect.w - first.w, h: rect.h }
    : { x: rect.x, y: rect.y + first.h, w: rect.w, h: rect.h - first.h };
  const left = layoutTree(node.a, first);
  const right = layoutTree(node.b, second);
  const handle: HandleRect = node.dir === 'row'
    ? { id: node.id, dir: 'row', x: second.x, y: rect.y, length: rect.h, split: rect }
    : { id: node.id, dir: 'col', x: rect.x, y: second.y, length: rect.w, split: rect };
  return { panes: [...left.panes, ...right.panes], handles: [handle, ...left.handles, ...right.handles] };
}

/** Which part of a pane the pointer is over; fx and fy are 0..1 inside the pane. */
export function dropZone(fx: number, fy: number): DropZone {
  if (fx > 0.25 && fx < 0.75 && fy > 0.25 && fy < 0.75) return 'center';
  const distances: [Exclude<DropZone, 'center'>, number][] = [['left', fx], ['right', 1 - fx], ['top', fy], ['bottom', 1 - fy]];
  return distances.reduce((best, entry) => (entry[1] < best[1] ? entry : best))[0];
}

export function dropPreviewRect(rect: Rect, zone: DropZone): Rect {
  switch (zone) {
    case 'left': return { ...rect, w: rect.w / 2 };
    case 'right': return { ...rect, x: rect.x + rect.w / 2, w: rect.w / 2 };
    case 'top': return { ...rect, h: rect.h / 2 };
    case 'bottom': return { ...rect, y: rect.y + rect.h / 2, h: rect.h / 2 };
    default: return rect;
  }
}

/** Read a saved tree: anything malformed gives null; unknown views fall back to auto. */
export function validateTree(raw: unknown, isView: (key: string) => boolean): TreeNode | null {
  const seen = new Set<string>();
  const read = (value: any, depth: number): TreeNode | null => {
    if (!value || typeof value !== 'object' || depth > 6 || typeof value.id !== 'string' || seen.has(value.id)) return null;
    seen.add(value.id);
    if (value.type === 'leaf') {
      return { type: 'leaf', id: value.id, view: typeof value.view === 'string' && isView(value.view) ? value.view : 'auto' };
    }
    if (value.type === 'split' && (value.dir === 'row' || value.dir === 'col') && Number.isFinite(value.ratio)) {
      const a = read(value.a, depth + 1);
      const b = read(value.b, depth + 1);
      return a && b ? { type: 'split', id: value.id, dir: value.dir, ratio: clampRatio(value.ratio), a, b } : null;
    }
    return null;
  };
  const tree = read(raw, 0);
  return tree && leaves(tree).length <= MAX_PANES ? tree : null;
}
