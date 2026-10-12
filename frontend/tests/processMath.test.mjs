import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_ESTIMATE_MS, estimateMs, estimateProgress, formatClock, isActive, median, shortLabel,
} from '../composables/processMath.ts';

test('estimateMs uses the median of past runs and falls back to the kind default', () => {
  assert.equal(estimateMs('sound', [100000, 300000, 200000]), 200000);
  assert.equal(estimateMs('sketch', []), DEFAULT_ESTIMATE_MS.sketch);
  assert.equal(estimateMs('mesh', [0]), DEFAULT_ESTIMATE_MS.mesh);
  assert.ok(Number.isNaN(median([])));
});

test('estimateProgress rises to 92 at the typical time and never reaches 100', () => {
  assert.equal(estimateProgress(0, 60000), 0);
  assert.ok(Math.abs(estimateProgress(60000, 60000) - 92) < 1e-9);
  assert.ok(estimateProgress(30000, 60000) > 40 && estimateProgress(30000, 60000) < 50);
  const late = estimateProgress(60 * 60000, 60000);
  assert.ok(late > 92 && late <= 98);
});

test('formatClock', () => {
  assert.equal(formatClock(7400), '7s');
  assert.equal(formatClock(65000), '1m05s');
  assert.equal(formatClock(720000), '12m00s');
  assert.equal(formatClock(-1), '—');
});

test('isActive and shortLabel', () => {
  assert.equal(isActive('running'), true);
  assert.equal(isActive('cancelling'), true);
  assert.equal(isActive('completed'), false);
  assert.equal(shortLabel('  a   long\nprompt ', 8), 'a long …');
  assert.equal(shortLabel('short'), 'short');
});
