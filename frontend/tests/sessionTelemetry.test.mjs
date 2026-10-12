import test from 'node:test';
import assert from 'node:assert/strict';
import { buildServerFacts, buildStageRows, formatSeconds, formatStageLabel } from '../composables/sessionTelemetry.ts';

test('stage rows: duration is the gap to the next stage, the last one follows the request', () => {
  const events = [
    { stage: 'received', t_ms: 0, cpu_pct: 12 },
    { stage: 'plot_llm', t_ms: 400, cpu_pct: 88.5 },
    { stage: 'rendering_organogram', t_ms: 9400 },
  ];
  const done = buildStageRows(events, { status: 'completed', started_at: 100, updated_at: 112 });
  assert.deepEqual(done.map((r) => r.label), ['Received', 'Plot LLM', 'Rendering Organogram']);
  assert.deepEqual(done.map((r) => r.duration), ['0.4s', '9.0s', '2.6s']);
  assert.deepEqual(done.map((r) => r.cpu), [12, 88.5, null]);
  assert.equal(done.some((r) => r.running), false);
  const live = buildStageRows(events, { status: 'running', started_at: 100, updated_at: 110 });
  assert.equal(live[2].running, true);
  assert.equal(live[2].duration, '…');
});

test('stage rows: empty and clamped input', () => {
  assert.deepEqual(buildStageRows([], { status: '', started_at: 0, updated_at: 0 }), []);
  const [row] = buildStageRows([{ stage: 'x', t_ms: 0, cpu_pct: 250 }], { status: 'completed', started_at: 0, updated_at: 1 });
  assert.equal(row.cpu, 100);
});

test('server facts format CPU, load, memory, swap and backend RSS', () => {
  assert.deepEqual(buildServerFacts(null), []);
  const facts = buildServerFacts({
    cpu_pct: 41.2, cpus: 8, load1: 4.68, load5: 4.68, load15: 4.86,
    mem_total_mb: 15360, mem_used_mb: 5120, swap_total_mb: 8192, swap_used_mb: 4096, process_rss_mb: 1536,
  });
  assert.deepEqual(facts.map((f) => f.label), ['CPU', 'Load 1/5/15', 'Memory', 'Swap', 'Backend RSS']);
  assert.equal(facts[0].value, '41.2% of 8 cores');
  assert.equal(facts[2].value, '5.0 / 15.0 GB');
  assert.equal(facts[4].value, '1.50 GB');
});

test('formatters', () => {
  assert.equal(formatSeconds(1500), '1.5s');
  assert.equal(formatSeconds(65000), '65s');
  assert.equal(formatSeconds(-1), '—');
  assert.equal(formatStageLabel('building_geometry'), 'Building Geometry');
  assert.equal(formatStageLabel('stl_llm'), 'STL LLM');
});
