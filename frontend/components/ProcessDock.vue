<template>
  <section v-if="rows.length" class="pdock" :class="{ 'pdock--collapsed': collapsed }" :style="{ left: `${inset}px`, bottom: `${bottom}px` }" aria-label="Processes">
    <button type="button" class="pdock__head" :aria-expanded="String(!collapsed)" @click="toggle">
      <span class="pdock__title">PROCESSES</span>
      <span class="pdock__count">{{ summary }}</span>
      <template v-if="collapsed && lead">
        <span class="pdock__lead">{{ lead.kindLabel }} {{ lead.label }}</span>
        <span class="pdock__lead-bar"><i :style="{ width: `${lead.percent}%` }"></i></span>
        <span class="pdock__lead-time">{{ lead.elapsed }}</span>
      </template>
      <span class="pdock__chevron" :class="{ open: !collapsed }" aria-hidden="true">▾</span>
    </button>
    <ol v-if="!collapsed" class="pdock__rows">
      <li v-for="row in rows" :key="row.id" class="prow" :class="`prow--${row.status}`">
        <span class="prow__kind">{{ row.kindLabel }}</span>
        <span class="prow__model" :title="row.model">{{ row.model || '—' }}</span>
        <span class="prow__target" :title="row.label">{{ row.label }}</span>
        <span class="prow__stage" :title="row.statusText">{{ row.statusText }}</span>
        <span class="prow__bar" role="progressbar" :aria-valuenow="Math.round(row.percent)" aria-valuemin="0" aria-valuemax="100"><i :style="{ width: `${row.percent}%` }"></i></span>
        <span class="prow__time" :title="row.timeTitle">{{ row.time }}</span>
        <button
          v-if="row.active ? row.cancellable : true"
          type="button"
          class="prow__x"
          :title="row.active ? 'Cancel this process' : 'Dismiss'"
          :aria-label="row.active ? 'Cancel this process' : 'Dismiss'"
          :disabled="row.status === 'cancelling'"
          @click="row.active ? cancelProcess(row.id) : dismissProcess(row.id)"
        >×</button>
        <span v-else class="prow__x prow__x--none" title="This job cannot be stopped from here">·</span>
        <span v-if="row.notice || row.preview" class="prow__preview" :class="{ 'prow__preview--notice': row.notice }" :title="row.notice || row.preview">{{ row.notice || row.preview }}</span>
      </li>
    </ol>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue';
import { useProcesses } from '~/composables/useProcesses';
import { KIND_LABEL, estimateProgress, formatClock, isActive } from '~/composables/processMath';
import { formatStageLabel } from '~/composables/sessionTelemetry';

// Bottom console with one row per process: kind, model, target, stage, bar, time and a cancel cross.
defineProps({
  inset: { type: Number, default: 44 },
  bottom: { type: Number, default: 0 },
});

const COLLAPSED_KEY = 'soog.pdock.collapsed';
const { processes, now, cancelProcess, dismissProcess } = useProcesses();
const collapsed = ref(false);

onMounted(() => {
  try {
    collapsed.value = localStorage.getItem(COLLAPSED_KEY) === '1';
  } catch {
    // storage blocked: stay expanded
  }
});

function toggle() {
  collapsed.value = !collapsed.value;
  try {
    localStorage.setItem(COLLAPSED_KEY, collapsed.value ? '1' : '0');
  } catch {
    // storage blocked: the choice lasts for this page only
  }
}

function describe(row) {
  const active = isActive(row.status);
  const end = row.finishedAt || now.value;
  const elapsed = Math.max(0, end - row.startedAt);
  let percent = 100;
  let statusText = '';
  if (active) {
    percent = estimateProgress(elapsed, row.estimateMs);
    statusText = row.status === 'cancelling' ? 'Cancelling' : (row.stage ? formatStageLabel(row.stage) : 'Starting');
  } else if (row.status === 'completed') {
    statusText = 'Done';
  } else if (row.status === 'cancelled') {
    percent = estimateProgress(elapsed, row.estimateMs);
    statusText = 'Cancelled';
  } else {
    percent = estimateProgress(elapsed, row.estimateMs);
    statusText = 'Error';
  }
  return {
    ...row,
    active,
    percent,
    statusText,
    kindLabel: KIND_LABEL[row.kind] || row.kind.toUpperCase(),
    elapsed: formatClock(elapsed),
    time: active ? `${formatClock(elapsed)} / ~${formatClock(row.estimateMs)}` : formatClock(elapsed),
    timeTitle: active ? `typical ${formatClock(row.estimateMs)} (median of past runs, else a default)` : '',
  };
}

const rows = computed(() => processes.value.map(describe));
const lead = computed(() => rows.value.find((row) => row.active) || rows.value[rows.value.length - 1] || null);
const summary = computed(() => {
  const running = rows.value.filter((row) => row.active).length;
  const failed = rows.value.filter((row) => row.status === 'error').length;
  const parts = [`${running} running`];
  if (failed) parts.push(`${failed} failed`);
  return parts.join(' · ');
});
</script>

<style scoped>
.pdock {
  position: fixed;
  right: 0;
  z-index: 900;
  background: #000;
  border-top: 1px solid rgba(255, 255, 255, 0.14);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.06em;
  color: rgba(255, 255, 255, 0.46);
  transition: left 0.18s ease;
}

.pdock__head {
  display: flex;
  align-items: center;
  gap: 14px;
  width: 100%;
  height: 24px;
  padding: 0 12px;
  background: transparent;
  border: 0;
  color: inherit;
  font: inherit;
  letter-spacing: inherit;
  text-transform: uppercase;
  cursor: pointer;
}

.pdock__head:hover,
.pdock__head:focus-visible {
  color: #fff;
  outline: none;
}

.pdock__title {
  color: rgba(255, 255, 255, 0.9);
}

.pdock__lead {
  flex: 0 1 320px;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pdock__lead-bar {
  flex: 0 0 120px;
  height: 3px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.pdock__lead-bar i,
.prow__bar i {
  display: block;
  height: 100%;
  background: rgba(150, 150, 150, 0.55);
  transition: width 0.25s linear;
}

.pdock__chevron {
  margin-left: auto;
  transition: transform 0.15s ease;
  transform: rotate(180deg);
}

.pdock__chevron.open {
  transform: none;
}

.pdock__rows {
  list-style: none;
  margin: 0;
  padding: 0 0 4px;
  max-height: 34vh;
  overflow-y: auto;
}

.prow {
  display: grid;
  grid-template-columns: 78px minmax(90px, 170px) minmax(80px, 1fr) minmax(90px, 190px) minmax(70px, 220px) 118px 22px;
  align-items: center;
  column-gap: 14px;
  padding: 3px 12px;
}

.prow > span,
.prow__x {
  min-width: 0;
}

.prow__kind {
  color: rgba(255, 255, 255, 0.9);
}

.prow__model,
.prow__target,
.prow__stage {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-transform: uppercase;
}

.prow__target {
  text-transform: none;
}

.prow__bar {
  display: block;
  height: 3px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.prow__time {
  text-align: right;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

.prow__x {
  width: 22px;
  height: 18px;
  padding: 0;
  background: transparent;
  border: 0;
  color: inherit;
  font: inherit;
  font-size: 15px;
  line-height: 1;
  text-align: center;
  cursor: pointer;
}

.prow__x:hover:not(:disabled),
.prow__x:focus-visible {
  color: #fff;
  outline: none;
}

.prow__x:disabled {
  opacity: 0.4;
  cursor: default;
}

.prow__x--none {
  cursor: default;
}

.prow__preview {
  grid-column: 2 / -1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: rgba(255, 255, 255, 0.3);
}

.prow__preview--notice {
  color: #e0867d;
}

.prow--completed .prow__stage,
.prow--completed .prow__time {
  color: rgba(255, 255, 255, 0.7);
}

.prow--cancelled,
.prow--cancelling {
  opacity: 0.7;
}

.prow--error .prow__stage,
.prow--error .prow__preview {
  color: #e0867d;
}

@media (max-width: 900px) {
  .prow {
    grid-template-columns: 64px minmax(0, 1fr) 96px 22px;
  }
  .prow__model,
  .prow__stage,
  .prow__bar {
    display: none;
  }
  .prow__preview {
    grid-column: 1 / -1;
  }
}
</style>
