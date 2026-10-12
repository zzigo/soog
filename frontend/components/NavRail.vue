<template>
  <div class="rail-root">
    <div v-if="autoHide" class="rail-hotzone" aria-hidden="true" @mouseenter="reveal" @mouseleave="scheduleHide"></div>
    <nav
      class="rail"
      :class="{ 'rail--expanded': expanded, 'rail--hidden': autoHide && hidden, 'rail--float': autoHide }"
      :style="{ bottom: `${bottom}px` }"
      aria-label="Tools"
      @mouseenter="reveal"
      @mouseleave="scheduleHide"
      @focusin="reveal"
      @focusout="scheduleHide"
    >
      <div class="rail__group">
        <slot name="head" />
        <NuxtLink
          v-for="page in PAGES"
          :key="page.to"
          :to="page.to"
          class="rail-item"
          :class="{ active: isActive(page.to) }"
          :title="page.title"
        >
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true">
            <path :d="page.icon" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" />
          </svg>
          <span class="rail-label">{{ page.label }}</span>
        </NuxtLink>
        <slot name="tools" />
      </div>
      <div class="rail__group">
        <slot name="bottom" />
      </div>
    </nav>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue';
import { useRoute } from 'vue-router';

// One narrow navigation rail for every page. Pages add their own controls through the slots.
// With autoHide (full-screen pages such as the concert stage) it slides away after one second
// and returns when the pointer reaches the left edge.
const props = defineProps({
  expanded: { type: Boolean, default: false },
  autoHide: { type: Boolean, default: false },
  bottom: { type: Number, default: 0 },
});

const PAGES = [
  { to: '/', label: 'SOOG', title: 'SOOG (Alt+1)', icon: 'M4.5 4.5h15v15h-15zM4.5 9.5h15M9.5 9.5v10' },
  { to: '/somap', label: 'Somap', title: 'Somap registry (Alt+2)', icon: 'M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18M12 8.5a3.5 3.5 0 1 0 0 7a3.5 3.5 0 0 0 0-7M12 3v3M12 18v3M3 12h3M18 12h3' },
  { to: '/concert', label: 'Concert', title: 'Concert room, full page (Alt+3)', icon: 'M3 20h18M5 20V10l7-6l7 6v10M9 20v-5h6v5' },
];

const route = useRoute();
const isActive = (to) => (to === '/' ? route.path === '/' : route.path.startsWith(to));

const hidden = ref(false);
let hideTimer = null;

function clearTimer() {
  if (hideTimer) clearTimeout(hideTimer);
  hideTimer = null;
}

function reveal() {
  if (!props.autoHide) return;
  clearTimer();
  hidden.value = false;
}

function scheduleHide() {
  if (!props.autoHide) return;
  clearTimer();
  hideTimer = setTimeout(() => { hidden.value = true; }, 1000);
}

onMounted(() => scheduleHide());
onBeforeUnmount(clearTimer);
</script>

<style>
/* Global on purpose: pages put their own .rail-item buttons in the slots. */
.rail {
  position: fixed;
  top: 0;
  left: 0;
  width: 44px;
  z-index: 130;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  background: #000;
  border-right: 1px solid rgba(255, 255, 255, 0.14);
  transition: width 0.18s ease, transform 0.18s ease;
  font-family: 'IBM Plex Mono', monospace;
}

.rail--expanded {
  width: 210px;
}

.rail--hidden {
  transform: translateX(-100%);
}

/* Over full-screen content such as the concert stage overlays. */
.rail--float {
  z-index: 2000;
}

.rail-hotzone {
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  width: 10px;
  z-index: 1999;
}

.rail__group {
  display: flex;
  flex-direction: column;
  align-items: stretch;
}

.rail-item {
  height: 38px;
  width: 100%;
  box-sizing: border-box;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 12px;
  background: transparent;
  border: 0;
  border-left: 1px solid transparent;
  color: #666;
  font: inherit;
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  text-decoration: none;
  white-space: nowrap;
  overflow: hidden;
  cursor: pointer;
}

.rail-item .icon {
  flex: none;
  width: 20px;
  height: 20px;
}

.rail-item:hover,
.rail-item.active {
  color: #fff;
}

.rail-item.active {
  border-left-color: #fff;
}

.rail-item:focus-visible {
  outline: 1px solid #fff;
  outline-offset: -1px;
}

.rail-label {
  display: none;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.rail--expanded .rail-label {
  display: inline;
}
</style>
