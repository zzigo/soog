<template>
  <div class="vmenu">
    <button
      ref="buttonRef"
      type="button"
      class="pane-icon vmenu__button"
      :class="{ open }"
      :title="`${currentLabel}: change pod type`"
      aria-haspopup="listbox"
      :aria-expanded="String(open)"
      @click="toggle"
      @keydown.down.prevent="openMenu"
    >
      <svg class="pane-svg" viewBox="0 0 24 24" aria-hidden="true">
        <path :d="paneIcon(modelValue)" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" />
      </svg>
      <svg class="vmenu__caret" viewBox="0 0 10 10" aria-hidden="true">
        <path d="M2 4l3 3 3-3" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    </button>
    <Teleport to="body">
      <ul v-if="open" ref="listRef" class="vmenu__list" :style="position" role="listbox" @keydown.esc.prevent="close">
        <li v-for="option in options" :key="option.key" role="none">
          <button
            type="button"
            class="vmenu__option"
            :class="{ active: option.key === modelValue }"
            role="option"
            :aria-selected="String(option.key === modelValue)"
            @click="choose(option.key)"
          >
            <svg class="pane-svg" viewBox="0 0 24 24" aria-hidden="true">
              <path :d="paneIcon(option.key)" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" />
            </svg>
            <span>{{ option.label }}</span>
          </button>
        </li>
      </ul>
    </Teleport>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, ref } from 'vue';
import { paneIcon } from '~/composables/paneIcons';

// Blender-style pod type menu: an icon in the header, the names only when it opens.
const props = defineProps({
  modelValue: { type: String, required: true },
  options: { type: Array, required: true },
});
const emit = defineEmits(['update:modelValue']);

const open = ref(false);
const buttonRef = ref(null);
const listRef = ref(null);
const position = ref({});

const currentLabel = computed(() => props.options.find((option) => option.key === props.modelValue)?.label || 'Pod');

function place() {
  const rect = buttonRef.value?.getBoundingClientRect();
  if (!rect) return;
  const width = 190;
  const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8));
  position.value = { left: `${left}px`, top: `${rect.bottom + 4}px`, minWidth: `${width}px` };
}

function outside(event) {
  if (buttonRef.value?.contains(event.target) || listRef.value?.contains(event.target)) return;
  close();
}

function onKey(event) {
  if (event.key === 'Escape') close();
}

function openMenu() {
  place();
  open.value = true;
  window.addEventListener('pointerdown', outside, true);
  window.addEventListener('keydown', onKey);
  window.addEventListener('resize', close);
  nextTick(() => listRef.value?.querySelector('.active, button')?.focus());
}

function close() {
  open.value = false;
  window.removeEventListener('pointerdown', outside, true);
  window.removeEventListener('keydown', onKey);
  window.removeEventListener('resize', close);
}

function toggle() {
  if (open.value) close();
  else openMenu();
}

function choose(key) {
  close();
  if (key !== props.modelValue) emit('update:modelValue', key);
  buttonRef.value?.focus();
}

onBeforeUnmount(close);
</script>

<style>
/* Global: shared by every pod header, and the list is teleported to the body so a pod's overflow cannot clip it. */
.pane-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 26px;
  min-width: 26px;
  padding: 0 4px;
  background: transparent;
  border: 0;
  border-radius: 3px;
  color: rgba(255, 255, 255, 0.55);
  cursor: pointer;
}

.pane-icon:hover,
.pane-icon:focus-visible,
.pane-icon.open {
  background: rgba(255, 255, 255, 0.09);
  color: #fff;
  outline: none;
}

.pane-icon.pane-close:hover {
  color: #ff9d94;
}

.pane-svg {
  width: 16px;
  height: 16px;
  flex: none;
}

/* Header text actions (BUILD 3D, REGEN SKETCH...) share the borderless icon look. */
.workspace-pane .workspace-pane__header .pane-action {
  border: 0;
  border-radius: 3px;
  background: transparent;
  color: rgba(255, 255, 255, 0.55);
}

.workspace-pane .workspace-pane__header .pane-action:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.09);
  color: #fff;
}

.vmenu__button {
  gap: 2px;
}

.vmenu__caret {
  width: 8px;
  height: 8px;
  flex: none;
  opacity: 0.7;
}

.vmenu__list {
  position: fixed;
  z-index: 2100;
  margin: 0;
  padding: 4px;
  list-style: none;
  background: #0b0b0b;
  border: 1px solid rgba(255, 255, 255, 0.16);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
  font-family: 'IBM Plex Mono', monospace;
}

.vmenu__option {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  height: 28px;
  padding: 0 10px;
  background: transparent;
  border: 0;
  color: rgba(255, 255, 255, 0.62);
  font: inherit;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-align: left;
  text-transform: uppercase;
  cursor: pointer;
}

.vmenu__option:hover,
.vmenu__option:focus-visible {
  background: rgba(255, 255, 255, 0.08);
  color: #fff;
  outline: none;
}

.vmenu__option.active {
  color: #fff;
}

.vmenu__option .pane-svg {
  width: 16px;
  height: 16px;
  flex: none;
}
</style>
