<template>
  <div v-if="modelValue" class="modal-overlay" @click.self="close">
    <div class="modal">
      <header class="modal-header">
        <h3>Gallery</h3>
        <button class="close" @click="close">×</button>
      </header>
      <section class="modal-body">
        <div class="left">
          <div class="list">
            <div
              v-for="(item, idx) in items"
              :key="item.basename"
              class="list-item"
              :class="{ active: idx === currentIndex }"
              @click="select(idx)"
            >
              <div class="title">{{ formatTitle(item) }}</div>
              <div class="time">{{ formatTime(item.timestamp) }}</div>
            </div>
          </div>
          <div v-if="current" class="details">
            <h4>Prompt</h4>
            <pre class="text">{{ current.prompt }}</pre>
            <h4>Answer</h4>
            <pre class="text">{{ current.answer }}</pre>
            <div class="actions">
              <button @click="loadCode">Load Code in Editor</button>
            </div>
          </div>
        </div>
        <div class="right">
          <img v-if="current" :src="imageSrc(current)" class="image" alt="organogram" />
          <div v-else class="empty">No item selected</div>
        </div>
      </section>
    </div>
  </div>
  
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useRuntimeConfig } from '#app'

const props = defineProps({
  modelValue: { type: Boolean, default: false }
})
const emit = defineEmits(['update:modelValue', 'load-code'])

const items = ref([])
const currentIndex = ref(0)
const current = computed(() => items.value[currentIndex.value] || null)

const config = useRuntimeConfig()
const apiBase = config.public.apiBase || 'http://localhost:10000/api'

function close() { emit('update:modelValue', false) }
function select(i) { currentIndex.value = i }

function imageSrc(item) {
  if (!item) return ''
  const url = item.image_url || ''
  if (url.startsWith('http')) return url
  // apiBase typically ends with /api; item.image_url starts with /api/...
  if (apiBase.endsWith('/api') && url.startsWith('/api/')) {
    return apiBase + url.substring(4)
  }
  return apiBase + url
}

async function load() {
  const res = await fetch(`${apiBase}/gallery/list`)
  const data = await res.json()
  items.value = data.items || []
  currentIndex.value = 0
}

function formatTitle(item) {
  return item?.basename || 'untitled'
}
function formatTime(ts) {
  if (!ts) return ''
  return ts
}

function onKey(e) {
  if (!props.modelValue) return
  if (!e.altKey) return
  if (e.key === 'ArrowUp') {
    e.preventDefault()
    if (currentIndex.value > 0) currentIndex.value--
  } else if (e.key === 'ArrowDown') {
    e.preventDefault()
    if (currentIndex.value < items.value.length - 1) currentIndex.value++
  }
}

function loadCode() {
  if (current.value?.code) {
    emit('load-code', current.value.code)
  }
}

watch(() => props.modelValue, (v) => { if (v) load() })
onMounted(() => window.addEventListener('keydown', onKey))
onUnmounted(() => window.removeEventListener('keydown', onKey))
</script>

<style scoped>
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 3000; }
.modal { width: 90vw; height: 80vh; background: #111; color: #fff; border: 1px solid #333; border-radius: 8px; display: flex; flex-direction: column; }
.modal-header { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-bottom: 1px solid #333; }
.close { background: transparent; color: #fff; border: none; font-size: 20px; cursor: pointer; }
.modal-body { display: flex; flex: 1; overflow: hidden; }
.left { width: 45%; border-right: 1px solid #333; display: flex; flex-direction: column; }
.right { flex: 1; display: flex; align-items: center; justify-content: center; background: #000; }
.image { max-width: 100%; max-height: 100%; object-fit: contain; }
.list { height: 40%; overflow: auto; border-bottom: 1px solid #333; }
.list-item { padding: 8px 12px; cursor: pointer; border-bottom: 1px solid #222; }
.list-item.active { background: #222; }
.title { font-weight: 600; }
.time { font-size: 12px; color: #aaa; }
.details { flex: 1; overflow: auto; padding: 12px; }
.text { white-space: pre-wrap; background: #0b0b0b; padding: 8px; border-radius: 4px; border: 1px solid #222; }
.actions { margin-top: 12px; }
.actions button { background: #4CAF50; border: none; padding: 8px 12px; border-radius: 4px; color: #fff; cursor: pointer; }
.empty { color: #777; }
</style>