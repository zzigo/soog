<template>
  <div v-if="modelValue" class="modal-overlay" @click.self="close">
    <div class="modal">
      <header class="modal-header">
        <div class="header-left">
          <h3>Gallery</h3>
        </div>
        <div class="header-center">
          <div v-if="current && !renaming" class="actions">
            <button @click="loadCode">Load Code</button>
            <a v-if="current.stl_url" :href="fileHref(current.stl_url)" class="download" download>STL</a>
            <button @click="toggleRename">Rename</button>
            <button @click="deleteItem" class="delete">Delete</button>
          </div>
          <div v-if="current && renaming" class="rename-container">
            <input type="text" v-model="newName" @keyup.enter="renameItem" placeholder="New name..." />
            <button @click="renameItem">Save</button>
            <button @click="renaming = false">Cancel</button>
          </div>
        </div>
        <div class="header-right">
          <button class="close" @click="close">×</button>
        </div>
      </header>
      <section class="modal-body">
        <div class="left">
          <div class="list" :key="galleryKey">
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
            <h4>Summary</h4>
            <pre class="text">{{ current.summary || current.answer }}</pre>
          </div>
        </div>
        <div class="right">
          <div class="tabs">
            <button :disabled="!current?.image_url" :class="{active: rightTab==='plot', disabled: !current?.image_url}" @click="rightTab='plot'">Organogram</button>
            <button :disabled="!current?.stl_url" :class="{active: rightTab==='stl', disabled: !current?.stl_url}" @click="rightTab='stl'">3D Model</button>
          </div>
          <div class="viewer">
            <template v-if="current">
              <img v-if="rightTab==='plot' && current?.image_url" :src="imageSrc(current)" class="image" alt="organogram" />
              <ClientOnly v-else>
                <StlViewer v-if="current?.stl_url" :url="fileHref(current.stl_url)" />
                <div v-else class="empty">No STL available</div>
              </ClientOnly>
            </template>
            <div v-else class="empty">No item selected</div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useRuntimeConfig } from '#app'
import StlViewer from '~/components/StlViewer.vue'

const props = defineProps({
  modelValue: { type: Boolean, default: false }
})
const emit = defineEmits(['update:modelValue', 'load-code'])

const items = ref([])
const currentIndex = ref(0)
const current = computed(() => items.value[currentIndex.value] || null)
const rightTab = ref('plot')
const renaming = ref(false)
const newName = ref('')
const galleryKey = ref(0)

watch(() => current.value, (val) => {
  if (!val) return
  if (val.image_url) rightTab.value = 'plot'
  else if (val.stl_url) rightTab.value = 'stl'
  else rightTab.value = 'plot'
  renaming.value = false
})

const config = useRuntimeConfig()
const apiBase = config.public.apiBase || 'http://localhost:10000/api'

function close() { emit('update:modelValue', false) }
function select(i) { currentIndex.value = i }

function imageSrc(item) {
  if (!item) return ''
  const url = item.image_url || ''
  if (url.startsWith('http')) return url

  const offloadApiBase = apiBase.endsWith('/api') ? apiBase.slice(0, -4) : apiBase;
  if (url.startsWith('/offload')) {
      return offloadApiBase + url;
  }
      
  if (apiBase.endsWith('/api') && url.startsWith('/api/')) {
    return apiBase + url.substring(4)
  }
  return apiBase + url
}

function fileHref(url) {
  if (!url) return ''
  if (url.startsWith('http')) return url
      
  const offloadApiBase = apiBase.endsWith('/api') ? apiBase.slice(0, -4) : apiBase;
  if (url.startsWith('/offload')) {
      return offloadApiBase + url;
  }

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
  if (!item?.basename) return 'untitled'
  const parts = item.basename.split('_')
  if (parts.length > 1) {
    return parts.slice(1).join('_')
  }
  return item.basename
}

function formatTime(ts) {
  if (!ts) return ''
  return ts
}

function onKey(e) {
  if (!props.modelValue) return
  if (renaming.value) return 
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

function toggleRename() {
  if (!current.value) return
  newName.value = formatTitle(current.value)
  renaming.value = true
}

async function deleteItem() {
  if (!current.value) return
  if (!confirm(`Are you sure you want to delete ${current.value.basename}?`)) return

  const res = await fetch(`${apiBase}/gallery/item/${current.value.basename}`, {
    method: 'DELETE'
  })

  if (res.ok) {
    const deletedIndex = currentIndex.value
    items.value.splice(deletedIndex, 1)
    if (currentIndex.value >= items.value.length) {
      currentIndex.value = items.value.length - 1
    }
    galleryKey.value++
  } else {
    const error = await res.json()
    alert(`Error deleting item: ${error.error}`)
  }
}

async function renameItem() {
  if (!current.value || !newName.value.trim()) return

  const res = await fetch(`${apiBase}/gallery/item/${current.value.basename}/rename`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ newName: newName.value.trim() })
  })

  if (res.ok) {
    const { newBasename } = await res.json()
    items.value[currentIndex.value].basename = newBasename
    renaming.value = false
    galleryKey.value++ // Force re-render
  } else {
    const error = await res.json()
    alert(`Error renaming item: ${error.error}`)
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
.header-left, .header-right { flex: 0 0 100px; }
.header-right { text-align: right; }
.header-center { flex: 1; display: flex; justify-content: center; gap: 8px; }
.close { background: transparent; color: #fff; border: none; font-size: 20px; cursor: pointer; }
.modal-body { display: flex; flex: 1; overflow: hidden; }
.left { width: 45%; border-right: 1px solid #333; display: flex; flex-direction: column; }
.right { flex: 1; display: flex; flex-direction: column; background: #000; }
.tabs { display: flex; gap: 8px; padding: 8px; border-bottom: 1px solid #222; }
.tabs button { background: #111; color: #eee; border: 1px solid #333; padding: 6px 10px; border-radius: 4px; cursor: pointer; }
.tabs button.active { background: #1f1f1f; border-color: #666; }
.tabs button.disabled { opacity: 0.5; cursor: not-allowed; }
.viewer { flex: 1; display: flex; align-items: center; justify-content: center; }
.image { max-width: 100%; max-height: 100%; object-fit: contain; }
.list { height: 40%; overflow: auto; border-bottom: 1px solid #333; }
.list-item { 
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px; 
  cursor: pointer; 
  border-bottom: 1px solid #222; 
}
.list-item.active { background: #222; }
.title { 
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-grow: 1;
}
.time { 
  font-size: 12px; 
  opacity: 0.5;
  font-style: italic;
  flex-shrink: 0;
  margin-left: 16px;
}
.details { flex: 1; overflow: auto; padding: 12px; }
.text { white-space: pre-wrap; background: #0b0b0b; padding: 8px; border-radius: 4px; border: 1px solid #222; }
.actions { display: flex; gap: 8px; flex-wrap: wrap; }
.actions button, .actions a { background: #111; color: #eee; border: 1px solid #333; padding: 6px 10px; border-radius: 4px; cursor: pointer; text-decoration: none; }
.actions button:hover, .actions a:hover { background: #1f1f1f; border-color: #666; }
.actions button.delete { background: #900; }
.rename-container { display: flex; gap: 8px; }
.rename-container input { background: #222; border: 1px solid #444; color: #fff; padding: 8px; border-radius: 4px; }
.rename-container button { background: #111; color: #eee; border: 1px solid #333; padding: 6px 10px; border-radius: 4px; cursor: pointer; }
.rename-container button:hover { background: #1f1f1f; border-color: #666; }
.empty { color: #777; }
</style>