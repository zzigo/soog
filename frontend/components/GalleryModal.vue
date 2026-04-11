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
            <button v-if="current.stl_url" @click="downloadCurrentStl">STL</button>
            <button 
              v-if="current" 
              @click="remakeSketch" 
              class="remake-header-btn" 
              :disabled="remakingSketch"
              title="Regenerate inferred sketch image"
            >
              GENERATE
            </button>
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
              v-for="group in groupedItems"
              :key="group.groupId"
              class="list-item"
              :class="{ active: currentGroupId === group.groupId }"
              @click="selectGroup(group)"
            >
              <div class="group-head">
                <div class="title">{{ group.title }}</div>
                <div class="item-meta">
                  <div v-if="group.hasAnyStl" class="stl-badge">STL</div>
                  <div class="time">{{ formatTime(group.latestTimestamp) }}</div>
                </div>
              </div>
              <div class="versions">
                <button
                  v-for="version in group.versions"
                  :key="version.basename"
                  class="version-chip"
                  :class="{ active: current?.basename === version.basename }"
                  @click.stop="selectVersion(version.basename)"
                >
                  {{ version.displayVersion }}
                </button>
              </div>
            </div>
          </div>

          <div v-if="current" class="details">
            <h4>Prompt</h4>
            <pre class="text">{{ current.prompt }}</pre>
            <h4>Summary</h4>
            <div class="text markdown" v-html="currentSummaryHtml"></div>
          </div>
        </div>

        <div class="right">
          <div v-if="current" class="split-view">
            <div class="media-row">
              <section class="viewer-panel">
                <div class="viewer-title-row">
                  <h4 class="viewer-title">Organogram</h4>
                </div>
                <div class="viewer">
                  <img v-if="current?.image_url" :src="imageSrc(current)" class="image" alt="organogram" />
                  <div v-else class="empty">No organogram image</div>
                </div>
              </section>
              <section class="viewer-panel">
                <div class="viewer-title-row">
                  <h4 class="viewer-title">Sketch</h4>
                </div>
                <div class="viewer">
                  <img v-if="current?.sketch_url" :src="sketchSrc(current)" class="image" :key="current?.sketch_url" alt="sketch render" />
                  <div v-else class="empty">No sketch available</div>
                </div>
              </section>
            </div>
            <section class="viewer-panel viewer-panel--full">
              <div class="viewer-title-row">
                <h4 class="viewer-title">3D Model</h4>
              </div>
              <div class="viewer">
                <ClientOnly>
                  <StlViewer v-if="current?.stl_url" :url="fileHref(current.stl_url)" />
                  <div v-else class="empty">No STL available</div>
                </ClientOnly>
              </div>
            </section>
          </div>
          <div v-else class="empty full-empty">No item selected</div>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useRuntimeConfig } from '#app'
import { marked } from 'marked'
import StlViewer from '~/components/StlViewer.vue'

const props = defineProps({
  modelValue: { type: Boolean, default: false }
})
const emit = defineEmits(['update:modelValue', 'load-code'])

const items = ref([])
const selectedBasename = ref('')
const renaming = ref(false)
const newName = ref('')
const galleryKey = ref(0)
const remakingSketch = ref(false)

const current = computed(() => items.value.find((item) => item.basename === selectedBasename.value) || null)
const currentSummaryHtml = computed(() => {
  const source = current.value?.summary || current.value?.answer || ''
  if (!source) return ''
  return marked(source)
})

const config = useRuntimeConfig()
const apiBase = config.public.apiBase || 'http://localhost:10000/api'

function close() {
  emit('update:modelValue', false)
}

function inferGroupId(item) {
  return (item?.group_id || item?.basename || '').trim()
}

function inferTitle(item) {
  if (item?.title && String(item.title).trim()) return String(item.title).trim()
  if (item?.title_slug) return String(item.title_slug).replace(/_/g, '-')
  const basename = String(item?.basename || '')
  if (!basename) return 'untitled'
  const body = basename.replace(/^\d{8}-\d{6}_/, '')
  return body.replace(/_v\d+(?:_\d+)?(?:_\d+)?$/i, '').replace(/_/g, '-')
}

function numericVersion(item) {
  const fromIndex = Number(item?.version_index)
  if (Number.isFinite(fromIndex) && fromIndex > 0) return Math.trunc(fromIndex)

  const versionText = String(item?.version || '').trim().toLowerCase().replace(/^v/, '')
  const plain = versionText.match(/^(\d+)$/)
  if (plain) return Math.max(1, Number(plain[1]))

  const legacy = versionText.match(/^(\d+)\.(\d+)$/)
  if (legacy) {
    const major = Number(legacy[1])
    const minor = Number(legacy[2])
    if (major === 1) return Math.max(1, minor + 1)
  }
  return 1
}

const groupedItems = computed(() => {
  const groups = new Map()
  const sorted = [...items.value].sort((a, b) => String(b?.timestamp || '').localeCompare(String(a?.timestamp || '')))

  for (const item of sorted) {
    const groupId = inferGroupId(item)
    if (!groupId) continue
    if (!groups.has(groupId)) {
      groups.set(groupId, {
        groupId,
        title: inferTitle(item),
        versions: [],
        latestTimestamp: item?.timestamp || '',
        hasAnyStl: false
      })
    }
    const group = groups.get(groupId)
    group.versions.push(item)
    if (String(item?.timestamp || '') > String(group.latestTimestamp || '')) {
      group.latestTimestamp = item.timestamp
    }
    group.hasAnyStl = group.hasAnyStl || Boolean(item?.stl_url)
  }

  const result = []
  for (const group of groups.values()) {
    const ascending = [...group.versions].sort((a, b) => {
      const av = numericVersion(a)
      const bv = numericVersion(b)
      if (av !== bv) return av - bv
      return String(a?.timestamp || '').localeCompare(String(b?.timestamp || ''))
    })
    group.versions = ascending.map((item, idx) => ({
      ...item,
      displayVersion: numericVersion(item) || idx + 1
    }))
    group.latest = group.versions[group.versions.length - 1] || null
    result.push(group)
  }

  result.sort((a, b) => String(b.latestTimestamp || '').localeCompare(String(a.latestTimestamp || '')))
  return result
})

const currentGroup = computed(() => {
  if (!current.value) return null
  return groupedItems.value.find((group) => group.versions.some((version) => version.basename === current.value.basename)) || null
})
const currentGroupId = computed(() => currentGroup.value?.groupId || '')

watch(() => current.value, (value) => {
  if (!value) return
  renaming.value = false
})

function selectVersion(basename) {
  selectedBasename.value = basename
}

function selectGroup(group) {
  if (!group) return
  const latest = group.latest || group.versions[group.versions.length - 1]
  if (latest?.basename) selectedBasename.value = latest.basename
}

function imageSrc(item) {
  if (!item) return ''
  return assetHref(item.image_url || '') + (item.image_url?.includes('?') ? '&' : '?') + 't=' + Date.now()
}

function sketchSrc(item) {
  if (!item) return ''
  return assetHref(item.sketch_url || '') + (item.sketch_url?.includes('?') ? '&' : '?') + 't=' + Date.now()
}

function assetHref(url) {
  if (!url) return ''
  if (url.startsWith('http')) return url

  const offloadApiBase = apiBase.endsWith('/api') ? apiBase.slice(0, -4) : apiBase
  if (url.startsWith('/offload')) return offloadApiBase + url
  if (apiBase.endsWith('/api') && url.startsWith('/api/')) return apiBase + url.substring(4)
  return apiBase + url
}

function fileHref(url) {
  return assetHref(url)
}

async function load(preferredBasename = '') {
  const res = await fetch(`${apiBase}/gallery/list`)
  const data = await res.json()
  const next = (data.items || []).slice().sort((a, b) => String(b?.timestamp || '').localeCompare(String(a?.timestamp || '')))
  items.value = next

  if (preferredBasename && next.some((item) => item.basename === preferredBasename)) {
    selectedBasename.value = preferredBasename
    return
  }
  selectedBasename.value = next[0]?.basename || ''
}

function formatTime(ts) {
  return ts || ''
}

function loadCode() {
  if (current.value) {
    emit('load-code', { ...current.value })
  }
}

function toggleRename() {
  if (!currentGroup.value) return
  newName.value = currentGroup.value.title
  renaming.value = true
}

async function downloadCurrentStl() {
  if (!current.value?.stl_url) return
  try {
    const res = await fetch(fileHref(current.value.stl_url))
    if (!res.ok) {
      const payload = await res.json().catch(() => ({}))
      throw new Error(payload?.error || `STL download failed (${res.status})`)
    }
    const blob = await res.blob()
    const blobUrl = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = blobUrl
    link.download = `${current.value.basename || 'model'}.stl`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(blobUrl)
  } catch (e) {
    alert(`Error downloading STL: ${e?.message || 'unknown error'}`)
  }
}

async function deleteItem() {
  if (!current.value) return
  if (!confirm(`Delete version ${current.value.version || ''} from "${currentGroup.value?.title || current.value.basename}"?`)) return

  const res = await fetch(`${apiBase}/gallery/item/${current.value.basename}`, { method: 'DELETE' })
  if (!res.ok) {
    const payload = await res.json().catch(() => ({}))
    alert(`Error deleting version: ${payload?.error || res.status}`)
    return
  }

  const keepGroupId = currentGroupId.value
  await load()
  const sameGroup = groupedItems.value.find((group) => group.groupId === keepGroupId)
  if (sameGroup?.latest?.basename) {
    selectedBasename.value = sameGroup.latest.basename
  }
  galleryKey.value++
}

async function renameItem() {
  if (!currentGroup.value || !newName.value.trim()) return

  const oldSelected = current.value?.basename || ''
  const groupId = currentGroup.value.groupId
  const res = await fetch(`${apiBase}/gallery/group/${encodeURIComponent(groupId)}/rename`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ newName: newName.value.trim() })
  })

  const payload = await res.json().catch(() => ({}))
  if (!res.ok) {
    alert(`Error renaming versions: ${payload?.error || res.status}`)
    return
  }

  const mapping = new Map((payload.updated || []).map((entry) => [entry.old, entry.new]))
  const preferred = mapping.get(oldSelected) || ''
  await load(preferred)
  renaming.value = false
  galleryKey.value++
}

async function remakeSketch() {
  if (!current.value || remakingSketch.value) return
  remakingSketch.value = true
  try {
    const res = await fetch(`${apiBase}/gallery/item/${current.value.basename}/remake_sketch`, {
      method: 'POST'
    })
    const payload = await res.json()
    if (!res.ok) {
      throw new Error(payload.error || 'Failed to remake sketch')
    }
    // Update local item
    if (current.value) {
      current.value.sketch_url = payload.sketch_url
      current.value.sketch_prompt = payload.sketch_prompt
      current.value.sketch_model = payload.sketch_model
    }
    // Reload full list to sync with disk metadata
    await load(current.value?.basename)
  } catch (e) {
    alert(`Error remaking sketch: ${e.message}`)
  } finally {
    remakingSketch.value = false
  }
}

function onKey(e) {
  if (!props.modelValue || renaming.value || !e.altKey) return
  if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
  e.preventDefault()

  const ordered = [...items.value].sort((a, b) => String(b?.timestamp || '').localeCompare(String(a?.timestamp || '')))
  if (!ordered.length) return
  let idx = ordered.findIndex((item) => item.basename === selectedBasename.value)
  if (idx < 0) idx = 0
  if (e.key === 'ArrowUp') idx = Math.max(0, idx - 1)
  if (e.key === 'ArrowDown') idx = Math.min(ordered.length - 1, idx + 1)
  selectedBasename.value = ordered[idx].basename
}

watch(() => props.modelValue, (value) => {
  if (value) load(selectedBasename.value)
})

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
.split-view { flex: 1; display: grid; grid-template-rows: minmax(220px, 1fr) minmax(260px, 1fr); min-height: 0; }
.media-row { display: grid; grid-template-columns: 1fr 1fr; min-height: 0; }
.viewer-panel { display: flex; flex-direction: column; min-height: 0; }
.viewer-panel + .viewer-panel { border-top: 1px solid #222; }
.media-row .viewer-panel + .viewer-panel { border-top: none; border-left: 1px solid #222; }
.viewer-panel--full { border-top: 1px solid #222; }
.viewer-title-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-right: 10px;
  border-bottom: 1px solid #1a1a1a;
}
.viewer-title {
  margin: 0;
  padding: 8px 10px;
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(255,255,255,0.7);
  border-bottom: none;
}
.remake-header-btn {
  background: #333;
  color: #fff;
  border: 1px solid #444;
  padding: 6px 12px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s;
  letter-spacing: 0.05em;
}
.remake-header-btn:hover:not(:disabled) { 
  background: #444;
  border-color: #666;
  box-shadow: 0 0 8px rgba(152, 255, 95, 0.2);
}
.remake-header-btn:disabled { 
  opacity: 0.8; 
  cursor: not-allowed; 
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0% { background: #333; }
  50% { background: #555; }
  100% { background: #333; }
}
.remake-btn {
  background: #444;
  color: #fff;
  border: 1px solid #555;
  border-radius: 4px;
  padding: 2px 10px;
  font-size: 10px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}
.remake-btn:hover:not(:disabled) { 
  background: #555;
  border-color: #777;
}
.remake-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.viewer { flex: 1; display: flex; align-items: center; justify-content: center; min-height: 0; }
.image { max-width: 100%; max-height: 100%; object-fit: contain; }

.list { height: 44%; overflow: auto; border-bottom: 1px solid #333; }
.list-item {
  padding: 9px 12px;
  cursor: pointer;
  border-bottom: 1px solid #222;
}
.list-item.active { background: #1b1b1b; }
.group-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
}
.title {
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-grow: 1;
}
.item-meta {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  justify-content: center;
  flex-shrink: 0;
  gap: 2px;
}
.versions {
  margin-top: 7px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.version-chip {
  min-width: 24px;
  border: 1px solid #3a3a3a;
  background: transparent;
  color: rgba(255, 255, 255, 0.9);
  font-size: 11px;
  line-height: 1;
  padding: 4px 6px;
  cursor: pointer;
}
.version-chip.active {
  border-color: #98ff5f;
  color: #98ff5f;
}
.version-chip:hover {
  border-color: #8a8a8a;
}
.stl-badge {
  font-size: 8px;
  line-height: 1;
  color: #ff9a2f;
  letter-spacing: 0.08em;
  font-weight: 700;
  text-transform: uppercase;
  text-align: right;
}
.time {
  font-size: 12px;
  opacity: 0.5;
  font-style: italic;
  text-align: right;
}
.details { flex: 1; overflow: auto; padding: 12px; }
.text { white-space: pre-wrap; background: #0b0b0b; padding: 8px; border-radius: 4px; border: 1px solid #222; }
.markdown { white-space: normal; line-height: 1.5; }
.markdown :deep(p) { margin: 0 0 0.7em 0; }
.markdown :deep(ul),
.markdown :deep(ol) { margin: 0.3em 0 0.7em 1.2em; }
.markdown :deep(li) { margin: 0.2em 0; }
.markdown :deep(code) { background: rgba(255, 255, 255, 0.08); padding: 1px 4px; border-radius: 3px; }
.markdown :deep(pre) { background: #090909; border: 1px solid #1f1f1f; padding: 8px; border-radius: 4px; overflow-x: auto; }
.markdown :deep(pre code) { background: transparent; padding: 0; }
.actions { display: flex; gap: 8px; flex-wrap: wrap; }
.actions button {
  background: #111;
  color: #eee;
  border: 1px solid #333;
  padding: 6px 10px;
  border-radius: 4px;
  cursor: pointer;
  text-decoration: none;
}
.actions button:hover { background: #1f1f1f; border-color: #666; }
.actions button.delete { background: #900; }
.rename-container { display: flex; gap: 8px; }
.rename-container input { background: #222; border: 1px solid #444; color: #fff; padding: 8px; border-radius: 4px; }
.rename-container button { background: #111; color: #eee; border: 1px solid #333; padding: 6px 10px; border-radius: 4px; cursor: pointer; }
.rename-container button:hover { background: #1f1f1f; border-color: #666; }
.empty { color: #777; }
.full-empty { display: flex; align-items: center; justify-content: center; flex: 1; }

@media (max-width: 900px) {
  .modal-body { flex-direction: column; }
  .left { width: 100%; height: 42%; border-right: none; border-bottom: 1px solid #333; }
  .media-row { grid-template-columns: 1fr; grid-template-rows: 1fr 1fr; }
  .media-row .viewer-panel + .viewer-panel { border-left: none; border-top: 1px solid #222; }
}
</style>
