<template>
  <aside v-if="modelValue" class="gallery-dock" aria-label="Gallery">
    <header class="dock-header">
      <span class="gallery-label">GALLERY</span>
      <input type="text" v-model="searchQuery" placeholder="SEARCH..." class="search-input" />
      <button class="close" title="Close gallery" @click="close">×</button>
    </header>

    <div v-if="current && !renaming" class="actions">
      <button @click="loadCode">LOAD</button>
      <button v-if="current.stl_url" @click="downloadCurrentStl">STL</button>
      <button @click="toggleRename">RENAME</button>
      <button @click="deleteItem" class="delete">DELETE</button>
    </div>
    <div v-if="current && renaming" class="rename-container">
      <input type="text" v-model="newName" @keyup.enter="renameItem" placeholder="NEW NAME..." />
      <button @click="renameItem">SAVE</button>
      <button @click="renaming = false">ESC</button>
    </div>

    <div class="list-container" :key="galleryKey" ref="listContainer">
      <div
        v-for="group in groupedItems"
        :key="group.groupId"
        :id="'group-' + group.groupId"
        class="group-item"
        :class="{ active: currentGroupId === group.groupId }"
        @click="selectGroup(group)"
      >
        <div class="group-row"><img v-if="group.latest.image_url" :src="imageSrc(group.latest)" class="nano-thumb" />
          <input 
            type="checkbox" 
            :checked="group.latest.featured" 
            @click.stop="toggleFeatured(group.latest)"
            class="feat-check"
          />
          <div class="title">{{ group.title }}</div>
          <div class="meta-info">
            <div class="badges">
              <span v-if="group.hasAnySound" class="b-icon">S</span>
              <span v-if="group.hasAnySketch" class="b-icon">K</span>
              <span v-if="group.hasAnyPinn" class="b-icon">P</span>
              <span v-if="group.hasAnyStl" class="b-icon">3D</span>
            </div>
          </div>
        </div>
        <div class="version-list">
          <span
            v-for="version in group.versions"
            :key="version.basename"
            class="v-tag"
            :class="{ active: current?.basename === version.basename }"
            @click.stop="selectVersion(version.basename)"
          >
            V{{ version.displayVersion }}
          </span>
        </div>
      </div>
    </div>


      <div v-if="current" class="metadata">
        <div class="meta-section">
          <label>PROMPT</label>
          <div class="raw-text">{{ current.prompt }}</div>
        </div>
        <div class="meta-section">
          <label>SUMMARY</label>
          <div class="summary-text markdown" v-html="currentSummaryHtml"></div>
        </div>
      </div>

    <footer v-if="isProcessingAny" class="process-footer">
      <div class="terminal-line">
        <span class="blink">_</span> {{ consoleDisplayMessage }}
        <span v-if="reasoningPreview" class="reason-txt">{{ reasoningPreview }}</span>
      </div>
    </footer>
  </aside>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useRuntimeConfig } from '#app'
import { marked } from 'marked'
import StlViewer from '~/components/StlViewer.vue'
import ModulusHeatmap from '~/components/ModulusHeatmap.vue'
import { useApi } from '~/composables/useApi'
import { useItemTools } from '~/composables/useItemTools'

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  initialBasename: { type: String, default: '' }
})
const emit = defineEmits(['update:modelValue', 'load-code', 'preview'])

const { apiBase } = useApi()
const config = useRuntimeConfig()
const { authHeaders } = useSoogAuth()

const items = ref([])
const selectedBasename = ref('')
const searchQuery = ref('')
const renaming = ref(false)
const newName = ref('')
const galleryKey = ref(0)
const listContainer = ref(null)

// Tool state (sketch, 3D, acoustics, sound) lives in useItemTools so the workspace pods can drive it too.
const {
  remakingSketch, generatingSound, remakingModulus, remakingLRM,
  isBusy: isProcessingAny, progressStage, reasoningPreview, onItemUpdated,
} = useItemTools()

const consoleDisplayMessage = computed(() => {
  if (progressStage.value) return formatProgressStage(progressStage.value)
  if (generatingSound.value) return 'SYNTHESIZING AUDIO...'
  if (remakingSketch.value) return 'INFERRING SKETCH...'
  if (remakingModulus.value) return 'SOLVING ACOUSTICS...'
  if (remakingLRM.value) return 'RECONSTRUCTING 3D...'
  return 'PROCESSING...'
})

// A pod finished a tool on an item: refresh the list and keep it selected.
const stopItemUpdates = onItemUpdated((item) => { if (props.modelValue) load(item.basename) })
onUnmounted(stopItemUpdates)

const current = computed(() => items.value.find((item) => item.basename === selectedBasename.value) || null)
const currentSummaryHtml = computed(() => {
  const source = current.value?.summary || current.value?.answer || ''
  if (!source) return ''
  return marked(source)
})

function close() { emit('update:modelValue', false) }

// Tell the workspace which item the user is looking at so its panes can show it.
function announce() { if (current.value) emit('preview', { ...current.value }) }

function inferGroupId(item) { return (item?.group_id || item?.basename || '').trim() }

function inferTitle(item) {
  if (item?.title && String(item.title).trim()) return String(item.title).trim()
  if (item?.title_slug) return String(item.title_slug).replace(/_/g, '-')
  const basename = String(item?.basename || '')
  if (!basename) return 'UNTITLED'
  const body = basename.replace(/^\d{8}-\d{6}_/, '')
  return body.replace(/_v\d+(?:_\d+)?(?:_\d+)?$/i, '').replace(/_/g, '-').toUpperCase()
}

function numericVersion(item) {
  const fromIndex = Number(item?.version_index)
  if (Number.isFinite(fromIndex) && fromIndex > 0) return Math.trunc(fromIndex)
  const versionText = String(item?.version || '').trim().toLowerCase().replace(/^v/, '')
  const plain = versionText.match(/^(\d+)$/)
  if (plain) return Math.max(1, Number(plain[1]))
  return 1
}

const groupedItems = computed(() => {
  const groups = new Map()
  const query = searchQuery.value.toLowerCase().trim()
  const filteredItems = items.value.filter(item => {
    if (!query) return true
    const title = inferTitle(item).toLowerCase()
    const basename = (item.basename || '').toLowerCase()
    const prompt = (item.prompt || '').toLowerCase()
    return title.includes(query) || basename.includes(query) || prompt.includes(query)
  })

  const sorted = [...filteredItems].sort((a, b) => String(b?.timestamp || '').localeCompare(String(a?.timestamp || '')))

  for (const item of sorted) {
    const groupId = inferGroupId(item)
    if (!groupId) continue
    if (!groups.has(groupId)) {
      groups.set(groupId, {
        groupId,
        title: inferTitle(item),
        versions: [],
        latestTimestamp: item?.timestamp || '',
        hasAnyStl: false,
        hasAnySketch: false,
        hasAnySound: false,
        hasAnyPinn: false
      })
    }
    const group = groups.get(groupId)
    group.versions.push(item)
    if (String(item?.timestamp || '') > String(group.latestTimestamp || '')) group.latestTimestamp = item.timestamp
    group.hasAnyStl = group.hasAnyStl || Boolean(item?.stl_url)
    group.hasAnySketch = group.hasAnySketch || Boolean(item?.sketch_url)
    group.hasAnySound = group.hasAnySound || (item?.sound_samples && item.sound_samples.length > 0)
    group.hasAnyPinn = group.hasAnyPinn || Boolean(item?.modulus)
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
      displayVersion: numericVersion(item) || idx + 1,
      featured: item.featured || false
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

function formatProgressStage(stage) {
  return String(stage || '').replace(/[_-]+/g, ' ').trim().toUpperCase();
}

function selectVersion(basename) { selectedBasename.value = basename; announce() }

function selectGroup(group) {
  if (!group) return
  const latest = group.latest || group.versions[group.versions.length - 1]
  if (latest?.basename) { selectedBasename.value = latest.basename; announce() }
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
  const base = apiBase.value
  const offloadApiBase = base.endsWith('/api') ? base.slice(0, -4) : base
  if (url.startsWith('/offload')) return offloadApiBase + url
  if (base.endsWith('/api') && url.startsWith('/api/')) return base + url.substring(4)
  return base + url
}

function fileHref(url) { return assetHref(url) }

async function scrollToCurrent() {
  if (!currentGroupId.value) return
  await nextTick()
  const el = document.getElementById('group-' + currentGroupId.value)
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
}

async function load(preferredBasename = '') {
  const res = await fetch(`${apiBase.value}/gallery/list`)
  const data = await res.json()
  const next = (data.items || []).slice().sort((a, b) => String(b?.timestamp || '').localeCompare(String(a?.timestamp || '')))
  items.value = next

  const target = preferredBasename || props.initialBasename
  if (target && next.some((item) => item.basename === target)) {
    selectedBasename.value = target
    scrollToCurrent()
    if (!preferredBasename) announce()
    return
  }
  selectedBasename.value = next[0]?.basename || ''
}

function loadCode() { if (current.value) emit('load-code', { ...current.value }) }

function toggleRename() {
  if (!currentGroup.value) return
  newName.value = currentGroup.value.title
  renaming.value = true
}

async function downloadCurrentStl() {
  if (!current.value?.stl_url) return
  try {
    const res = await fetch(fileHref(current.value.stl_url))
    const blob = await res.blob()
    const blobUrl = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = blobUrl
    link.download = `${current.value.basename || 'model'}.stl`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(blobUrl)
  } catch (e) { alert(`Error: ${e?.message}`) }
}

async function deleteItem() {
  if (!current.value) return
  if (!confirm(`Delete version?`)) return
  try {
    await fetch(`${apiBase.value}/gallery/item/${current.value.basename}`, { method: 'DELETE' })
    await load()
    announce()
  } catch (e) { alert(`Error: ${e.message}`) }
}

async function renameItem() {
  if (!currentGroup.value || !newName.value.trim()) return
  const res = await fetch(`${apiBase.value}/gallery/group/${encodeURIComponent(currentGroup.value.groupId)}/rename`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ newName: newName.value.trim() })
  })
  if (res.ok) { await load(current.value?.basename); renaming.value = false; }
}

async function toggleFeatured(item) {
  if (!item) return
  const newValue = !item.featured
  try {
    await fetch(`${apiBase.value}/gallery/item/${item.basename}/featured`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ featured: newValue })
    })
    item.featured = newValue
  } catch (e) { alert(e.message) }
}

function onKey(e) {
  if (!props.modelValue || renaming.value || !e.altKey) return
  if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
  e.preventDefault()
  const ordered = groupedItems.value
  let idx = ordered.findIndex(g => g.groupId === currentGroupId.value)
  if (idx < 0) return
  if (e.key === 'ArrowUp') idx = Math.max(0, idx - 1)
  if (e.key === 'ArrowDown') idx = Math.min(ordered.length - 1, idx + 1)
  selectGroup(ordered[idx])
}

watch(() => props.modelValue, (value) => { if (value) load(selectedBasename.value) })
onMounted(() => window.addEventListener('keydown', onKey))
onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
})
</script>

<style scoped>
.gallery-dock {
  position: fixed;
  top: 0;
  bottom: 17px; /* the fixed footer is 17px tall */
  left: 44px; /* tools rail width */
  width: 300px;
  z-index: 125;
  display: flex;
  flex-direction: column;
  background: #000;
  color: #eee;
  border-right: 1px solid rgba(255, 255, 255, 0.14);
  font-family: 'IBM Plex Mono', monospace;
  overflow: hidden;
}

.dock-header { display: flex; align-items: center; gap: 10px; height: 40px; flex: 0 0 40px; border-bottom: 1px solid #111; padding: 0 10px 0 15px; }
.gallery-label { font-weight: 800; font-size: 11px; letter-spacing: 0.2em; color: #444; }
.search-input { background: #000; border: 1px solid #111; color: #fff; padding: 4px 8px; font-size: 10px; flex: 1; min-width: 0; outline: none; }
.search-input:focus { border-color: #333; }
.close { background: transparent; color: #444; border: none; font-size: 20px; cursor: pointer; transition: color 0.2s; }
.close:hover { color: #fff; }

.actions { display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 15px; border-bottom: 1px solid #111; }
.actions button { background: #000; border: 1px solid #111; color: #666; font-size: 9px; padding: 2px 8px; letter-spacing: 0.05em; cursor: pointer; transition: all 0.1s; }
.actions button:hover { border-color: #444; color: #fff; }
.actions button.delete:hover { border-color: #600; color: #f44; }

.rename-container { display: flex; gap: 5px; padding: 8px 15px; border-bottom: 1px solid #111; }
.rename-container input { background: #000; border: 1px solid #333; color: #fff; font-size: 10px; padding: 2px 5px; flex: 1; min-width: 0; }
.rename-container button { background: #000; border: 1px solid #111; color: #666; font-size: 9px; padding: 2px 6px; cursor: pointer; }

.list-container { flex: 1; min-height: 80px; overflow-y: auto; scrollbar-width: thin; }
.group-item { padding: 10px 15px; border-bottom: 1px solid #080808; cursor: pointer; }
.group-item:hover { background: #050505; }
.group-item.active { border-left: 2px solid #fff; background: #0a0a0a; }

.group-row { display: flex; align-items: center; gap: 10px; }
.nano-thumb { width: 24px; height: 24px; border: 1px solid #111; object-fit: contain; background: #000; flex-shrink: 0; }
.feat-check { appearance: none; width: 10px; height: 10px; border: 1px solid #333; cursor: pointer; flex-shrink: 0; }
.feat-check:checked { background: #fff; border-color: #fff; }

.title { font-size: 11px; font-weight: 700; color: #999; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; letter-spacing: 0.05em; }
.group-item.active .title { color: #fff; }

.meta-info { display: flex; align-items: center; gap: 5px; }
.badges { display: flex; gap: 3px; }
.b-icon { font-size: 7px; border: 1px solid #111; color: #333; padding: 0 2px; }
.active .b-icon { color: #666; border-color: #222; }

.version-list { margin-top: 5px; display: flex; gap: 5px; flex-wrap: wrap; padding-left: 20px; }
.v-tag { font-size: 8px; color: #444; border: 1px solid #080808; padding: 1px 4px; cursor: pointer; }
.v-tag.active { color: #fff; border-color: #444; }


.metadata { flex: 0 1 200px; border-top: 1px solid #111; overflow-y: auto; padding: 12px 15px; background: #000; }
.meta-section { margin-bottom: 12px; }
.meta-section label { display: block; font-size: 8px; color: #444; letter-spacing: 0.2em; margin-bottom: 5px; }
.raw-text { font-size: 9px; line-height: 1.4; color: #666; white-space: pre-wrap; word-break: break-word; }
.summary-text { font-size: 10px; line-height: 1.4; color: #888; }


.process-footer { flex: 0 0 auto; padding: 6px 15px; background: #050505; border-top: 1px solid #111; }
.terminal-line { font-size: 9px; color: #fff; letter-spacing: 0.05em; font-weight: 700; }
.blink { animation: term-blink 1s infinite; }
@keyframes term-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
.reason-txt { display: block; color: #555; font-weight: 400; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

@media (max-width: 768px) {
  .gallery-dock { left: 0; right: 0; width: auto; border-right: none; }
}
</style>
