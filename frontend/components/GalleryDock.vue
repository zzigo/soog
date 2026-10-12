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


    <div v-if="current" class="dock-tools">
      <div class="sound-controls">
        <button
          v-for="(sample, idx) in current.sound_samples"
          :key="idx"
          class="sound-node"
          @click="toggleAudio(sample, idx)"
          :class="{ playing: playingIdx === idx }"
        >
          <svg v-if="playingIdx !== idx" viewBox="0 0 24 24"><path fill="currentColor" d="M8,5.14V19.14L19,12.14L8,5.14Z"/></svg><svg v-else viewBox="0 0 24 24"><path fill="currentColor" d="M14,19H18V5H14M6,19H10V5H6V19Z"/></svg><span class="s-idx">{{ idx + 1 }}</span>
        </button>
        <button class="add-node" title="Generate sound samples" @click="openSoundGenerator" :disabled="generatingSound">+</button>
        <div v-if="playingIdx !== null" class="timer">
          {{ formatAudioTime(currentTime) }} / {{ formatAudioTime(duration) }}
        </div>
      </div>
      <div class="tool-row">
        <button class="pane-action" @click="remakeSketch" :disabled="remakingSketch">REGEN SKETCH</button>
        <button class="pane-action" @click="generateModulus" :disabled="remakingModulus">CALC ACOUSTICS</button>
        <button class="pane-action" @click="generateLRM" :disabled="remakingLRM">BUILD 3D</button>
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

    <!-- Sound Generation Sub-Modal -->
    <div v-if="showSoundGenModal" class="sub-modal-overlay" @click.self="showSoundGenModal = false">
      <div class="sub-modal">
        <header class="sub-modal-header">
          <h4>CUSTOM SOUND SYNTHESIS</h4>
          <button @click="showSoundGenModal = false">×</button>
        </header>
        <div class="sub-modal-body">
          <p class="sub-modal-desc">Generate new timbral samples using Stable Audio Open. Describe the sonic characteristics you want to synthesize.</p>
          <textarea 
            v-model="customSoundPrompt" 
            placeholder="e.g. resonant wooden pipes, metallic harmonic overtones..." 
            class="sub-modal-textarea"
          ></textarea>
          <div class="sub-modal-actions">
            <button @click="showSoundGenModal = false" class="cancel-btn">Cancel</button>
            <button @click="executeSoundGen" class="confirm-btn" :disabled="generatingSound">
              {{ generatingSound ? 'GENERATING...' : 'GENERATE SAMPLES' }}
            </button>
          </div>
        </div>
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

// Processing States
const remakingSketch = ref(false)
const generatingSound = ref(false)
const remakingModulus = ref(false)
const remakingLRM = ref(false)
const progressStage = ref('')
const reasoningPreview = ref('')
const generationRequestId = ref('')

const isProcessingAny = computed(() => 
  generatingSound.value || remakingSketch.value || remakingModulus.value || remakingLRM.value
)

const consoleDisplayMessage = computed(() => {
  if (progressStage.value) return formatProgressStage(progressStage.value)
  if (generatingSound.value) return 'SYNTHESIZING AUDIO...'
  if (remakingSketch.value) return 'INFERRING SKETCH...'
  if (remakingModulus.value) return 'SOLVING ACOUSTICS...'
  if (remakingLRM.value) return 'RECONSTRUCTING 3D...'
  return 'PROCESSING...'
})

// Audio state
const playingIdx = ref(null)
const currentTime = ref(0)
const duration = ref(0)
let audioObj = null

// Sub-modal state
const showSoundGenModal = ref(false)
const customSoundPrompt = ref('')

function openSoundGenerator() {
  customSoundPrompt.value = current.value?.prompt || ''
  showSoundGenModal.value = true
}

async function readJsonSafe(res) {
  const text = await res.text()
  try { return JSON.parse(text) } catch { return { error: `Server returned ${res.status} ${res.statusText || ''} (not JSON). The request may have timed out at the proxy.`.trim() } }
}

async function waitForSoundJob(requestId, timeoutMs = 40 * 60 * 1000) {
  const started = Date.now()
  while (Date.now() - started < timeoutMs) {
    await new Promise((resolve) => setTimeout(resolve, 3000))
    try {
      const res = await fetch(`${apiBase.value}/generate/progress/${requestId}`, { headers: { Accept: 'application/json' } })
      if (!res.ok) continue
      const payload = await readJsonSafe(res)
      if (payload?.status === 'completed') return
      if (payload?.status === 'error') throw new Error(payload.reasoning_preview || 'Sound generation failed')
    } catch (e) {
      if (e instanceof Error && e.message && !/fetch|network|Failed/i.test(e.message)) throw e
    }
  }
  throw new Error('Sound generation timed out')
}

async function executeSoundGen() {
  if (!current.value || generatingSound.value) return
  generatingSound.value = true
  showSoundGenModal.value = false
  generationRequestId.value = createRequestId()
  startReasoningPolling()
  try {
    const res = await fetch(`${apiBase.value}/gallery/item/${current.value.basename}/generate_sound`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(await authHeaders()) },
      body: JSON.stringify({
        request_id: generationRequestId.value,
        prompt: customSoundPrompt.value
      })
    })
    const payload = await readJsonSafe(res)
    if (!res.ok) throw new Error(payload.error || 'Failed to generate sound')
    if (res.status === 202) await waitForSoundJob(generationRequestId.value)
    await load(current.value?.basename)
    announce()
  } catch (e) { alert(e.message) }
  finally { generatingSound.value = false; stopReasoningPolling(); }
}

async function generateLRM() {
  if (!current.value || remakingLRM.value) return
  remakingLRM.value = true
  generationRequestId.value = createRequestId()
  startReasoningPolling()
  try {
    const res = await fetch(`${apiBase.value}/gallery/item/${current.value.basename}/generate_lrm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(await authHeaders()) },
      body: JSON.stringify({ request_id: generationRequestId.value })
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.error || 'Failed to generate high-res mesh')
    }
    await load(current.value?.basename)
    announce()
  } catch (e) { alert(e.message) }
  finally { remakingLRM.value = false; stopReasoningPolling(); }
}

async function remakeSketch() {
  if (!current.value || remakingSketch.value) return
  remakingSketch.value = true
  generationRequestId.value = createRequestId()
  startReasoningPolling()
  try {
    const res = await fetch(`${apiBase.value}/gallery/item/${current.value.basename}/remake_sketch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(await authHeaders()) },
      body: JSON.stringify({ request_id: generationRequestId.value })
    })
    if (!res.ok) throw new Error('Failed to remake sketch')
    await load(current.value?.basename)
    announce()
  } catch (e) { alert(e.message) }
  finally { remakingSketch.value = false; stopReasoningPolling(); }
}

async function generateModulus() {
  if (!current.value || remakingModulus.value) return
  remakingModulus.value = true
  generationRequestId.value = createRequestId()
  startReasoningPolling()
  try {
    const prompt = `[MODULUS] ${current.value.prompt}`
    const res = await fetch(`${apiBase.value}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(await authHeaders()) },
      body: JSON.stringify({ 
        prompt, 
        request_id: generationRequestId.value, 
        gallery_update: current.value.basename 
      })
    })
    if (!res.ok) throw new Error('Failed to run simulation')
    await load(current.value?.basename)
    announce()
  } catch (e) { alert(e.message) }
  finally { remakingModulus.value = false; stopReasoningPolling(); }
}

function formatAudioTime(seconds) {
  if (isNaN(seconds)) return '0:00'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function toggleAudio(sample, idx) {
  if (playingIdx.value === idx) {
    audioObj?.pause()
    playingIdx.value = null
    return
  }
  if (audioObj) audioObj.pause()
  playingIdx.value = idx
  const url = assetHref(sample.ogg_url || sample.url)
  audioObj = new Audio(url)
  audioObj.addEventListener('timeupdate', () => { currentTime.value = audioObj.currentTime })
  audioObj.addEventListener('loadedmetadata', () => { duration.value = audioObj.duration })
  audioObj.addEventListener('ended', () => { playingIdx.value = null })
  audioObj.play()
}

let progressPollInterval = null

function createRequestId() {
  return `gallery-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function stopReasoningPolling() {
  if (progressPollInterval) { clearInterval(progressPollInterval); progressPollInterval = null; }
}

async function fetchReasoningProgress() {
  if (!generationRequestId.value) return;
  try {
    const response = await fetch(`${apiBase.value}/generate/progress/${generationRequestId.value}`, {
      headers: { Accept: 'application/json' }
    });
    if (!response.ok) return;
    const payload = await response.json();
    if (!payload?.ok) return;
    reasoningPreview.value = String(payload.reasoning_preview || '').trim();
    progressStage.value = String(payload.stage || '').trim();
    if (payload.status === 'completed' || payload.status === 'error') stopReasoningPolling();
  } catch (e) { reasoningPreview.value = 'CONNECTION LOST...'; }
}

function startReasoningPolling() {
  stopReasoningPolling();
  reasoningPreview.value = '';
  progressStage.value = '';
  progressPollInterval = setInterval(fetchReasoningProgress, 800);
}

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
  if (audioObj) audioObj.pause()
  stopReasoningPolling()
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

.dock-tools { flex: 0 0 auto; border-top: 1px solid #111; padding: 10px 15px; display: flex; flex-direction: column; gap: 8px; background: #020202; }
.sound-controls { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
.sound-node { position: relative; background: #000; border: 1px solid #111; color: #ff9a2f; width: 32px; height: 32px; font-size: 9px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
.sound-node svg { width: 24px; height: 24px; }
.s-idx { position: absolute; bottom: -2px; right: -2px; font-size: 7px; background: #000; color: #fff; width: 12px; height: 12px; border-radius: 50%; display: flex; align-items: center; justify-content: center; }
.sound-node.playing { background: rgba(255, 255, 255, 0.6); color: #000; border-color: #fff; box-shadow: 0 0 10px rgba(255,255,255,0.2); }
.add-node { background: transparent; border: 1px dashed #333; color: #555; width: 22px; height: 22px; cursor: pointer; }
.timer { font-size: 9px; color: #555; letter-spacing: 0.1em; }
.tool-row { display: flex; flex-wrap: wrap; gap: 6px; }
.pane-action { background: transparent; border: 1px solid #1a1a1a; color: #555; font-size: 8px; padding: 3px 6px; letter-spacing: 0.05em; cursor: pointer; }
.pane-action:hover:not(:disabled) { color: #fff; border-color: #333; }
.pane-action:disabled { opacity: 0.4; cursor: default; }

.metadata { flex: 0 1 200px; border-top: 1px solid #111; overflow-y: auto; padding: 12px 15px; background: #000; }
.meta-section { margin-bottom: 12px; }
.meta-section label { display: block; font-size: 8px; color: #444; letter-spacing: 0.2em; margin-bottom: 5px; }
.raw-text { font-size: 9px; line-height: 1.4; color: #666; white-space: pre-wrap; word-break: break-word; }
.summary-text { font-size: 10px; line-height: 1.4; color: #888; }

/* Sound generation dialog (small, stays modal) */
.sub-modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.9); display: flex; align-items: center; justify-content: center; z-index: 4000; }
.sub-modal { width: min(400px, 92vw); background: #000; border: 1px solid #111; padding: 20px; }
.sub-modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
.sub-modal-header h4 { font-size: 10px; letter-spacing: 0.2em; color: #fff; margin: 0; }
.sub-modal-header button { background: transparent; border: none; color: #444; font-size: 18px; cursor: pointer; }
.sub-modal-desc { font-size: 9px; color: #444; margin-bottom: 10px; }
.sub-modal-textarea { width: 100%; height: 80px; background: #000; border: 1px solid #111; color: #eee; padding: 8px; font-size: 11px; resize: none; outline: none; margin-bottom: 15px; }
.sub-modal-actions { display: flex; justify-content: flex-end; gap: 10px; }
.cancel-btn { background: transparent; border: 1px solid #111; color: #444; font-size: 9px; padding: 4px 10px; cursor: pointer; }
.confirm-btn { background: #fff; color: #000; border: 1px solid #fff; font-size: 9px; padding: 4px 15px; font-weight: 700; cursor: pointer; }

.process-footer { flex: 0 0 auto; padding: 6px 15px; background: #050505; border-top: 1px solid #111; }
.terminal-line { font-size: 9px; color: #fff; letter-spacing: 0.05em; font-weight: 700; }
.blink { animation: term-blink 1s infinite; }
@keyframes term-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
.reason-txt { display: block; color: #555; font-weight: 400; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

@media (max-width: 768px) {
  .gallery-dock { left: 0; right: 0; width: auto; border-right: none; }
}
</style>
