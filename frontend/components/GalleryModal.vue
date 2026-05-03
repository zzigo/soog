<template>
  <div v-if="modelValue" class="modal-overlay" @click.self="close">
    <div class="modal">
      <header class="modal-header">
        <div class="header-left">
          <h3>Gallery</h3>
          <div class="search-box">
            <input 
              type="text" 
              v-model="searchQuery" 
              placeholder="Search organogram..." 
              class="search-input"
            />
          </div>
        </div>
        <div class="header-center">
          <div v-if="current && !renaming" class="actions">
            <button @click="loadCode">Load Code</button>
            <button v-if="current.stl_url" @click="downloadCurrentStl">STL</button>
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
          <div class="list" :key="galleryKey" ref="listContainer">
            <div
              v-for="group in groupedItems"
              :key="group.groupId"
              :id="'group-' + group.groupId"
              class="list-item"
              :class="{ active: currentGroupId === group.groupId }"
              @click="selectGroup(group)"
            >
              <div class="group-head">
                <input 
                  type="checkbox" 
                  :checked="group.latest.featured" 
                  @click.stop="toggleFeatured(group.latest)"
                  title="Show in Welcome Modal mosaic"
                  class="featured-checkbox"
                />
                <div class="title">{{ group.title }}</div>
                <div class="item-meta">
                  <div class="status-icons">
                    <svg v-if="group.hasAnySound" class="status-icon sound" viewBox="0 0 24 24" title="Sound available"><path fill="currentColor" d="M14,3.23V5.29C16.89,6.15 19,8.83 19,12C19,15.17 16.89,17.85 14,18.71V20.77C18.03,19.86 21,16.28 21,12C21,7.72 18.03,4.14 14,3.23M16.5,12C16.5,10.23 15.5,8.71 14,7.97V16.04C15.5,15.29 16.5,13.77 16.5,12M3,9V15H7L12,20V4L7,9H3Z"/></svg>
                    <svg v-if="group.hasAnySketch" class="status-icon sketch" viewBox="0 0 24 24" title="Sketch available"><path fill="currentColor" d="M20.71,7.04C21.1,6.65 21.1,6 20.71,5.63L18.37,3.29C18,2.9 17.35,2.9 16.96,3.29L15.12,5.12L18.87,8.87M3,17.25V21H6.75L17.81,9.93L14.06,6.18L3,17.25Z"/></svg>
                    <svg v-if="group.hasAnyPinn" class="status-icon pinn" viewBox="0 0 24 24" title="PINN simulation available"><path fill="currentColor" d="M10,21V19H14V21H10M12,3L2,12H5V20H19V12H22L12,3M12,12.5C10.62,12.5 9.5,11.12 9.5,9.75C9.5,8.38 10.62,7 12,7C13.38,7 14.5,8.38 14.5,9.75C14.5,11.12 13.38,12.5 12,12.5Z"/></svg>
                    <div v-if="group.hasAnyStl" class="stl-badge-noborder">STL</div>
                  </div>
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
          <div v-if="current" class="gallery-results-layout">
            <!-- Sound Bar (Full Width Top) -->
            <div class="sound-library-bar">
              <div class="sound-bar-label">TIMBRES (STABLE AUDIO)</div>
              <div class="sound-bar-content">
                <button 
                  v-for="(sample, idx) in current.sound_samples" 
                  :key="idx"
                  class="bar-play-btn"
                  @click="toggleAudio(sample, idx)"
                  :class="{ playing: playingIdx === idx }"
                  :title="sample.prompt"
                >
                  <svg v-if="playingIdx !== idx" viewBox="0 0 24 24"><path fill="currentColor" d="M8,5.14V19.14L19,12.14L8,5.14Z"/></svg>
                  <svg v-else viewBox="0 0 24 24"><path fill="currentColor" d="M14,19H18V5H14M6,19H10V5H6V19Z"/></svg>
                  <span class="sample-num-bubble">{{ idx + 1 }}</span>
                </button>
                
                <button 
                  class="add-sound-btn" 
                  @click="openSoundGenerator" 
                  :disabled="generatingSound"
                  :class="{ 'is-processing': generatingSound }"
                  title="Add custom sound samples"
                >
                  <svg v-if="!generatingSound" viewBox="0 0 24 24"><path fill="currentColor" d="M19,13H13V19H11V13H5V11H11V5H13V11H19V13Z"/></svg>
                  <svg v-else class="reg-icon is-spinning" viewBox="0 0 24 24"><path fill="currentColor" d="M17.65,6.35C16.2,4.9 14.21,4 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20C15.73,20 18.84,17.45 19.73,14H17.65C16.83,16.33 14.61,18 12,18A6,6 0 0,1 6,12A6,6 0 0,1 12,6C13.66,6 15.14,6.69 16.22,7.78L13,11H20V4L17.65,6.35Z"/></svg>
                </button>
                
                <div v-if="generatingSound" class="active-process-tag">
                  <span class="heartbeat"></span>
                  <span class="process-text">{{ progressStage ? formatProgressStage(progressStage) : 'Synthesizing...' }}</span>
                </div>

                <div v-if="playingIdx !== null" class="sample-time-display">
                  {{ formatAudioTime(currentTime) }} / {{ formatAudioTime(duration) }}
                </div>
              </div>
            </div>

            <div class="quad-grid">
              <!-- Organogram -->
              <section class="viewer-panel">
                <div class="viewer-title-row">
                  <h4 class="viewer-title">Organogram</h4>
                </div>
                <div class="viewer">
                  <img v-if="current?.image_url" :src="imageSrc(current)" class="image" alt="organogram" />
                  <div v-else class="empty">No organogram image</div>
                </div>
              </section>

              <!-- Sketch -->
              <section class="viewer-panel">
                <div v-if="current" class="viewer-title-row">
                  <h4 class="viewer-title">Sketch</h4>
                  <button @click="remakeSketch" class="reg-btn" :disabled="remakingSketch" title="Regenerate sketch image">
                    <svg class="reg-icon" :class="{ 'is-spinning': remakingSketch }" viewBox="0 0 24 24"><path fill="currentColor" d="M17.65,6.35C16.2,4.9 14.21,4 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20C15.73,20 18.84,17.45 19.73,14H17.65C16.83,16.33 14.61,18 12,18A6,6 0 0,1 6,12A6,6 0 0,1 12,6C13.66,6 15.14,6.69 16.22,7.78L13,11H20V4L17.65,6.35Z"/></svg>
                    <span class="reg-label">{{ remakingSketch ? 'DRAWING...' : 'REGENERATE' }}</span>
                  </button>
                </div>
                <div class="viewer">
                  <img v-if="current?.sketch_url" :src="sketchSrc(current)" class="image" :key="current?.sketch_url" alt="sketch render" />
                  <div v-else class="empty">No sketch available</div>
                </div>
              </section>

              <!-- Acoustics (PINN) -->
              <section class="viewer-panel">
                <div v-if="current" class="viewer-title-row">
                  <h4 class="viewer-title">Acoustics (PINN)</h4>
                  <button @click="generateModulus" class="reg-btn" :disabled="remakingModulus" title="Run physical simulation">
                    <svg class="reg-icon" :class="{ 'is-spinning': remakingModulus }" viewBox="0 0 24 24"><path fill="currentColor" d="M17.65,6.35C16.2,4.9 14.21,4 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20C15.73,20 18.84,17.45 19.73,14H17.65C16.83,16.33 14.61,18 12,18A6,6 0 0,1 6,12A6,6 0 0,1 12,6C13.66,6 15.14,6.69 16.22,7.78L13,11H20V4L17.65,6.35Z"/></svg>
                    <span class="reg-label">{{ remakingModulus ? 'CALCULATING...' : 'CALCULATE' }}</span>
                  </button>
                </div>
                <div class="viewer acoustics-viewer">
                  <ModulusHeatmap 
                    v-if="current?.modulus?.results?.pressure_map" 
                    :data="current.modulus.results.pressure_map" 
                    :size="300"
                  />
                  <div v-else class="empty">No simulation data</div>
                </div>
              </section>

              <!-- 3D Model -->
              <section class="viewer-panel">
                <div v-if="current" class="viewer-title-row">
                  <h4 class="viewer-title">3D Model</h4>
                  <button @click="generateLRM" class="reg-btn" :disabled="remakingLRM" title="Generate High-Fidelity Mesh (InstantMesh)">
                    <svg class="reg-icon" :class="{ 'is-spinning': remakingLRM }" viewBox="0 0 24 24"><path fill="currentColor" d="M17.65,6.35C16.2,4.9 14.21,4 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20C15.73,20 18.84,17.45 19.73,14H17.65C16.83,16.33 14.61,18 12,18A6,6 0 0,1 6,12A6,6 0 0,1 12,6C13.66,6 15.14,6.69 16.22,7.78L13,11H20V4L17.65,6.35Z"/></svg>
                    <span class="reg-label">{{ remakingLRM ? 'RECONSTRUCTING...' : 'RECONSTRUCT' }}</span>
                  </button>
                </div>
                <div class="viewer">
                  <ClientOnly>
                    <StlViewer v-if="current?.stl_url" :url="fileHref(current.stl_url)" />
                    <div v-else class="empty">No STL available</div>
                  </ClientOnly>
                </div>
              </section>
            </div>
          </div>
          <div v-else class="empty full-empty">No item selected</div>
        </div>
      </section>

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
      
      <footer v-if="isProcessingAny" class="modal-footer crt-monitor">
        <div class="crt-overlay"></div>
        <div class="progress-info">
          <div class="status-row">
            <span class="heartbeat amber"></span>
            <span class="status-label amber-text">{{ consoleDisplayMessage }}</span>
          </div>
          <div v-if="reasoningPreview" class="reasoning-preview amber-text">{{ reasoningPreview }}</div>
        </div>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useRuntimeConfig } from '#app'
import { marked } from 'marked'
import StlViewer from '~/components/StlViewer.vue'
import ModulusHeatmap from '~/components/ModulusHeatmap.vue'

const props = defineProps({
  modelValue: { type: Boolean, default: false },
  initialBasename: { type: String, default: '' }
})
const emit = defineEmits(['update:modelValue', 'load-code'])

const items = ref([])
const selectedBasename = ref('')
const searchQuery = ref('')
const renaming = ref(false)
const newName = ref('')
const galleryKey = ref(0)

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
  if (generatingSound.value) return 'TIMBRAL SYNTHESIS...'
  if (remakingSketch.value) return 'INFERRING SKETCH...'
  if (remakingModulus.value) return 'ACOUSTIC SIMULATION...'
  if (remakingLRM.value) return 'LRM 3D RECONSTRUCTION...'
  return 'PROCESSING...'
})

async function generateLRM() {
  if (!current.value || remakingLRM.value) return
  remakingLRM.value = true
  generationRequestId.value = createRequestId()
  startReasoningPolling()
  try {
    const res = await fetch(`${apiBase}/gallery/item/${current.value.basename}/generate_lrm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_id: generationRequestId.value })
    })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.error || 'Failed to generate high-res mesh')
    }
    await load(current.value?.basename)
  } catch (e) { 
    alert(e.message) 
  } finally { 
    remakingLRM.value = false; 
    stopReasoningPolling(); 
  }
}
const listContainer = ref(null)

// Sound generation modal state
const showSoundGenModal = ref(false)
const customSoundPrompt = ref('')

function openSoundGenerator() {
  customSoundPrompt.value = current.value?.prompt || ''
  showSoundGenModal.value = true
}

async function executeSoundGen() {
  if (!current.value || generatingSound.value) return
  generatingSound.value = true
  showSoundGenModal.value = false
  generationRequestId.value = createRequestId()
  startReasoningPolling()
  try {
    const res = await fetch(`${apiBase}/gallery/item/${current.value.basename}/generate_sound`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        request_id: generationRequestId.value,
        prompt: customSoundPrompt.value 
      })
    })
    const payload = await res.json()
    if (!res.ok) throw new Error(payload.error || 'Failed to generate sound')
    await load(current.value?.basename)
  } catch (e) { 
    alert(e.message) 
  } finally { 
    generatingSound.value = false; 
    stopReasoningPolling(); 
  }
}

// Audio state
const playingIdx = ref(null)
const currentTime = ref(0)
const duration = ref(0)
let audioObj = null

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
  
  if (audioObj) {
    audioObj.pause()
  }
  
  playingIdx.value = idx
  const url = assetHref(sample.ogg_url || sample.url)
  audioObj = new Audio(url)
  audioObj.addEventListener('timeupdate', () => {
    currentTime.value = audioObj.currentTime
  })
  audioObj.addEventListener('loadedmetadata', () => {
    duration.value = audioObj.duration
  })
  audioObj.addEventListener('ended', () => {
    playingIdx.value = null
  })
  audioObj.play()
}

let progressPollInterval = null

function createRequestId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `soog-gallery-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function stopReasoningPolling() {
  if (progressPollInterval) {
    clearInterval(progressPollInterval);
    progressPollInterval = null;
  }
}

async function fetchReasoningProgress() {
  if (!generationRequestId.value) return;
  try {
    const response = await fetch(`${apiBase}/generate/progress/${generationRequestId.value}`, {
      headers: { Accept: 'application/json' }
    });
    if (!response.ok) return;
    const payload = await response.json();
    if (!payload?.ok) return;
    reasoningPreview.value = String(payload.reasoning_preview || '').trim();
    progressStage.value = String(payload.stage || '').trim();
    if (payload.status === 'completed' || payload.status === 'error') {
      stopReasoningPolling();
    }
  } catch (e) {
    reasoningPreview.value = 'Lost connection to progress tracker...';
  }
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
    if (String(item?.timestamp || '') > String(group.latestTimestamp || '')) {
      group.latestTimestamp = item.timestamp
    }
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
  return String(stage || '')
    .replace(/[_-]+/g, ' ')
    .trim()
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

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

function fileHref(url) { return assetHref(url) }

async function scrollToCurrent() {
  if (!currentGroupId.value) return
  await nextTick()
  const el = document.getElementById('group-' + currentGroupId.value)
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
}

async function load(preferredBasename = '') {
  const res = await fetch(`${apiBase}/gallery/list`)
  const data = await res.json()
  const next = (data.items || []).slice().sort((a, b) => String(b?.timestamp || '').localeCompare(String(a?.timestamp || '')))
  items.value = next

  const target = preferredBasename || props.initialBasename
  if (target && next.some((item) => item.basename === target)) {
    selectedBasename.value = target
    scrollToCurrent()
    return
  }
  selectedBasename.value = next[0]?.basename || ''
}

function formatTime(ts) { return ts || '' }

function loadCode() {
  if (current.value) emit('load-code', { ...current.value })
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
    alert(`Error: ${e?.message}`)
  }
}

async function deleteItem() {
  if (!current.value) return
  if (!confirm(`Delete version?`)) return
  try {
    await fetch(`${apiBase}/gallery/item/${current.value.basename}`, { method: 'DELETE' })
    await load()
  } catch (e) {
    alert(`Error: ${e.message}`)
  }
}

async function renameItem() {
  if (!currentGroup.value || !newName.value.trim()) return
  const res = await fetch(`${apiBase}/gallery/group/${encodeURIComponent(currentGroup.value.groupId)}/rename`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ newName: newName.value.trim() })
  })
  if (res.ok) {
    await load(current.value?.basename)
    renaming.value = false
  }
}

async function remakeSketch() {
  if (!current.value || remakingSketch.value) return
  remakingSketch.value = true
  generationRequestId.value = createRequestId()
  startReasoningPolling()
  try {
    const res = await fetch(`${apiBase}/gallery/item/${current.value.basename}/remake_sketch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_id: generationRequestId.value })
    })
    if (!res.ok) throw new Error('Failed to remake sketch')
    await load(current.value?.basename)
  } catch (e) { alert(e.message) }
  finally { remakingSketch.value = false; stopReasoningPolling(); }
}

async function generateSound() {
  if (!current.value || generatingSound.value) return
  generatingSound.value = true
  generationRequestId.value = createRequestId()
  startReasoningPolling()
  try {
    const res = await fetch(`${apiBase}/gallery/item/${current.value.basename}/generate_sound`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_id: generationRequestId.value })
    })
    if (!res.ok) throw new Error('Failed to generate sound')
    await load(current.value?.basename)
  } catch (e) { alert(e.message) }
  finally { generatingSound.value = false; stopReasoningPolling(); }
}

async function generateModulus() {
  if (!current.value || remakingModulus.value) return
  remakingModulus.value = true
  generationRequestId.value = createRequestId()
  startReasoningPolling()
  try {
    const prompt = `[MODULUS] ${current.value.prompt}`
    const res = await fetch(`${apiBase}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        prompt, 
        request_id: generationRequestId.value, 
        gallery_update: current.value.basename 
      })
    })
    if (!res.ok) throw new Error('Failed to run simulation')
    await load(current.value?.basename)
  } catch (e) { alert(e.message) }
  finally { remakingModulus.value = false; stopReasoningPolling(); }
}

async function toggleFeatured(item) {
  if (!item) return
  const newValue = !item.featured
  try {
    await fetch(`${apiBase}/gallery/item/${item.basename}/featured`, {
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
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center; z-index: 3000; backdrop-filter: blur(4px); }
.modal { width: 95vw; height: 90vh; background: #080808; color: #fff; border: 1px solid #222; border-radius: 4px; display: flex; flex-direction: column; overflow: hidden; }
.modal-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; border-bottom: 1px solid #222; background: #000; }
.header-left { flex: 1; display: flex; align-items: center; gap: 20px; }
.header-right { flex: 0 0 50px; text-align: right; }
.header-center { flex: 2; display: flex; justify-content: center; gap: 10px; }

.search-input { background: #111; border: 1px solid #333; border-radius: 4px; color: #fff; padding: 6px 12px; font-size: 12px; width: 250px; }
.close { background: transparent; color: #fff; border: none; font-size: 24px; cursor: pointer; opacity: 0.5; transition: opacity 0.2s; }
.close:hover { opacity: 1; }

.modal-body { display: flex; flex: 1; overflow: hidden; }
.left { width: 380px; border-right: 1px solid #222; display: flex; flex-direction: column; background: #0a0a0a; }
.list { flex: 1; overflow-y: auto; }
.list-item { padding: 12px 15px; cursor: pointer; border-bottom: 1px solid #151515; transition: background 0.2s; }
.list-item:hover { background: #111; }
.list-item.active { background: #1a1a1a; border-left: 3px solid #ff9a2f; }

.group-head { display: flex; align-items: center; gap: 12px; }
.title { font-weight: 600; font-size: 13px; color: #eee; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.item-meta { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
.status-icons { display: flex; align-items: center; gap: 6px; }
.status-icon { width: 14px; height: 14px; color: #ff9a2f; opacity: 0.8; }
.stl-badge-noborder { font-size: 9px; font-weight: 800; color: #ff9a2f; padding: 0 3px; border-radius: 2px; letter-spacing: 0.05em; }
.time { font-size: 10px; color: #555; font-family: monospace; }

.versions { margin-top: 8px; display: flex; gap: 4px; flex-wrap: wrap; }
.version-chip { background: #111; border: 1px solid #333; color: #888; font-size: 10px; padding: 2px 6px; border-radius: 2px; cursor: pointer; }
.version-chip.active { background: #ff9a2f; border-color: #ff9a2f; color: #000; font-weight: 700; }

.details { flex: 0 0 300px; border-top: 1px solid #222; overflow-y: auto; padding: 15px; background: #000; }
.details h4 { font-size: 10px; text-transform: uppercase; color: #555; letter-spacing: 0.1em; margin: 15px 0 8px 0; }
.text { font-size: 12px; line-height: 1.4; color: #aaa; background: #0a0a0a; padding: 10px; border-radius: 4px; border: 1px solid #151515; }

/* Sound Bar Layout */
.sound-library-bar { 
  background: #050505; 
  border-bottom: 1px solid #111; 
  padding: 12px 20px; 
  display: flex; 
  align-items: center; 
  gap: 20px;
  min-height: 60px;
}
.sound-bar-label { font-size: 9px; font-weight: 800; color: #333; letter-spacing: 0.2em; white-space: nowrap; }
.sound-bar-content { display: flex; align-items: center; gap: 12px; flex: 1; }

.bar-play-btn { 
  background: #111; 
  border: 1px solid #222; 
  color: #ff9a2f; 
  width: 42px; 
  height: 42px; 
  border-radius: 50%; 
  display: flex; 
  align-items: center; 
  justify-content: center; 
  cursor: pointer; 
  position: relative; 
  transition: all 0.2s; 
}
.bar-play-btn:hover { background: #222; border-color: #ff9a2f; transform: scale(1.05); }
.bar-play-btn.playing { background: #ff9a2f; color: #000; border-color: #ff9a2f; box-shadow: 0 0 15px rgba(255,154,47,0.3); }
.bar-play-btn svg { width: 24px; height: 24px; }

.add-sound-btn {
  background: transparent;
  border: 2px dashed #222;
  color: #444;
  width: 42px;
  height: 42px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
}
.add-sound-btn:hover { border-color: #ff9a2f; color: #ff9a2f; }
.add-sound-btn svg { width: 20px; height: 20px; }
.add-sound-btn.is-processing { border-style: solid; border-color: #ff9a2f; color: #ff9a2f; }

.active-process-tag {
  display: flex;
  align-items: center;
  gap: 8px;
  background: rgba(255, 154, 47, 0.1);
  padding: 4px 12px;
  border-radius: 4px;
  border: 1px solid rgba(255, 154, 47, 0.2);
}
.process-text {
  font-size: 10px;
  font-weight: 700;
  color: #ff9a2f;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.heartbeat {
  width: 6px;
  height: 6px;
  background-color: #ff9a2f;
  border-radius: 50%;
  animation: heartbeat-anim 1.2s infinite;
}
@keyframes heartbeat-anim {
  0% { transform: scale(0.8); opacity: 0.5; }
  50% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 8px #ff9a2f; }
  100% { transform: scale(0.8); opacity: 0.5; }
}

.sample-num-bubble { position: absolute; bottom: -2px; right: -2px; background: #000; color: #fff; font-size: 8px; width: 16px; height: 16px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 1px solid #333; font-weight: 700; }
.sample-time-display { font-family: monospace; font-size: 11px; color: #ff9a2f; margin-left: auto; letter-spacing: 0.05em; background: rgba(255,154,47,0.1); padding: 4px 10px; border-radius: 12px; }

.right { flex: 1; background: #000; overflow: hidden; display: flex; flex-direction: column; }
.gallery-results-layout { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.quad-grid { flex: 1; display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 1px; background: #111; }
.viewer-panel { background: #000; display: flex; flex-direction: column; overflow: hidden; }
.viewer-title-row { display: flex; justify-content: space-between; align-items: center; padding: 0 15px; height: 36px; border-bottom: 1px solid #0a0a0a; }
.viewer-title { margin: 0; font-size: 10px; letter-spacing: 0.15em; text-transform: uppercase; color: #444; }

/* Sub-modal for sound generation */
.sub-modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.85); display: flex; align-items: center; justify-content: center; z-index: 4000; backdrop-filter: blur(8px); }
.sub-modal { width: 500px; background: #0a0a0a; border: 1px solid #333; border-radius: 8px; overflow: hidden; box-shadow: 0 20px 50px rgba(0,0,0,0.8); }
.sub-modal-header { padding: 15px 20px; background: #000; border-bottom: 1px solid #222; display: flex; justify-content: space-between; align-items: center; }
.sub-modal-header h4 { margin: 0; font-size: 11px; letter-spacing: 0.2em; color: #ff9a2f; }
.sub-modal-header button { background: transparent; border: none; color: #555; font-size: 20px; cursor: pointer; }
.sub-modal-body { padding: 20px; }
.sub-modal-desc { font-size: 12px; color: #888; margin-bottom: 15px; line-height: 1.5; }
.sub-modal-textarea { width: 100%; height: 120px; background: #000; border: 1px solid #333; border-radius: 4px; color: #eee; padding: 12px; font-family: inherit; font-size: 13px; resize: none; outline: none; margin-bottom: 20px; }
.sub-modal-textarea:focus { border-color: #ff9a2f; }
.sub-modal-actions { display: flex; justify-content: flex-end; gap: 12px; }
.cancel-btn { background: transparent; border: 1px solid #333; color: #888; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 11px; }
.confirm-btn { background: #ff9a2f; border: 1px solid #ff9a2f; color: #000; padding: 8px 20px; border-radius: 4px; cursor: pointer; font-size: 11px; font-weight: 700; }
.confirm-btn:disabled { opacity: 0.5; }

.viewer { flex: 1; display: flex; align-items: center; justify-content: center; min-height: 0; position: relative; }
.image { max-width: 95%; max-height: 95%; object-fit: contain; }
.acoustics-viewer { background: #020202; padding: 20px; }

.reg-btn { background: transparent; border: none; color: #333; display: flex; align-items: center; gap: 5px; cursor: pointer; transition: all 0.2s; }
.reg-btn:hover:not(:disabled) { color: #ff9a2f; }
.reg-btn:disabled { opacity: 0.3; cursor: not-allowed; }
.reg-icon { width: 14px; height: 14px; }
.reg-icon.is-spinning { animation: spin 2s linear infinite; }
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

.reg-label { font-size: 9px; font-weight: 700; text-transform: uppercase; display: none; }
.reg-btn:hover .reg-label { display: inline; }

.modal-footer { position: absolute; bottom: 0; left: 0; right: 0; background: #000; padding: 10px 20px; border-top: 1px solid #ff9a2f; z-index: 5000; height: 50px; display: flex; align-items: center; overflow: hidden; }
.crt-monitor {
  background: #2a1a00; /* Lighter amber/brown base for better visibility */
  border-top: 2px solid #ffb000;
  box-shadow: inset 0 0 30px rgba(255, 176, 0, 0.15);
}
.crt-overlay {
  position: absolute;
  top: 0; left: 0; bottom: 0; right: 0;
  background: linear-gradient(rgba(18, 16, 1, 0) 50%, rgba(0, 0, 0, 0.15) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.04), rgba(0, 255, 0, 0.01), rgba(0, 255, 0, 0.04));
  z-index: 2;
  background-size: 100% 2px, 3px 100%;
  pointer-events: none;
  opacity: 0.8;
}
.amber-text {
  color: #ffb000;
  text-shadow: 0 0 8px rgba(255, 176, 0, 0.8);
  font-family: 'Courier New', Courier, monospace;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.status-label.amber-text { font-size: 12px; font-weight: 900; } /* Increased from 10px (~20% but user asked 10%, 11-12px is safer) */
.reasoning-preview.amber-text { font-size: 10.5px; opacity: 0.9; max-width: 90%; line-height: 1.2; } /* Increased from 9px */

.heartbeat.amber { background: #ffb000; box-shadow: 0 0 10px #ffb000; }
@keyframes pulse-glow { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.8); } 100% { opacity: 1; transform: scale(1); } }

.actions button { background: #111; border: 1px solid #222; color: #eee; font-size: 11px; padding: 5px 12px; border-radius: 2px; cursor: pointer; }
.actions button:hover { border-color: #ff9a2f; color: #ff9a2f; }
.actions button.delete:hover { background: #400; border-color: #900; color: #fff; }

@media (max-width: 1200px) {
  .left { width: 300px; }
  .quad-grid { grid-template-columns: 1fr; grid-template-rows: repeat(4, 300px); overflow-y: auto; }
}
</style>
