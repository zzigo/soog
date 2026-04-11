<template>
  <div class="app-container">
    <!-- Left column: resizable editor -->
    <div class="left-column" :style="{ width: leftWidth + '%' }">
      <div class="editor-wrapper">
        <AceEditor 
          ref="editorRef" 
          @evaluate="handleEvaluate"
        />
      </div>
    </div>
    <!-- Draggable divider -->
    <div 
      class="divider" 
      @mousedown="startDrag" 
      @touchstart.prevent="startDragTouch"
    ></div>
    <!-- Right column hosts HUD and Results -->
    <div class="right-column" :style="{ width: (100 - leftWidth) + '%' }">
      <div class="hud">
      <span class="model-name" :title="currentModel || ''">{{ shortModelLabel }}</span>
      <button
        @click="cycleOllamaModel"
        class="icon-button model-cycle-button"
        :disabled="modelSwitching || ollamaModels.length < 2"
        :title="`Model: ${currentModel || 'loading...'} (click to cycle)`"
      >
        <svg class="icon" viewBox="0 0 24 24">
          <path fill="currentColor" d="M12,4V1L8,5L12,9V6A6,6 0 0,1 18,12C18,13 17.75,13.96 17.3,14.8L18.76,16.26C19.53,15.05 20,13.57 20,12A8,8 0 0,0 12,4M6.7,9.2L5.24,7.74C4.47,8.95 4,10.43 4,12A8,8 0 0,0 12,20V23L16,19L12,15V18A6,6 0 0,1 6,12C6,11 6.25,10.04 6.7,9.2Z"/>
        </svg>
      </button>

      <button @click="toggleShowCode" class="icon-button" :title="showCode ? 'Hide Code' : 'Show Code'">
        <svg v-if="showCode" class="icon" viewBox="0 0 24 24">
          <path fill="currentColor" d="M12,9A3,3 0 0,1 15,12A3,3 0 0,1 12,15A3,3 0 0,1 9,12A3,3 0 0,1 12,9M12,4.5C17,4.5 21.27,7.61 23,12C21.27,16.39 17,19.5 12,19.5C7,19.5 2.73,16.39 1,12C2.73,7.61 7,4.5 12,4.5M3.18,12C4.83,15.36 8.24,17.5 12,17.5C15.76,17.5 19.17,15.36 20.82,12C19.17,8.64 15.76,6.5 12,6.5C8.24,6.5 4.83,8.64 3.18,12Z" />
        </svg>
        <svg v-else class="icon" viewBox="0 0 24 24">
          <path fill="currentColor" d="M11.83,9L15,12.16C15,12.11 15,12.05 15,12A3,3 0 0,0 12,9C11.94,9 11.89,9 11.83,9M7.53,9.8L9.08,11.35C9.03,11.56 9,11.77 9,12A3,3 0 0,0 12,15C12.22,15 12.44,14.97 12.65,14.92L14.2,16.47C13.53,16.8 12.79,17 12,17A5,5 0 0,1 7,12C7,11.21 7.2,10.47 7.53,9.8M2,4.27L4.28,6.55L4.73,7C3.08,8.3 1.78,10 1,12C2.73,16.39 7,19.5 12,19.5C13.55,19.5 15.03,19.2 16.38,18.66L16.81,19.08L19.73,22L21,20.73L3.27,3M12,7A5,5 0 0,1 17,12C17,12.64 16.87,13.26 16.64,13.82L19.57,16.75C21.07,15.5 22.27,13.86 23,12C21.27,7.61 17,4.5 12,4.5C10.6,4.5 9.26,4.75 8,5.2L10.17,7.35C10.74,7.13 11.35,7 12,7Z" />
        </svg>
      </button>
      <button @click="handleMobileEvaluate" class="icon-button" title="Evaluate selected text or all if no selection">
        <svg class="icon" viewBox="0 0 24 24">
          <path fill="currentColor" d="M8,5.14V19.14L19,12.14L8,5.14Z" />
        </svg>
      </button>
      <button @click="handleClear" class="icon-button" title="Clear Editor (Ctrl+H)">
        <svg class="icon" viewBox="0 0 24 24">
          <path fill="currentColor" d="M9,3V4H4V6H5V19A2,2 0 0,0 7,21H17A2,2 0 0,0 19,19V6H20V4H15V3H9M7,6H17V19H7V6M9,8V17H11V8H9M13,8V17H15V8H13Z" />
        </svg>
      </button>
      <button @click="handleRandomPrompt" class="icon-button" title="Random Prompt">
        <svg class="icon" viewBox="0 0 24 24">
          <path fill="currentColor" d="M14.83,13.41L13.42,14.82L16.55,17.95L14.5,20H20V14.5L17.96,16.54L14.83,13.41M14.5,4L16.54,6.04L4,18.59L5.41,20L17.96,7.46L20,9.5V4M10.59,9.17L5.41,4L4,5.41L9.17,10.58L10.59,9.17Z" />
        </svg>
      </button>
      <button @click="showHelp = true" class="icon-button" title="Help">
        <svg class="icon" viewBox="0 0 24 24">
          <path fill="currentColor" d="M11,18H13V16H11V18M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,20C7.59,20 4,16.41 4,12C4,7.59 7.59,4 12,4C16.41,4 20,7.59 20,12C20,16.41 16.41,20 12,20M12,6A4,4 0 0,0 8,10H10A2,2 0 0,1 12,8A2,2 0 0,1 14,10C14,12 11,11.75 11,15H13C13,12.75 16,12.5 16,10A4,4 0 0,0 12,6Z" />
        </svg>
      </button>
      <button @click="showGallery = true" class="icon-button" title="Gallery (Alt+↑/↓)">
        <svg class="icon" viewBox="0 0 24 24">
          <path fill="currentColor" d="M21 19V5C21 3.89 20.1 3 19 3H5C3.89 3 3 3.89 3 5V19C3 20.1 3.89 21 5 21H19C20.1 21 21 20.1 21 19M8.5 13.5L11 16.51L14.5 12L19 18H5L8.5 13.5Z"/>
        </svg>
      </button>

      <button
  @click="toggleSomap"
  class="icon-button"
  :class="{ active: $route.path === '/somap' }"
  title="Somap (Alt+2)"
>
  <svg class="icon" viewBox="0 0 24 24">
    <!-- Un círculo y un vórtice central, evocando un nodo de knowledge graph -->
    <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2" fill="none"/>
    <circle cx="12" cy="12" r="4" fill="currentColor" opacity="0.8"/>
    <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" stroke="currentColor" stroke-width="1.2" fill="none"/>
  </svg>
</button>
      </div>
      
      <Transition
        enter-active-class="fadeIn"
        leave-active-class="fadeOut"
        :duration="300"
        mode="out-in"
      >
        <div v-if="hasResults" class="results-panel" :key="transitionKey">
          <div class="results-split">
            <section class="panel panel-organogram">
              <h3 class="section-title">ORGANOGRAM</h3>
              <img
                v-if="plotImage"
                :src="plotImage"
                alt="Organogram"
                @click="openLightbox(plotImage, 'Organogram')"
                class="plot-image"
              />
            </section>

            <section class="panel panel-text">
              <div class="section-header">
                <h3 class="section-title">CONCEPTUAL SUMMARY</h3>
                <span v-if="responseMetaText" class="section-meta">{{ responseMetaText }}</span>
              </div>
              <div class="summary-content" v-html="summaryHtml"></div>
              <div class="section-header materials-title">
                <h3 class="section-title">MATERIALS</h3>
                <span v-if="responseMetaText" class="section-meta">{{ responseMetaText }}</span>
              </div>
              <pre class="materials-list">{{ materialsText }}</pre>
            </section>

            <section class="panel panel-visualizer">
              <div class="section-header tab-header">
                <div class="tabs">
                  <button 
                    class="tab-btn" 
                    :class="{ active: viewMode === 'stl' }" 
                    @click="viewMode = 'stl'"
                  >
                    GEOMETRY (3D)
                  </button>
                  <button 
                    class="tab-btn" 
                    :class="{ active: viewMode === 'sketch' }" 
                    @click="viewMode = 'sketch'"
                  >
                    SKETCH (INFERRED)
                  </button>
                </div>
                <div class="section-meta-group">
                  <button v-if="viewMode === 'stl' && stlUrl" @click="downloadCurrentStl" class="download-btn">
                    Download STL
                  </button>
                  <span v-if="viewMode === 'sketch' && sketchModel" class="section-meta">{{ sketchModel }}</span>
                  <button 
                    v-if="viewMode === 'sketch' && hasResults" 
                    @click="remakeSketch" 
                    class="remake-btn-small" 
                    :disabled="loading"
                  >
                    {{ loading ? '...' : 'GENERATE' }}
                  </button>
                </div>
              </div>

              <div class="tab-content">
                <div v-show="viewMode === 'stl'" class="tab-pane">
                  <ClientOnly>
                    <div v-if="stlUrl" class="stl-viewer-container">
                      <StlViewer :url="stlUrl" />
                    </div>
                    <div v-else class="stl-placeholder">No STL geometry generated for this response.</div>
                  </ClientOnly>
                </div>
                <div v-show="viewMode === 'sketch'" class="tab-pane">
                  <img
                    v-if="sketchImage"
                    :src="sketchImage"
                    alt="Sketch render"
                    @click="openLightbox(sketchImage, 'Sketch')"
                    class="plot-image"
                  />
                  <div v-else class="sketch-placeholder">No diffusion sketch generated for this response.</div>
                </div>
              </div>
            </section>
          </div>
        </div>
      </Transition>
    </div>

    <Transition
      enter-active-class="fadeIn"
      leave-active-class="fadeOut"
      :duration="300"
    >
      <div v-if="showLightbox" class="lightbox" @click="closeLightbox">
        <button class="close-button" @click.stop="closeLightbox">
          <svg class="icon" viewBox="0 0 24 24">
            <path fill="currentColor" d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" />
          </svg>
        </button>
        <img 
          :src="lightboxImage || plotImage" 
          :alt="lightboxAlt"
          class="lightbox-image"
          @click.stop
        />
      </div>
    </Transition>
    <div class="footer">
      <div v-if="loading" class="loading">
        <div class="loading-status">{{ loadingStatus }}</div>
        <div v-if="progressStage" class="progress-stage">{{ formatProgressStage(progressStage) }}</div>
        <div v-if="reasoningPreview" class="reasoning-preview" :title="progressStage || 'reasoning'">
          {{ reasoningPreview }}
        </div>
      </div>
      <button 
        v-if="isMobileOrTablet" 
        @click="handleMobileEvaluate"
        class="mobile-evaluate-btn"
        title="Alt+Enter"
      >
        Evaluate
      </button>
    </div>
    <div v-if="error" class="error">{{ error }}</div>
    <HelpModal v-model="showHelp" />
    <GalleryModal 
      v-model="showGallery" 
      @load-code="loadCodeFromGallery"
    />
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue';
import { useRuntimeConfig } from '#app';
import { marked } from 'marked';
import AceEditor from '~/components/AceEditor.vue';
import HelpModal from '~/components/HelpModal.vue';
import GalleryModal from '~/components/GalleryModal.vue';
import StlViewer from '~/components/StlViewer.vue';
import { useRandomPrompt } from '~/composables/useRandomPrompt';
import { useFavicon } from '~/composables/useFavicon';

// Configure marked
marked.setOptions({
  breaks: true,
  gfm: true,
});

// State variables
const { startProcessing, completeProcessing } = useFavicon();
const editorRef = ref(null);
const leftWidth = ref(50);
let dragging = false;
let startX = 0;
let startLeft = 50;
const loading = ref(false);
const progress = ref(0);
const elapsedMs = ref(0);
const responseTimesMs = ref([]);
const isReversioning = ref(false);
const error = ref(null);
const plotImage = ref(null);
const sketchImage = ref(null);
const summary = ref(null);
const organogramCode = ref('');
const geometryCode = ref('');
const stlUrl = ref(null);
const materialsText = ref('');
const ollamaModels = ref([]);
const currentModel = ref('');
const modelSwitching = ref(false);
const responseModel = ref('');
const responseElapsedMs = ref(0);
const sketchModel = ref('');
const lightboxImage = ref(null);
const lightboxAlt = ref('Preview');
const generationRequestId = ref('');
const reasoningPreview = ref('');
const progressStage = ref('');
const viewMode = ref('stl');
let progressPollInterval = null;

const summaryHtml = computed(() => {
  if (!summary.value) return '';
  return marked(summary.value);
});
const showCode = ref(true);
const showHelp = ref(false);
const showGallery = ref(false);
const transitionKey = ref(0);
const isMobileOrTablet = ref(false);
const showLightbox = ref(false);
const RESPONSE_TIMES_KEY = 'soog_response_times_ms';
const MAX_RESPONSE_SAMPLES = 20;

const averageResponseMs = computed(() => {
  if (!responseTimesMs.value.length) return 20000;
  const total = responseTimesMs.value.reduce((sum, ms) => sum + ms, 0);
  return total / responseTimesMs.value.length;
});

const loadingStatus = computed(() => {
  const pct = Math.round(progress.value);
  const elapsedSec = (elapsedMs.value / 1000).toFixed(1);
  const avgSec = (averageResponseMs.value / 1000).toFixed(1);
  const etaMs = Math.max(0, averageResponseMs.value - elapsedMs.value);
  const etaSec = (etaMs / 1000).toFixed(1);
  const prefix = isReversioning.value ? '[reversion] ' : '';
  return `${prefix}Processing... ${pct}% | ${elapsedSec}s elapsed | avg ${avgSec}s | ETA ${etaSec}s`;
});

const shortModelLabel = computed(() => {
  const model = currentModel.value || 'model?';
  return model.length > 22 ? `${model.slice(0, 22)}...` : model;
});

const responseMetaText = computed(() => {
  const parts = [];
  if (responseModel.value) parts.push(`model: ${responseModel.value}`);
  if (responseElapsedMs.value > 0) parts.push(`elapsed: ${(responseElapsedMs.value / 1000).toFixed(1)}s`);
  return parts.join(' | ');
});

const hasResults = computed(() => !!(plotImage.value || sketchImage.value || summary.value || materialsText.value || stlUrl.value));

const checkDevice = () => {
  isMobileOrTablet.value = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
};

const handleEscapeKey = (e) => {
  if (e.key === 'Escape' && showLightbox.value) {
    closeLightbox();
  }
};

function openLightbox(src, alt = 'Preview') {
  if (!src) return;
  lightboxImage.value = src;
  lightboxAlt.value = alt;
  showLightbox.value = true;
}

function closeLightbox() {
  showLightbox.value = false;
  lightboxImage.value = null;
  lightboxAlt.value = 'Preview';
}

function createRequestId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `soog-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
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
    const response = await fetch(`${apiBase.value}/generate/progress/${generationRequestId.value}`, {
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
  } catch {
    // keep progress polling quiet
  }
}

function startReasoningPolling() {
  stopReasoningPolling();
  reasoningPreview.value = '';
  progressStage.value = '';
  progressPollInterval = setInterval(fetchReasoningProgress, 700);
}

function formatProgressStage(stage) {
  return String(stage || '')
    .replace(/[_-]+/g, ' ')
    .trim()
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

onMounted(() => {
  loadResponseTimeHistory();
  fetchOllamaModels();
  checkDevice();
  window.addEventListener('resize', checkDevice);
  window.addEventListener('keydown', handleEscapeKey);
  window.addEventListener('mousemove', onDrag);
  window.addEventListener('mouseup', stopDrag);
  window.addEventListener('touchmove', onDragTouch, { passive: false });
  window.addEventListener('touchend', stopDrag);
});

onUnmounted(() => {
  clearInterval(progressInterval);
  stopReasoningPolling();
  window.removeEventListener('resize', checkDevice);
  window.removeEventListener('keydown', handleEscapeKey);
  window.removeEventListener('mousemove', onDrag);
  window.removeEventListener('mouseup', stopDrag);
  window.removeEventListener('touchmove', onDragTouch);
  window.removeEventListener('touchend', stopDrag);
});

const handleClear = () => {
  if (editorRef.value) {
    editorRef.value.clearEditor();
  }
};

const toggleShowCode = () => {
  showCode.value = !showCode.value;
};

function startDrag(e) {
  dragging = true;
  startX = e.clientX;
  startLeft = leftWidth.value;
}
function onDrag(e) {
  if (!dragging) return;
  const dx = e.clientX - startX;
  const vw = window.innerWidth;
  const deltaPct = (dx / vw) * 100;
  leftWidth.value = Math.min(80, Math.max(20, startLeft + deltaPct));
}
function stopDrag() { dragging = false; }
function startDragTouch(e) {
  dragging = true;
  startX = e.touches[0].clientX;
  startLeft = leftWidth.value;
}
function onDragTouch(e) {
  if (!dragging) return;
  const dx = e.touches[0].clientX - startX;
  const vw = window.innerWidth;
  const deltaPct = (dx / vw) * 100;
  leftWidth.value = Math.min(80, Math.max(20, startLeft + deltaPct));
}

const handleRandomPrompt = async () => {
  if (editorRef.value) {
    const { getRandomPrompt } = useRandomPrompt();
    const prompt = await getRandomPrompt();
    editorRef.value.clearEditor();
    editorRef.value.addToEditor(prompt);
    // Add random prompt to command history
    editorRef.value.addToHistory(prompt);
  }
};

const handleMobileEvaluate = () => {
  if (editorRef.value) {
    const editor = editorRef.value.aceEditor();
    if (editor) {
      const selectedText = editor.getSelectedText();
      const textToEvaluate = selectedText || editor.getValue();
      if (textToEvaluate.trim()) {
        startProcessing();
        handleEvaluate(textToEvaluate);
      } else {
        error.value = 'Please enter some text to evaluate.';
      }
    }
  }
};

const loadResponseTimeHistory = () => {
  if (typeof window === 'undefined') return;
  try {
    const raw = window.localStorage.getItem(RESPONSE_TIMES_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return;
    responseTimesMs.value = parsed
      .filter((n) => Number.isFinite(n) && n > 0 && n < 10 * 60 * 1000)
      .slice(-MAX_RESPONSE_SAMPLES);
  } catch {
    responseTimesMs.value = [];
  }
};

const saveResponseTimeHistory = () => {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(RESPONSE_TIMES_KEY, JSON.stringify(responseTimesMs.value.slice(-MAX_RESPONSE_SAMPLES)));
  } catch {
    // ignore storage write errors
  }
};

const registerResponseTime = (ms) => {
  if (!Number.isFinite(ms) || ms <= 0) return;
  responseTimesMs.value.push(ms);
  if (responseTimesMs.value.length > MAX_RESPONSE_SAMPLES) {
    responseTimesMs.value = responseTimesMs.value.slice(-MAX_RESPONSE_SAMPLES);
  }
  saveResponseTimeHistory();
};

// Progress based on real elapsed time and rolling average
let progressInterval;
let requestStartAt = 0;
const startProgress = () => {
  clearInterval(progressInterval);
  requestStartAt = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  elapsedMs.value = 0;
  progress.value = 0;
  progressInterval = setInterval(() => {
    const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    elapsedMs.value = Math.max(0, now - requestStartAt);

    // Keep progress realistic: approach ~92% by average time, then slowly creep to 98%.
    const avg = Math.max(3000, averageResponseMs.value);
    const ratio = elapsedMs.value / avg;
    if (ratio <= 1) {
      progress.value = Math.min(92, ratio * 92);
    } else {
      const over = (elapsedMs.value - avg) / avg;
      progress.value = Math.min(98, 92 + Math.log1p(over) * 6);
    }
  }, 200);
};

const completeProgress = () => {
  const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  if (requestStartAt > 0) {
    elapsedMs.value = Math.max(0, now - requestStartAt);
    registerResponseTime(elapsedMs.value);
  }
  clearInterval(progressInterval);
  progress.value = 100;
  setTimeout(() => {
    progress.value = 0;
    elapsedMs.value = 0;
    requestStartAt = 0;
  }, 500);
};

// Runtime configuration
const config = useRuntimeConfig();
const apiBase = ref(config.public.apiBase || 'http://127.0.0.1:10000/api');
const getGenerateFetchTimeoutMs = () => {
  const value = Number(config.public.generateTimeoutMs);
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.floor(value));
};

function formatGeneratedCode(plotCode, stlCode) {
  const sections = [];
  if (plotCode && plotCode.trim()) {
    sections.push([
      "## organogram (matplotlib)",
      "```python",
      plotCode.trim(),
      "```"
    ].join('\n'));
  }
  if (stlCode && stlCode.trim()) {
    sections.push([
      "## geometry (trimesh)",
      "```python",
      stlCode.trim(),
      "```"
    ].join('\n'));
  }
  return sections.join('\n\n');
}

function safeToken(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function inferTitleToken(item) {
  const explicit = safeToken(item?.title_slug || item?.title);
  if (explicit) return explicit;
  const basename = String(item?.basename || '');
  if (!basename) return '';
  const rawTail = basename.split('_').slice(1).join('_');
  if (!rawTail) return '';
  const withoutVersion = rawTail.replace(/_v\d+(?:_\d+)?(?:_\d+)?$/i, '');
  return safeToken(withoutVersion || rawTail);
}

function resolveAssetUrl(url) {
  if (!url) return '';
  if (url.startsWith('http')) return url;
  if (apiBase.value.endsWith('/api') && url.startsWith('/api/')) {
    return apiBase.value + url.substring(4);
  }
  return apiBase.value + url;
}

function stlFilenameFromUrl(url, fallback = 'model.stl') {
  try {
    if (!url) return fallback;
    const pathname = new URL(resolveAssetUrl(url), window.location.origin).pathname;
    const last = pathname.split('/').pop() || '';
    if (last.toLowerCase().endsWith('.stl')) return last;
    return fallback;
  } catch {
    return fallback;
  }
}

async function remakeSketch() {
  if (loading.value || !hasResults.value) return
  loading.value = true
  progressStage.value = 'regenerating sketch'
  try {
    const prompt = editorRef.value?.aceEditor()?.getValue() || ''
    const summaryText = summary.value || ''
    const materials_text = materialsText.value || ''
    const plot_code = organogramCode.value || ''

    const response = await fetch(`${apiBase.value}/generate/sketch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        summary: summaryText,
        materials: materials_text,
        plot_code,
        image: plotImage.value?.split(',')[1] // send base64 if available
      })
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.error || 'Failed to remake sketch')
    
    if (data.sketch) {
      sketchImage.value = `data:image/png;base64,${data.sketch}`
    } else if (data.sketch_url) {
      sketchImage.value = resolveAssetUrl(data.sketch_url)
    }
    sketchModel.value = data.sketch_model || ''
  } catch (e) {
    error.value = `Sketch remake failed: ${e.message}`
  } finally {
    loading.value = false
    progressStage.value = ''
  }
}

async function downloadCurrentStl() {
  if (!stlUrl.value) return;
  try {
    const response = await fetch(stlUrl.value);
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload?.error || `STL download failed (${response.status})`);
    }
    const blob = await response.blob();
    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = stlFilenameFromUrl(stlUrl.value);
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(blobUrl);
  } catch (e) {
    error.value = e?.message || 'STL download failed';
  }
}

function buildRefactorDraft(item) {
  if (!item || typeof item !== 'object') return '';

  const source = safeToken(item.basename);
  const group = safeToken(item.group_id || item.basename || item.title_slug || item.title);
  const title = inferTitleToken(item);

  const headerTokens = ['REFACT'];
  if (source) headerTokens.push(`source=${source}`);
  if (group) headerTokens.push(`group=${group}`);
  if (title) headerTokens.push(`title=${title}`);
  const header = `[${headerTokens.join(' ')}]`;

  const summaryText = (item.summary || item.answer || '').trim();
  const materials = (item.materials_text || '').trim();
  const plotCode = (item.plot_code || item.code || '').trim();
  const stlCode = (item.stl_code || '').trim();

  const lines = [
    header,
    'Refine this existing organogram and geometry as a new version.',
    'Keep the same instrument identity and improve only what is requested.',
    '',
    'Change Request:',
    '- describe the corrections/additions for this next version',
    '',
    'BASE CONTEXT (for iteration):',
    '',
    '## Base Prompt',
    (item.prompt || '(no stored prompt)').trim(),
    '',
    '## Base Conceptual Summary',
    summaryText || '(no stored summary)',
    '',
    '## Base Materials',
    materials || '(no stored materials)',
    ''
  ];

  if (plotCode) {
    lines.push('## Base Organogram Code (matplotlib)');
    lines.push('```python');
    lines.push(plotCode);
    lines.push('```');
    lines.push('');
  }

  if (stlCode) {
    lines.push('## Base Geometry Code (trimesh)');
    lines.push('```python');
    lines.push(stlCode);
    lines.push('```');
    lines.push('');
  }

  return `${lines.join('\n').trim()}\n`;
}

async function loadCodeFromGallery(item) {
  if (!item || !editorRef.value) return;

  const draft = buildRefactorDraft(item);
  if (editorRef.value.setEditorContent) {
    editorRef.value.setEditorContent(draft);
  } else {
    editorRef.value.clearEditor();
    editorRef.value.addToEditor(draft, 'code');
  }
  if (editorRef.value.addToHistory) {
    editorRef.value.addToHistory(draft);
  }

  summary.value = item.summary || item.answer || null;
  materialsText.value = item.materials_text || '';
  organogramCode.value = (item.plot_code || item.code || '').trim();
  geometryCode.value = (item.stl_code || '').trim();
  responseModel.value = (item.llm_model || '').trim();
  sketchModel.value = (item.sketch_model || '').trim();
  responseElapsedMs.value = Number.isFinite(Number(item.elapsed_ms))
    ? Number(item.elapsed_ms)
    : 0;

  stlUrl.value = item.stl_url ? resolveAssetUrl(item.stl_url) : null;
  plotImage.value = item.image_url ? resolveAssetUrl(item.image_url) : null;
  sketchImage.value = item.sketch_url ? resolveAssetUrl(item.sketch_url) : null;
  showGallery.value = false;
}

async function fetchOllamaModels() {
  try {
    const response = await fetch(`${apiBase.value}/ollama/models`, {
      headers: { 'Accept': 'application/json' }
    });
    const payload = await response.json();
    if (!response.ok || !payload?.ok) {
      return;
    }
    ollamaModels.value = Array.isArray(payload.models) ? payload.models : [];
    currentModel.value = payload.current_model || '';
  } catch {
    // keep UI quiet when models endpoint isn't available
  }
}

async function cycleOllamaModel() {
  if (modelSwitching.value) return;
  modelSwitching.value = true;
  try {
    if (!ollamaModels.value.length) {
      await fetchOllamaModels();
    }
    if (ollamaModels.value.length < 2) {
      error.value = 'Need at least 2 installed Ollama models to cycle.';
      return;
    }
    const currentIdx = Math.max(0, ollamaModels.value.indexOf(currentModel.value));
    const nextIdx = (currentIdx + 1) % ollamaModels.value.length;
    const nextModel = ollamaModels.value[nextIdx];

    const response = await fetch(`${apiBase.value}/ollama/model`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ model: nextModel })
    });
    const payload = await response.json();
    if (!response.ok || !payload?.ok) {
      throw new Error(payload?.error || `Failed to switch model (${response.status})`);
    }
    currentModel.value = payload.model || nextModel;
    error.value = null;
  } catch (e) {
    error.value = e?.message || 'Model switch failed.';
  } finally {
    modelSwitching.value = false;
  }
}

function isRefactorPrompt(text) {
  const trimmed = String(text || '').trim();
  if (!trimmed) return false;
  const first = trimmed.split('\n')[0].trim();
  if (!first) return false;
  return first.startsWith('[REFACT') || first.startsWith('*') || first.startsWith('+');
}

// Handle evaluation of selected text
const handleEvaluate = async (selectedText) => {
  if (!selectedText.trim()) {
    error.value = 'Please select some text to evaluate.';
    return;
  }

  const requestStartedAt = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  loading.value = true;
  isReversioning.value = isRefactorPrompt(selectedText);
  error.value = null;
  generationRequestId.value = createRequestId();
  reasoningPreview.value = '';
  progressStage.value = '';
  startProgress();
  startProcessing();
  startReasoningPolling();

    async function callOnce() {
      const timeoutMs = getGenerateFetchTimeoutMs();
      const controller = timeoutMs > 0 ? new AbortController() : null;
      const timeoutId = timeoutMs > 0 ? setTimeout(() => controller?.abort(), timeoutMs) : null;
      let response;
      try {
        response = await fetch(`${apiBase.value}/generate`, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({ prompt: selectedText, request_id: generationRequestId.value }),
          signal: controller?.signal
        });
      } catch (fetchErr) {
        if (fetchErr?.name === 'AbortError') {
          if (timeoutMs > 0) {
            throw new Error(`Generation timed out after ${Math.round(timeoutMs / 1000)}s. Try a shorter prompt or lighter model.`);
          }
          throw new Error('Generation request was aborted.');
        }
        throw fetchErr;
      } finally {
        if (timeoutId) clearTimeout(timeoutId);
      }
      
      // Read body only once as text
      const text = await response.text();
      
      if (!response.ok) {
        // Handle 502 Bad Gateway and similar errors
        if (response.status === 502 || response.status === 503) {
          throw new Error('Backend service unavailable (502/503). Please check server status.');
        }
        
        // Try to parse as JSON for error message
        try {
          const errorData = JSON.parse(text);
          throw new Error(errorData.error || `Server error ${response.status}`);
        } catch (jsonError) {
          throw new Error(text || `Server error ${response.status}`);
        }
      }
      
      // Validate response has content
      if (!text || text.trim().length === 0) {
        throw new Error("Empty response from server");
      }
      
      const contentType = response.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        throw new Error("Invalid response type from server");
      }
      
      return JSON.parse(text);
    }

  function extractMaterials(text) {
    if (!text) return [];
    const physical = [
      'spruce','maple','rosewood','ebony','mahogany','cedar','pine','oak','bamboo','reed','brass','bronze','copper','steel','aluminum','nickel','silver','gold','titanium','carbon fiber','fiberglass','plastic','acrylic','rubber','leather','gut','nylon','silk','ceramic','clay','glass','cork','felt'
    ];
    const virtual = [
      'texture','shader','sample','sampling','synthesis','granular','wavetable','fm','additive','subtractive','midi','vst','plugin','max/msp','pure data','supercollider','osc','convolution','impulse response','ir','reverb','impulse','unity','unreal','game engine','shader graph','material graph'
    ];
    const found = new Map();
    const lower = text.toLowerCase();
    function add(name, type) {
      const key = name.toLowerCase();
      if (!found.has(key)) found.set(key, { name, type });
    }
    physical.forEach(w => { if (lower.includes(w)) add(w, 'physical'); });
    virtual.forEach(w => { if (lower.includes(w)) add(w, 'virtual'); });
    return Array.from(found.values());
  }

  try {
    const data = await callOnce();

    // Reset results
    plotImage.value = null;
    sketchImage.value = null;
    summary.value = null;
    organogramCode.value = '';
    geometryCode.value = '';
    stlUrl.value = null;
    materialsText.value = '';
    responseModel.value = '';
    responseElapsedMs.value = 0;
    sketchModel.value = '';

    const imageUrl = data.image_url || data.gallery?.image_url || null;
    const sketchUrl = data.sketch_url || data.gallery?.sketch_url || null;
    if (data.image) {
      plotImage.value = `data:image/png;base64,${data.image}`;
    } else if (imageUrl) {
      plotImage.value = resolveAssetUrl(imageUrl);
    } else {
      throw new Error('Backend did not return an organogram image. Generation aborted.');
    }
    if (data.sketch) {
      sketchImage.value = `data:image/png;base64,${data.sketch}`;
    } else if (sketchUrl) {
      sketchImage.value = resolveAssetUrl(sketchUrl);
    }

    if (data.summary) summary.value = data.summary;
    responseModel.value = (data.llm_model || currentModel.value || '').trim();
    sketchModel.value = (data.sketch_model || data.gallery?.sketch_model || '').trim();
    const requestEndedAt = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const clientElapsed = Math.max(0, requestEndedAt - requestStartedAt);
    responseElapsedMs.value = Number.isFinite(Number(data.elapsed_ms))
      ? Number(data.elapsed_ms)
      : Math.round(clientElapsed);

    organogramCode.value = (data.plot_code || data.content || '').trim();
    geometryCode.value = (data.stl_code || '').trim();

    stlUrl.value = data.gallery?.stl_url ? resolveAssetUrl(data.gallery.stl_url) : null;

    if (typeof data.materials === 'string' && data.materials.trim()) {
      materialsText.value = data.materials.trim();
    } else if (Array.isArray(data.materials) && data.materials.length) {
      materialsText.value = data.materials.map(x => (typeof x === 'string' ? x : (x.name || ''))).filter(Boolean).join('\n');
    } else if (summary.value) {
      const list = extractMaterials(summary.value).map(m => `- ${m.name}`);
      materialsText.value = list.join('\n');
    }

    // Append structured code sections in editor (left panel)
    const codeBundle = formatGeneratedCode(organogramCode.value, geometryCode.value);
    if (showCode.value && codeBundle && editorRef.value) {
      editorRef.value.addToEditor(codeBundle, 'code');
    }

    transitionKey.value++;
  } catch (err) {
    console.error(err);
    error.value = err.message;
  } finally {
    await fetchReasoningProgress();
    stopReasoningPolling();
    generationRequestId.value = '';
    completeProgress();
    loading.value = false;
    isReversioning.value = false;
    completeProcessing();
  }
};

import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

function toggleSomap() {
  if (route.path === '/somap') {
    router.push('/')
  } else {
    router.push('/somap')
  }
}
function goToSoog() {
  if (route.path !== '/') router.push('/')
}
function goToSomap() {
  if (route.path !== '/somap') router.push('/somap')
}

function isAltDigitShortcut(event, digit) {
  return event.altKey && (event.code === `Digit${digit}` || event.key === String(digit))
}

// Shortcuts Alt+1 (Soog), Alt+2 (Somap)
function handleToggleShortcuts(e) {
  if (isAltDigitShortcut(e, 1)) {
    goToSoog()
    e.preventDefault()
    return
  }
  if (isAltDigitShortcut(e, 2)) {
    goToSomap()
    e.preventDefault()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleToggleShortcuts)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleToggleShortcuts)
})

// Alt+ArrowUp/Down to navigate saved gallery outputs and load code
let galleryItems = []
let galleryIndex = -1

async function ensureGalleryLoaded() {
  if (galleryItems.length === 0) {
    try {
      const res = await fetch(`${apiBase.value}/gallery/list`)
      const data = await res.json()
      galleryItems = data.items || []
      galleryIndex = galleryItems.length > 0 ? 0 : -1
    } catch (e) {
      console.error('Failed to load gallery', e)
    }
  }
}

async function handleGalleryArrows(e) {
  if (!e.altKey) return
  if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
  e.preventDefault()
  await ensureGalleryLoaded()
  if (galleryIndex === -1) return
  if (e.key === 'ArrowUp') {
    galleryIndex = Math.max(0, galleryIndex - 1)
  } else if (e.key === 'ArrowDown') {
    galleryIndex = Math.min(galleryItems.length - 1, galleryIndex + 1)
  }
  const item = galleryItems[galleryIndex]
  if (item && editorRef.value) {
    await loadCodeFromGallery(item)
  }
}

onMounted(() => window.addEventListener('keydown', handleGalleryArrows))
onUnmounted(() => window.removeEventListener('keydown', handleGalleryArrows))

</script>

<style scoped>
.app-container {
  display: flex;
  flex-direction: row;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  position: relative;
  gap: 0;
  padding: 0 !important;
  margin: 0;
}

.left-column {
  width: 50%;
  height: 100vh;
  display: block;
  min-width: 0;
  overflow: hidden;
}

.right-column {
  width: 50%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.divider {
  width: 1px;
  cursor: col-resize;
  background: rgba(255, 255, 255, 0.25);
  z-index: 5;
  flex-shrink: 0;
  position: relative;
}

.divider::before {
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  left: -4px;
  right: -4px;
  background: transparent;
}

.editor-wrapper {
  flex: 1;
  width: 100%;
  height: 100%;
  position: relative;
  overflow: hidden;
  margin: 0;
  padding: 0;
}

.left-column :deep(.ace_editor),
.left-column :deep(.ace_scroller),
.left-column :deep(.ace_content) {
  margin: 0 !important;
  padding: 0 !important;
}

.hud {
  padding: 8px 16px;
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  align-items: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.16);
}

.icon-button {
  background: transparent;
  border: none;
  color: white;
  padding: 6px;
  cursor: pointer;
  border-radius: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: opacity 0.2s;
}

.icon-button:hover {
  opacity: 0.7;
}

.icon-button.active,
.icon-button:hover {
  opacity: 1;
  filter: none;
}

.icon-button:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.model-name {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.7);
  margin-right: 2px;
  letter-spacing: 0.03em;
}

.model-cycle-button {
  margin-right: 2px;
}

.icon {
  width: 24px;
  height: 24px;
}

.plot-image {
  cursor: pointer;
  transition: transform 0.3s;
}

.plot-image:hover {
  transform: scale(1.02);
}

.lightbox {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}

.lightbox-image {
  max-width: 80vw;
  max-height: 80vh;
  object-fit: contain;
}

.close-button {
  position: absolute;
  top: 20px;
  right: 20px;
  background: transparent;
  border: none;
  color: white;
  cursor: pointer;
  padding: 8px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.3s;
}

.close-button:hover {
  background: rgba(255, 255, 255, 0.1);
}

.footer {
  position: fixed;
  bottom: 0;
  right: 0;
  left: 0;
  display: flex;
  justify-content: flex-end;
  align-items: center;
  padding: 1rem;
  gap: 1rem;
  background: black !important;
  z-index: 1000;
}

.loading {
  margin-right: auto;
}

.loading-status {
  color: rgba(255, 255, 255, 0.9);
}

.progress-stage {
  margin-top: 3px;
  color: rgba(255, 255, 255, 0.46);
  font-size: 11px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.reasoning-preview {
  margin-top: 4px;
  max-width: min(62vw, 860px);
  color: rgba(255, 255, 255, 0.3);
  font-size: 11px;
  line-height: 1.3;
  white-space: normal;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.mobile-evaluate-btn {
  background: #4CAF50;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  transition: background-color 0.3s;
}

.mobile-evaluate-btn:hover {
  background: #45a049;
}

.mobile-evaluate-btn:active {
  background: #3d8b40;
  transform: translateY(1px);
}

.error {
  position: fixed;
  bottom: 60px;
  left: 50%;
  transform: translateX(-50%);
  background: #ff5252;
  color: white;
  padding: 8px 16px;
  border-radius: 4px;
  z-index: 1000;
}

.results-panel {
  height: 100%;
  background: #000;
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
}

.results-split {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  background: transparent;
  display: grid;
  grid-template-rows: minmax(250px, 1fr) minmax(250px, 1fr) minmax(220px, 0.95fr) minmax(260px, 1fr);
  gap: 0;
  padding: 0 16px 88px 16px;
  box-sizing: border-box;
}

.panel {
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 12px 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.panel + .panel {
  border-top: 1px solid rgba(255, 255, 255, 0.2);
}

.section-title {
  margin: 0 0 10px 0;
  font-size: 12px;
  letter-spacing: 0.12em;
  color: rgba(255, 255, 255, 0.78);
  font-weight: 400;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
}

.section-meta {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.58);
  white-space: nowrap;
}

.panel-organogram {
  justify-content: center;
}

.panel-visualizer {
  min-height: 380px;
}

.tab-header {
  margin-bottom: 12px;
}

.tabs {
  display: flex;
  gap: 16px;
}

.tab-btn {
  background: transparent;
  border: none;
  color: rgba(255, 255, 255, 0.4);
  font-size: 11px;
  letter-spacing: 0.12em;
  padding: 0 0 4px 0;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s;
}

.tab-btn:hover {
  color: rgba(255, 255, 255, 0.8);
}

.tab-btn.active {
  color: #4CAF50;
  border-bottom-color: #4CAF50;
}

.remake-btn-small {
  background: rgba(255, 255, 255, 0.1);
  color: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 10px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  margin-left: 8px;
}

.remake-btn-small:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.2);
  color: white;
  border-color: rgba(255, 255, 255, 0.4);
}

.remake-btn-small:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.tab-content {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.tab-pane {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.section-meta-group {
  display: flex;
  align-items: center;
  gap: 12px;
}

.panel-text {
  overflow: auto;
}

.materials-title {
  margin-top: 14px;
}

.materials-list {
  white-space: pre-wrap;
  margin: 0;
  color: rgba(255, 255, 255, 0.86);
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0;
  font-size: 13px;
  line-height: 1.5;
}

.stl-viewer-container {
  width: 100%;
  flex: 1;
  min-height: 240px;
}

.stl-placeholder {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #9e9e9e;
  font-size: 13px;
  border: none;
  border-radius: 0;
}

.sketch-placeholder {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #9e9e9e;
  font-size: 13px;
  border: none;
  border-radius: 0;
}

.download-btn {
  background: transparent;
  color: rgba(255, 255, 255, 0.86);
  border: none;
  padding: 0;
  border-radius: 0;
  font-size: 12px;
  cursor: pointer;
  text-decoration: none;
  transition: opacity 0.2s;
}

.download-btn:hover {
  opacity: 0.7;
}

.plot-image {
  width: 100%;
  height: 100%;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  display: block;
  border-radius: 0;
}

.summary-content {
  color: #eee;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  font-size: 14px;
  line-height: 1.7;
  width: 100%;
  text-align: left;
}

.summary-content :deep(h1),
.summary-content :deep(h2),
.summary-content :deep(h3),
.summary-content :deep(h4),
.summary-content :deep(h5),
.summary-content :deep(h6) {
  color: #4CAF50;
  margin-top: 1.5em;
  margin-bottom: 0.5em;
  font-weight: 600;
}

.summary-content :deep(h1) { font-size: 1.8em; }
.summary-content :deep(h2) { font-size: 1.5em; }
.summary-content :deep(h3) { font-size: 1.3em; }

.summary-content :deep(p) {
  margin: 0.8em 0;
}

.summary-content :deep(ul),
.summary-content :deep(ol) {
  margin: 1em 0;
  padding-left: 2em;
}

.summary-content :deep(li) {
  margin: 0.4em 0;
}

.summary-content :deep(code) {
  background: rgba(255, 255, 255, 0.1);
  padding: 2px 6px;
  border-radius: 3px;
  font-family: 'Courier New', monospace;
  font-size: 0.9em;
}

.summary-content :deep(pre) {
  background: rgba(255, 255, 255, 0.05);
  padding: 12px;
  border-radius: 4px;
  overflow-x: auto;
  margin: 1em 0;
}

.summary-content :deep(pre code) {
  background: none;
  padding: 0;
}

.summary-content :deep(a) {
  color: #4CAF50;
  text-decoration: none;
}

.summary-content :deep(a:hover) {
  text-decoration: underline;
}

.summary-content :deep(strong) {
  font-weight: 600;
  color: #fff;
}

.summary-content :deep(em) {
  font-style: italic;
  color: #ddd;
}

.summary-content :deep(blockquote) {
  border-left: 3px solid #4CAF50;
  padding-left: 1em;
  margin: 1em 0;
  color: #aaa;
}

@media (max-width: 768px) {
  .app-container {
    flex-direction: column;
  }
  
  .left-column, .right-column {
    width: 100%;
  }
  .left-column { height: 50vh; }
  .right-column { height: 50vh; }
  .results-split {
    grid-template-rows: minmax(180px, 1fr) minmax(180px, 1fr) minmax(180px, 0.95fr) minmax(220px, 1fr);
    padding-bottom: 110px;
  }
}
</style>
