<template>
  <div class="app-container">
    <div class="settings">
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
    <div class="editor-wrapper">
      <AceEditor 
        ref="editorRef" 
        @evaluate="handleEvaluate"
      />
    </div>
    <Transition
      enter-active-class="fadeIn"
      leave-active-class="fadeOut"
      :duration="300"
      mode="out-in"
    > 
      <div v-if="hasResults" class="results-panel" :key="transitionKey">
        <div class="results-header">
          <div class="tabs">
            <button 
              v-if="plotImage" 
              :class="{active: activeTab==='plot'}" 
              @click="activeTab='plot'"
            >
              Organogram
            </button>
            <button 
              v-if="summary" 
              :class="{active: activeTab==='summary'}" 
              @click="activeTab='summary'"
            >
              Summary
            </button>
            <button 
              v-if="generatedCode" 
              :class="{active: activeTab==='code'}" 
              @click="activeTab='code'"
            >
              Code
            </button>
            <button 
              v-if="stlUrl" 
              :class="{active: activeTab==='stl'}" 
              @click="activeTab='stl'"
            >
              3D Model
            </button>
          </div>
          <div class="actions">
            <a v-if="stlUrl" :href="stlUrl" class="download-btn" download>
              Download STL
            </a>
            <button @click="expandResults = !expandResults" class="icon-button" :title="expandResults ? 'Minimize' : 'Expand'">
              <svg v-if="expandResults" class="icon" viewBox="0 0 24 24">
                <path fill="currentColor" d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" />
              </svg>
              <svg v-else class="icon" viewBox="0 0 24 24">
                <path fill="currentColor" d="M7,14L12,9L17,14H7Z" />
              </svg>
            </button>
          </div>
        </div>
        <div v-if="expandResults" class="results-content">
          <div v-if="activeTab==='plot' && plotImage" class="tab-content">
            <img 
              :src="`data:image/png;base64,${plotImage}`" 
              alt="Plot"
              @click="showLightbox = true"
              class="plot-image"
            />
          </div>
          <div v-if="activeTab==='summary' && summary" class="tab-content">
            <pre class="summary-text">{{ summary }}</pre>
          </div>
          <div v-if="activeTab==='code' && generatedCode" class="tab-content">
            <pre class="code-text">{{ generatedCode }}</pre>
          </div>
          <div v-if="activeTab==='stl' && stlUrl" class="tab-content">
            <ClientOnly>
              <StlViewer :url="stlUrl" />
            </ClientOnly>
          </div>
        </div>
      </div>
    </Transition>
    <Transition
      enter-active-class="fadeIn"
      leave-active-class="fadeOut"
      :duration="300"
    >
      <div v-if="showLightbox" class="lightbox" @click="showLightbox = false">
        <button class="close-button" @click.stop="showLightbox = false">
          <svg class="icon" viewBox="0 0 24 24">
            <path fill="currentColor" d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" />
          </svg>
        </button>
        <img 
          :src="`data:image/png;base64,${plotImage}`" 
          alt="Plot"
          class="lightbox-image"
          @click.stop
        />
      </div>
    </Transition>
    <div class="footer">
      <div v-if="loading" class="loading">
        Processing... {{ Math.round(progress) }}%
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
      @load-code="code => { if (editorRef?.value) { editorRef.value.clearEditor(); editorRef.value.addToEditor(code, 'code') } }" 
    />
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import { useRuntimeConfig } from '#app';
import AceEditor from '~/components/AceEditor.vue';
import HelpModal from '~/components/HelpModal.vue';
import GalleryModal from '~/components/GalleryModal.vue';
import StlViewer from '~/components/StlViewer.vue';
import { useRandomPrompt } from '~/composables/useRandomPrompt';
import { useFavicon } from '~/composables/useFavicon';

// State variables
const { startProcessing, completeProcessing } = useFavicon();
const editorRef = ref(null);
const loading = ref(false);
const progress = ref(0);
const error = ref(null);
const plotImage = ref(null);
const summary = ref(null);
const generatedCode = ref(null);
const stlUrl = ref(null);
const activeTab = ref('plot');
const expandResults = ref(true);
const showCode = ref(true);
const showHelp = ref(false);
const showGallery = ref(false);
const transitionKey = ref(0);
const isMobileOrTablet = ref(false);
const showLightbox = ref(false);

const hasResults = computed(() => !!(plotImage.value || summary.value || generatedCode.value || stlUrl.value));

const checkDevice = () => {
  isMobileOrTablet.value = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
};

const handleEscapeKey = (e) => {
  if (e.key === 'Escape' && showLightbox.value) {
    showLightbox.value = false;
  }
};

onMounted(() => {
  checkDevice();
  window.addEventListener('resize', checkDevice);
  window.addEventListener('keydown', handleEscapeKey);
});

onUnmounted(() => {
  window.removeEventListener('resize', checkDevice);
  window.removeEventListener('keydown', handleEscapeKey);
});

const handleClear = () => {
  if (editorRef.value) {
    editorRef.value.clearEditor();
  }
};

const toggleShowCode = () => {
  showCode.value = !showCode.value;
};

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

// Progress simulation
let progressInterval;
const startProgress = () => {
  progress.value = 0;
  progressInterval = setInterval(() => {
    if (progress.value < 90) {
      progress.value += Math.random() * 15;
      if (progress.value > 90) progress.value = 90;
    }
  }, 1200);
};

const completeProgress = () => {
  clearInterval(progressInterval);
  progress.value = 100;
  setTimeout(() => {
    progress.value = 0;
  }, 500);
};

// Runtime configuration
const config = useRuntimeConfig();
const apiBase = ref(config.public.apiBase || 'https://soog.onrender.com/api');

// Handle evaluation of selected text
const handleEvaluate = async (selectedText) => {
  if (!selectedText.trim()) {
    error.value = 'Please select some text to evaluate.';
    return;
  }

  loading.value = true;
  error.value = null;
  startProgress();
  startProcessing();

  try {
    const response = await fetch(`${apiBase.value}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: selectedText }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to process request');
    }

    const data = await response.json();
    
    // Add response to editor with appropriate styling
    if (data.type === 'text') {
      editorRef.value.addToEditor(data.content, 'text');
      plotImage.value = null;
    } else if (data.type === 'code') {
      if (showCode.value) {
        editorRef.value.addToEditor(data.content, 'code');
      }
      plotImage.value = null;
    } else if (data.type === 'plot') {
      if (showCode.value) {
        editorRef.value.addToEditor(data.content, 'plot');
      }
      transitionKey.value++; // Increment transition key for new plot
      plotImage.value = data.image;
    } else {
      editorRef.value.addToEditor('Unexpected response type.', 'text');
      plotImage.value = null;
    }
  } catch (err) {
    console.error(err);
    error.value = err.message;
  } finally {
    completeProgress();
    loading.value = false;
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

// Shortcuts Alt+1 (Soog), Alt+2 (Somap)
function handleToggleShortcuts(e) {
  if (e.altKey && e.key === '1') {
    goToSoog()
    e.preventDefault()
  }
  if (e.altKey && e.key === '2') {
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
  if (item?.code && editorRef.value) {
    editorRef.value.clearEditor()
    editorRef.value.addToEditor(item.code, 'code')
  }
}

onMounted(() => window.addEventListener('keydown', handleGalleryArrows))
onUnmounted(() => window.removeEventListener('keydown', handleGalleryArrows))

</script>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  position: relative;
}

.settings {
  position: fixed;
  top: 10px;
  right: 20px;
  z-index: 100;
  display: flex;
  gap: 8px;
}

.icon-button {
  background: transparent;
  border: none;
  color: white;
  padding: 8px;
  cursor: pointer;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.3s;
}

.icon-button:hover {
  background: rgba(255, 255, 255, 0.1);
}

.icon-button.active,
.icon-button:hover {
  opacity: 1;
  filter: drop-shadow(0 0 4px #fff);
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
</style>
