<template>
  <div class="app-container">
    <div class="settings">
      <button @click="toggleShowCode" class="toggle-button">
        {{ showCode ? 'Hide Code' : 'Show Code' }}
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
      :duration="5000"
      mode="out-in"
    > 
      <div v-if="plotImage" class="plot-display" :key="transitionKey">
        <img 
          :src="`data:image/png;base64,${plotImage}`" 
          alt="Plot"
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
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useRuntimeConfig } from '#app';
import AceEditor from '~/components/AceEditor.vue';

// State variables
const editorRef = ref(null);
const loading = ref(false);
const progress = ref(0);
const error = ref(null);
const plotImage = ref(null);
const showCode = ref(true);
const transitionKey = ref(0);
const isMobileOrTablet = ref(false);

const checkDevice = () => {
  isMobileOrTablet.value = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
};

onMounted(() => {
  checkDevice();
  window.addEventListener('resize', checkDevice);
});

const toggleShowCode = () => {
  showCode.value = !showCode.value;
};

const handleMobileEvaluate = () => {
  if (editorRef.value) {
    const selectedText = editorRef.value.getSelectedText();
    const textToEvaluate = selectedText || editorRef.value.getValue();
    handleEvaluate(textToEvaluate);
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
  }, 1100);
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
const apiBase = ref(config.public.apiBase || 'https://soog.onrender.com/api'); // Updated to use backend's public URL

// Handle evaluation of selected text
const handleEvaluate = async (selectedText) => {
  if (!selectedText.trim()) {
    error.value = 'Please select some text to evaluate.';
    return;
  }

  loading.value = true;
  error.value = null;
  startProgress();

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
  }
};
</script>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  position: relative;
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
