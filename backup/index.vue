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
    <div v-if="loading" class="loading">
      Processing... {{ Math.round(progress) }}%
    </div>
    <div v-if="error" class="error">{{ error }}</div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
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

const toggleShowCode = () => {
  showCode.value = !showCode.value;
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
const apiBase = ref(config.public.apiBase || 'http://127.0.0.1:2604/api');

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
