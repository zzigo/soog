<template>
  <div ref="editor" class="ace-editor"></div>
</template>

<script setup>
import { onMounted, ref, defineExpose, defineEmits, onUnmounted, nextTick } from 'vue';
import { useRuntimeConfig } from '#app';
import ace from 'ace-builds/src-noconflict/ace';
import 'ace-builds/src-noconflict/mode-python';
import 'ace-builds/src-noconflict/theme-monokai';
import { useRandomPrompt } from '~/composables/useRandomPrompt';

const emit = defineEmits(['evaluate']);
const editor = ref(null);
const version = ref('0.0.0');
let aceEditorInstance;
const commandHistory = ref([]);
let historyIndex = ref(-1);
let currentCommand = ref('');

const fetchVersion = async () => {
  try {
    const config = useRuntimeConfig();
    const baseURL = config.public.apiBase || 'http://localhost:10000';
    const response = await fetch(`${baseURL}/api/version`);
    const data = await response.json();
    if (data.version !== version.value) {
      version.value = data.version;
      // Update welcome message when version changes
      if (aceEditorInstance) {
        const currentContent = aceEditorInstance.getValue();
        const newContent = currentContent.replace(
          /# Welcome to SOOG \[The Speculative Organology Organogram Generator [0-9.]+\]/,
          `# Welcome to SOOG [The Speculative Organology Organogram Generator ${data.version}]`
        );
        aceEditorInstance.setValue(newContent);
        aceEditorInstance.clearSelection();
      }
    }
  } catch (error) {
    console.error('Error fetching version:', error);
  }
};

// Set up polling for version updates
let versionPollInterval;
const startVersionPolling = () => {
  // Check version every 30 seconds
  versionPollInterval = setInterval(fetchVersion, 30000);
};

// Update font size based on device
const updateFontSize = () => {
  if (!aceEditorInstance) return;
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
  aceEditorInstance.setOption('fontSize', isMobile ? '16px' : '20px');
};

// Add content to Ace Editor dynamically
const addToEditor = (content, type = 'text') => {
  const cssClass = type === 'text' ? 'gpt-text-response' : 
                  type === 'plot' ? 'gpt-plot-response' : 
                  'gpt-code-response';
  if (!aceEditorInstance) return;

  const session = aceEditorInstance.getSession();
  const doc = session.getDocument();
  const currentLength = doc.getLength();
  const newContent = `\n\n${content}\n\n`; // Add extra spacing

  doc.insert({ row: currentLength, column: 0 }, newContent);

  // Highlight GPT responses using markers
  const Range = ace.require('ace/range').Range || ace.require('ace/edit_session').Range;
  const startRow = currentLength + 1;
  const endRow = currentLength + newContent.split('\n').length - 1;

  aceEditorInstance.session.addMarker(
    new Range(startRow, 0, endRow, 1),
    cssClass,
    'fullLine'
  );

  // Scroll to the bottom of the AceEditor and page
  nextTick(() => {
    aceEditorInstance.scrollToLine(endRow, true, true, () => {});
    aceEditorInstance.gotoLine(endRow, 0, true);

    // Ensure the page also scrolls to the bottom
    const editorElement = editor.value;
    if (editorElement) {
      editorElement.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  });
};

// Clear editor content
const clearEditor = () => {
  if (aceEditorInstance) {
    aceEditorInstance.session.setValue("");
    aceEditorInstance.clearSelection();
  }
};

// Add to command history
const addToHistory = (content) => {
  if (content.trim() && commandHistory.value[commandHistory.value.length - 1] !== content) {
    commandHistory.value.push(content);
    historyIndex.value = commandHistory.value.length;
  }
};

// Expose methods for parent components
defineExpose({
  addToEditor,
  aceEditor: () => aceEditorInstance,
  clearEditor,
  addToHistory,
});

onMounted(async () => {
  try {
    aceEditorInstance = ace.edit(editor.value);
    aceEditorInstance.setTheme('ace/theme/monokai');
    aceEditorInstance.session.setMode('ace/mode/python');
    aceEditorInstance.setOption('wrap', true);
    aceEditorInstance.setOption('printMargin', false);
    aceEditorInstance.setOption('tabSize', 2);
    aceEditorInstance.setOption('showGutter', false);

    // Set initial font size and listen for changes
    updateFontSize();
    window.addEventListener('resize', updateFontSize);

    // Get random prompt
    const { getRandomPrompt } = useRandomPrompt();
    const prompt = await getRandomPrompt();

    // Fetch version and start polling
    await fetchVersion();
    startVersionPolling();
    aceEditorInstance.setValue(`# Welcome to SOOG [The Speculative Organology Organogram Generator ${version.value}]\n# Write your invented instrument, select text and press Alt+Enter to evaluate\n\n${prompt}\n`);
    aceEditorInstance.clearSelection();
    
    // Add initial prompt to command history
    addToHistory(prompt);

    // Add custom keybindings
    aceEditorInstance.commands.addCommands([
      {
        name: 'evaluateCode',
        bindKey: { win: 'Alt-Enter', mac: 'Alt-Enter' },
        exec: () => {
          const selectedText = aceEditorInstance.getSelectedText();
          const codeToEvaluate = selectedText || aceEditorInstance.getValue();
          addToHistory(codeToEvaluate);
          emit('evaluate', codeToEvaluate);
        },
      },
      {
        name: 'clearEditor',
        bindKey: { win: 'Ctrl-H', mac: 'Command-H' },
        exec: clearEditor,
      },
      {
        name: 'previousCommand',
        bindKey: { win: 'Ctrl-Up', mac: 'Command-Up' },
        exec: () => {
          if (historyIndex.value > 0) {
            // Save current input if we're starting to navigate history
            if (historyIndex.value === commandHistory.value.length) {
              currentCommand.value = aceEditorInstance.getValue();
            }
            historyIndex.value--;
            aceEditorInstance.setValue(commandHistory.value[historyIndex.value]);
            aceEditorInstance.clearSelection();
          }
        },
      },
      {
        name: 'nextCommand',
        bindKey: { win: 'Ctrl-Down', mac: 'Command-Down' },
        exec: () => {
          if (historyIndex.value < commandHistory.value.length) {
            historyIndex.value++;
            if (historyIndex.value === commandHistory.value.length) {
              aceEditorInstance.setValue(currentCommand.value);
            } else {
              aceEditorInstance.setValue(commandHistory.value[historyIndex.value]);
            }
            aceEditorInstance.clearSelection();
          }
        },
      },
    ]);
  } catch (error) {
    console.error("AceEditor initialization failed:", error);
  }
});

onUnmounted(() => {
  window.removeEventListener('resize', updateFontSize);
  if (versionPollInterval) {
    clearInterval(versionPollInterval);
  }
});
</script>
