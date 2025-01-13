<template>
  <div ref="editor" class="ace-editor"></div>
</template>

<script setup>
import { onMounted, ref, defineExpose, defineEmits, onUnmounted, nextTick } from 'vue';
import ace from 'ace-builds/src-noconflict/ace';
import 'ace-builds/src-noconflict/mode-python';
import 'ace-builds/src-noconflict/theme-monokai';

const emit = defineEmits(['evaluate']);
const editor = ref(null);
let aceEditorInstance;

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

// Expose methods for parent components
defineExpose({
  addToEditor,
  aceEditor: () => aceEditorInstance,
  clearEditor,
});

onMounted(() => {
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

    // Add welcome message
    aceEditorInstance.setValue("# Welcome to SOOG [The Speculative Organology Organogram Generator v0.1]\n# Imagine an instrument, select and press Alt+Enter to evaluate\n\n");
    aceEditorInstance.clearSelection();

    // Add custom keybindings
    aceEditorInstance.commands.addCommands([
      {
        name: 'evaluateCode',
        bindKey: { win: 'Alt-Enter', mac: 'Alt-Enter' },
        exec: () => {
          const selectedText = aceEditorInstance.getSelectedText();
          const codeToEvaluate = selectedText || aceEditorInstance.getValue();
          emit('evaluate', codeToEvaluate);
        },
      },
      {
        name: 'clearEditor',
        bindKey: { win: 'Ctrl-H', mac: 'Command-H' },
        exec: clearEditor,
      },
    ]);
  } catch (error) {
    console.error("AceEditor initialization failed:", error);
  }
});

onUnmounted(() => {
  window.removeEventListener('resize', updateFontSize);
});
</script>

