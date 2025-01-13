<template>
  <div ref="editor" class="ace-editor"></div>
</template>

<script setup>
import { onMounted, ref, defineExpose, defineEmits } from 'vue';
import ace from 'ace-builds/src-noconflict/ace';
import 'ace-builds/src-noconflict/mode-python';
import 'ace-builds/src-noconflict/theme-monokai';

const emit = defineEmits(['evaluate']);

const editor = ref(null);
let aceEditorInstance;

onMounted(() => {
  aceEditorInstance = ace.edit(editor.value);
  aceEditorInstance.setTheme('ace/theme/monokai');
  aceEditorInstance.session.setMode('ace/mode/python');
  aceEditorInstance.setOption('wrap', true);
  aceEditorInstance.setOption('printMargin', false);
  aceEditorInstance.setOption('fontSize', '20px');
  aceEditorInstance.setOption('tabSize', 2);
  
  // Add welcome message
  aceEditorInstance.setValue("# Welcome to SOOG [The Speculative Organology Organogram Generator v0.1]\n# Imagine an instrument, select and press Alt+Enter to evaluate\n\n");
  aceEditorInstance.clearSelection();

  // Add custom keybinding for Alt+Enter to evaluate selected code
  aceEditorInstance.commands.addCommand({
    name: 'evaluateCode',
    bindKey: { win: 'Alt-Enter', mac: 'Alt-Enter' },
    exec: () => {
      const selectedText = aceEditorInstance.getSelectedText();
      const codeToEvaluate = selectedText || aceEditorInstance.getValue();
      emit('evaluate', codeToEvaluate);
    },
  });
});

// Add content to Ace Editor dynamically
const addToEditor = (content, type = 'text') => {
  const cssClass = type === 'text' ? 'gpt-text-response' : 
                  type === 'plot' ? 'gpt-plot-response' : 
                  'gpt-code-response';
  if (!aceEditorInstance) return;

  const session = aceEditorInstance.getSession();
  const doc = session.getDocument();
  const currentLength = doc.getLength();
  // Add extra line before the content
  const newContent = `\n\n${content}\n`;

  doc.insert({ row: currentLength, column: 0 }, newContent);

  // Scroll to the newly added content
  aceEditorInstance.scrollToLine(currentLength + newContent.split('\n').length - 1, true, true, () => {});
  aceEditorInstance.gotoLine(currentLength + newContent.split('\n').length, 0, true);

  // Highlight GPT responses using markers
  const Range = ace.require('ace/range').Range;
  const startRow = currentLength + 1; // Skip the extra line we added
  const endRow = currentLength + newContent.split('\n').length - 1;

  aceEditorInstance.session.addMarker(
    new Range(startRow, 0, endRow, 1),
    cssClass,
    'fullLine'
  );

  // Apply custom styling for text responses
  if (type === 'text') {
    const textRange = new Range(startRow, 0, endRow, 1);
    session.addGutterDecoration(startRow, 'gpt-text-line');
    aceEditorInstance.renderer.setStyle('gpt-text-style');
    aceEditorInstance.renderer.updateText();
  }
};

// Expose methods for parent components to call
defineExpose({
  addToEditor,
});
</script>
