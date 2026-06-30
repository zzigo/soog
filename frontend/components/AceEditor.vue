<template>
  <div ref="editor" class="ace-editor"></div>
</template>

<script setup>
import {
  onMounted,
  ref,
  onUnmounted,
  nextTick,
} from "vue";
import { useRuntimeConfig } from "#app";
import ace from "ace-builds/src-noconflict/ace";
import "ace-builds/src-noconflict/mode-python";
import "ace-builds/src-noconflict/theme-monokai";
import { useRandomPrompt } from "~/composables/useRandomPrompt";
import { useApi } from "~/composables/useApi";

const props = defineProps({
  terminalMode: { type: Boolean, default: false }
});

const emit = defineEmits(["evaluate"]);
const editor = ref(null);
const version = ref("0.0.0");
let aceEditorInstance;
const commandHistory = ref([]);
let historyIndex = -1;
let currentCommand = "";
let versionFetchWarned = false;

const { apiBase } = useApi();

const fetchVersion = async () => {
  try {
    const response = await fetch(`${apiBase.value}/version`, {
      headers: {
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    // Check if response has content before parsing JSON
    const contentType = response.headers.get("content-type");
    if (!contentType || !contentType.includes("application/json")) {
      throw new Error("Invalid response type");
    }

    const text = await response.text();
    if (!text || text.trim().length === 0) {
      throw new Error("Empty response");
    }

    const data = JSON.parse(text);
    if (data.version !== version.value) {
      version.value = data.version;
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
    versionFetchWarned = false;
  } catch (error) {
    if (import.meta.dev && !versionFetchWarned) {
      console.warn("Version polling unavailable:", error.message);
      versionFetchWarned = true;
    }
    version.value = "local-dev";
  }
};

let versionPollInterval;
const startVersionPolling = () => {
  versionPollInterval = setInterval(fetchVersion, 30000);
};

const updateFontSize = () => {
  if (!aceEditorInstance) return;
  const isMobile =
    /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
      navigator.userAgent
    );
  aceEditorInstance.setOption("fontSize", isMobile ? "16px" : "20px");
};

const addToEditor = (content, type = "text") => {
  const cssClass =
    type === "text"
      ? "gpt-text-response"
      : type === "plot"
      ? "gpt-plot-response"
      : "gpt-code-response";

  if (!aceEditorInstance) return;

  const session = aceEditorInstance.getSession();
  const doc = session.getDocument();
  const currentLength = doc.getLength();
  const newContent = `\n\n${content}\n\n`;

  doc.insert({ row: currentLength, column: 0 }, newContent);

  const Range =
    ace.require("ace/range").Range || ace.require("ace/edit_session").Range;
  const startRow = currentLength + 1;
  const endRow = currentLength + newContent.split("\n").length - 1;

  aceEditorInstance.session.addMarker(
    new Range(startRow, 0, endRow, 1),
    cssClass,
    "fullLine"
  );

  nextTick(() => {
    aceEditorInstance.scrollToLine(endRow, true, true, () => {});
    aceEditorInstance.gotoLine(endRow, 0, true);
    const editorElement = editor.value;
    if (editorElement) {
      editorElement.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  });
};

const setEditorContent = (content = "") => {
  if (!aceEditorInstance) return;
  aceEditorInstance.session.setValue(content);
  aceEditorInstance.clearSelection();
  aceEditorInstance.gotoLine(1, 0, true);
};

const clearEditor = () => {
  if (aceEditorInstance) {
    aceEditorInstance.session.setValue("");
    aceEditorInstance.clearSelection();
  }
};

const addToHistory = (content) => {
  if (
    content.trim() &&
    commandHistory.value[commandHistory.value.length - 1] !== content
  ) {
    commandHistory.value.push(content);
    historyIndex = commandHistory.value.length;
  }
};

defineExpose({
  addToEditor,
  aceEditor: () => aceEditorInstance,
  clearEditor,
  addToHistory,
  setEditorContent,
});

onMounted(async () => {
  try {
    aceEditorInstance = ace.edit(editor.value);
    aceEditorInstance.setTheme("ace/theme/monokai");
    aceEditorInstance.session.setMode("ace/mode/python");
    aceEditorInstance.setOption("wrap", true);
    aceEditorInstance.setOption("printMargin", false);
    aceEditorInstance.setOption("tabSize", 2);
    aceEditorInstance.setOption("showGutter", false);
    aceEditorInstance.setOption("fontFamily", "'IBM Plex Mono', monospace");

    updateFontSize();
    window.addEventListener("resize", updateFontSize);

    const { getRandomPrompt } = useRandomPrompt();
    const prompt = await getRandomPrompt();

    await fetchVersion();
    startVersionPolling();
    aceEditorInstance.setValue(
      `# Welcome to SOOG [The Speculative Organology Organogram Generator ${version.value}]\n# Write your invented instrument, select text and press Alt+Enter to evaluate\n\n${prompt}\n`
    );
    aceEditorInstance.clearSelection();

    addToHistory(prompt);

    aceEditorInstance.commands.addCommands([
      {
        name: "evaluateCode",
        bindKey: { win: "Alt-Enter", mac: "Alt-Enter" },
        exec: () => {
          const selectedText = aceEditorInstance.getSelectedText();
          const codeToEvaluate = selectedText || aceEditorInstance.getValue();
          addToHistory(codeToEvaluate);
          emit("evaluate", codeToEvaluate);
        },
      },
      {
        name: "clearEditor",
        bindKey: { win: "Ctrl-H", mac: "Command-H" },
        exec: clearEditor,
      },
      {
        name: "previousCommand",
        bindKey: { win: "Ctrl-Up|Up", mac: "Command-Up|Up" },
        exec: (editor) => {
          const cursor = editor.getCursorPosition();
          const isAtTop = cursor.row === 0;
          // Check if the command was triggered by a naked "Up" key or a modified one
          const isNakedKey = !ace.require("ace/lib/event").getModifierString(window.event); 
          
          if (isAtTop && (props.terminalMode || !isNakedKey)) {
            if (historyIndex > 0) {
              if (historyIndex === commandHistory.value.length) {
                currentCommand = editor.getValue();
              }
              historyIndex--;
              editor.setValue(commandHistory.value[historyIndex]);
              editor.clearSelection();
              return true;
            }
          }
          editor.selection.moveCursorUp();
        },
      },
      {
        name: "nextCommand",
        bindKey: { win: "Ctrl-Down|Down", mac: "Command-Down|Down" },
        exec: (editor) => {
          const cursor = editor.getCursorPosition();
          const isAtBottom = cursor.row === editor.session.getLength() - 1;
          const isNakedKey = !ace.require("ace/lib/event").getModifierString(window.event);

          if (isAtBottom && (props.terminalMode || !isNakedKey)) {
            if (historyIndex < commandHistory.value.length) {
              historyIndex++;
              editor.setValue(
                historyIndex === commandHistory.value.length
                  ? currentCommand
                  : commandHistory.value[historyIndex]
              );
              editor.clearSelection();
              return true;
            }
          }
          editor.selection.moveCursorDown();
        },
      },
    ]);
  } catch (error) {
    console.error("AceEditor initialization failed:", error);
  }
});

onUnmounted(() => {
  window.removeEventListener("resize", updateFontSize);
  if (versionPollInterval) {
    clearInterval(versionPollInterval);
  }
});
</script>
