<template>
  <div ref="editor" class="ace-editor"></div>
</template>

<script setup>
import {
  onMounted,
  ref,
  defineExpose,
  defineEmits,
  onUnmounted,
  nextTick,
} from "vue";
import { useRuntimeConfig } from "#app";
import ace from "ace-builds/src-noconflict/ace";
import "ace-builds/src-noconflict/mode-python";
import "ace-builds/src-noconflict/theme-monokai";
import { useRandomPrompt } from "~/composables/useRandomPrompt";

const emit = defineEmits(["evaluate"]);
const editor = ref(null);
const version = ref("0.0.0");
let aceEditorInstance;
const commandHistory = ref([]);
let historyIndex = ref(-1);
let currentCommand = ref("");

const fetchVersion = async () => {
  try {
    const config = useRuntimeConfig();
    const baseURL = config.public.apiBase || "http://localhost:10000";
    const response = await fetch(`${baseURL}/version`, {
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
  } catch (error) {
    console.error("Error fetching version:", error.message);
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
    historyIndex.value = commandHistory.value.length;
  }
};

defineExpose({
  addToEditor,
  aceEditor: () => aceEditorInstance,
  clearEditor,
  addToHistory,
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
        bindKey: { win: "Ctrl-Up", mac: "Command-Up" },
        exec: () => {
          if (historyIndex.value > 0) {
            if (historyIndex.value === commandHistory.value.length) {
              currentCommand.value = aceEditorInstance.getValue();
            }
            historyIndex.value--;
            aceEditorInstance.setValue(
              commandHistory.value[historyIndex.value]
            );
            aceEditorInstance.clearSelection();
          }
        },
      },
      {
        name: "nextCommand",
        bindKey: { win: "Ctrl-Down", mac: "Command-Down" },
        exec: () => {
          if (historyIndex.value < commandHistory.value.length) {
            historyIndex.value++;
            aceEditorInstance.setValue(
              historyIndex.value === commandHistory.value.length
                ? currentCommand.value
                : commandHistory.value[historyIndex.value]
            );
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
  window.removeEventListener("resize", updateFontSize);
  if (versionPollInterval) {
    clearInterval(versionPollInterval);
  }
});
</script>
