<script setup>
import { ref } from 'vue';

const prompt = ref('');

async function getRandomPrompt() {
  try {
    const response = await fetch('/prompts.json');
    const prompts = await response.json();
    return prompts[Math.floor(Math.random() * prompts.length)];
  } catch (err) {
    console.error('Error loading prompts:', err);
    return "# Imagine a novel musical instrument that combines traditional acoustics with modern technology...";
  }
}

defineExpose({ getRandomPrompt });
</script>
