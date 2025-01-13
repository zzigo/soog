<script setup>
    import { ref, onMounted } from 'vue';

export function useRandomPrompt() {
  const prompt = ref('');

  const loadRandomPrompt = async () => {
    try {
      const response = await fetch('/prompts.json'); // Fetch from public folder
      const prompts = await response.json();
      prompt.value = prompts[Math.floor(Math.random() * prompts.length)];
    } catch (err) {
      console.error('Error loading prompts:', err);
      prompt.value = "# Imagine a novel musical instrument that combines traditional acoustics with objects from technics or from nature...";
    }
  };

  onMounted(loadRandomPrompt);

  return { prompt };
}

</script>
