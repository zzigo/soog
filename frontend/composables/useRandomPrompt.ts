export function useRandomPrompt() {
  async function getRandomPrompt() {
    try {
      const response = await fetch('/prompts.json');
      const prompts = await response.json();
      return prompts[Math.floor(Math.random() * prompts.length)];
    } catch (err) {
      console.error('Error loading prompts:', err);
      return "Imagine a novel musical instrument that combines traditional acoustics with objects from technic or nature...";
    }
  }

  return {
    getRandomPrompt
  };
}
