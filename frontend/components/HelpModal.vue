<template>
  <Transition name="modal">
    <div v-if="modelValue" class="modal-overlay" @click="$emit('update:modelValue', false)">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <h2>Welcome to SOOG</h2>
          <button class="close-button" @click="$emit('update:modelValue', false)">×</button>
        </div>
        <div class="modal-body">
          <h3>The Speculative Organology Organogram Generator</h3>
          <p>
            SOOG helps you visualize musical instruments based on the organogram technique from ethnomusicologist Mantle Hood. 
            It can also help create speculative instruments by mixing, morphing, and entangling geometrical and acoustical information.
          </p>

          <h4>How it works:</h4>
          <ol>
            <li>Abstract instrument shapes using geometrical figures to represent resonant acoustical spaces</li>
            <li>Abstract interfaces with different colored geometrical figures and indicate movements with arrows</li>
            <li>Mix different acoustical shapes to create new polygons or freehand drawings</li>
            <li>View spectrum simulations of acoustical shapes when available</li>
            <li>Represent measurable components with light-blue numbers (e.g., strings, holes, keys)</li>
          </ol>

          <h4>Commands:</h4>
          <ul>
            <li><kbd>Alt</kbd> + <kbd>Enter</kbd>: Evaluate selected text or all text if nothing is selected</li>
            <li>Click the eye icon to show/hide generated code</li>
            <li>On mobile devices, use the "Evaluate" button at the bottom of the screen</li>
          </ul>

          <p class="tip">
            <strong>Tip:</strong> Start by describing an instrument or a combination of instruments you'd like to visualize. 
            SOOG will help you create an organogram representation.
          </p>
        </div>
      </div>
    </div>
  </Transition>
</template>

<script setup>
defineProps({
  modelValue: Boolean
});

defineEmits(['update:modelValue']);
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.85);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 2000;
}

.modal-content {
  background: #1a1a1a;
  border-radius: 8px;
  width: 90%;
  max-width: 600px;
  max-height: 90vh;
  overflow-y: auto;
  color: white;
  font-family: 'IBM Plex Mono', monospace;
}

.modal-header {
  padding: 1rem;
  border-bottom: 1px solid #333;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.modal-body {
  padding: 1.5rem;
}

.close-button {
  background: none;
  border: none;
  color: white;
  font-size: 24px;
  cursor: pointer;
  padding: 0.5rem;
}

.close-button:hover {
  color: #ccc;
}

h2, h3, h4 {
  margin-top: 0;
  color: #fff;
}

h3 {
  font-size: 1.2rem;
  margin-bottom: 1rem;
}

h4 {
  margin-top: 1.5rem;
  color: #4CAF50;
}

p {
  line-height: 1.6;
  margin-bottom: 1rem;
}

ul, ol {
  padding-left: 1.5rem;
  margin-bottom: 1rem;
}

li {
  margin-bottom: 0.5rem;
}

kbd {
  background: #333;
  border-radius: 3px;
  padding: 2px 6px;
  font-size: 0.9em;
}

.tip {
  background: rgba(76, 175, 80, 0.1);
  border-left: 4px solid #4CAF50;
  padding: 1rem;
  margin-top: 1.5rem;
}

/* Modal transition */
.modal-enter-active,
.modal-leave-active {
  transition: opacity 0.3s ease;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}
</style>
