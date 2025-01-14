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

          <h4>Organogram Basics:</h4>
          <ul>
            <li><strong>Instrument Types:</strong>
              <ul>
                <li>Idiophones: squares</li>
                <li>Membranophones: horizontal rectangles</li>
                <li>Chordophones: vertical rectangles</li>
                <li>Aerophones: circles</li>
                <li>Electronophones: rhombus</li>
              </ul>
            </li>
            <li><strong>Special Markings:</strong>
              <ul>
                <li>Genus: marked with semi-circle</li>
                <li>Performer position: little white circle with dotted line</li>
                <li>Electronic components:
                  <ul>
                    <li>Microphones: small rhombus</li>
                    <li>Speakers: small horizontal cone (rotated to focus on sweet spot)</li>
                  </ul>
                </li>
                <li>Arrows: Used for connections/relationships (with proportional heads)</li>
                <li>Aerophone tubes: Parallel lines (straight) or conical lines (conical)</li>
              </ul>
            </li>
            <li><strong>Materials (Colors):</strong>
              <ul>
                <li>Wood: orange</li>
                <li>Bamboo: yellow</li>
                <li>Skin: pink</li>
                <li>Glass: green</li>
                <li>Stone: white</li>
                <li>Water: blue</li>
                <li>Gourd: beige</li>
                <li>Earth: brown</li>
                <li>Plastic: grey</li>
                <li>Bone: light grey</li>
              </ul>
            </li>
            <li><strong>Symbols (Orange):</strong> H=hammer, Y=lacing, P=precise, R=relative, C=cord/string, Ri=ring, M=male, F=female</li>
          </ul>

          <h4>Commands:</h4>
          <ul>
            <li><kbd>Alt</kbd> + <kbd>Enter</kbd>: Evaluate selected text or all text if nothing is selected</li>
            <li><kbd>Ctrl</kbd> + <kbd>H</kbd>: Clear editor content</li>
            <li><kbd>Ctrl</kbd> + <kbd>↑</kbd>/<kbd>↓</kbd> (<kbd>⌘</kbd> + <kbd>↑</kbd>/<kbd>↓</kbd> on Mac): Navigate command history</li>
            <li>Click the eye icon to show/hide generated code</li>
            <li>Click the trash icon to clear editor content</li>
            <li>On mobile devices, use the "Evaluate" button at the bottom of the screen</li>
          </ul>

          <p class="tip">
            <strong>Tip:</strong> Start by describing an instrument or a combination of instruments you'd like to visualize. 
            SOOG will help you create an organogram representation.
          </p>

          <div class="reference">
            <p>
              The organogram methodology implemented in SOOG represents an extension of the original visualization technique developed by ethnomusicologist Mantle Hood. For comprehensive information about the foundational organogram system, please refer to:
            </p>
            <p class="citation">
              Hood, Mantle (1982). <em>The ethnomusicologist</em> (2nd ed.). Kent State University Press.
            </p>
          </div>

          <div class="credits">
            <h4>Academic Attribution</h4>
            <p>
              SOOG is a research project developed by Luciano Azzigotti in conjunction with the doctoral dissertation "<em>Speculative Organology</em>" within the Specialized Master and PhD in Music Performance Research programme at the Hochschule der Künste Bern.
            </p>
            <div class="supervisors">
              <p>Under the supervision of:</p>
              <ul>
                <li>Artistic Supervisor: Irene Galindo Quero</li>
                <li>Scientific Supervisor: Prof. Dr. Michael Harenberg</li>
              </ul>
            </div>
            <div class="institution">
              <img src="/hkb.svg" alt="Hochschule der Künste Bern" class="hkb-logo">
            </div>
          </div>
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

.reference {
  margin-top: 2rem;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
}

.citation {
  font-style: normal;
  padding-left: 2rem;
  text-indent: -2rem;
  color: #ccc;
}

.citation em {
  font-style: italic;
}

.credits {
  margin-top: 2rem;
  padding: 1.5rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
}

.supervisors {
  margin-top: 1rem;
}

.supervisors ul {
  list-style: none;
  padding-left: 1rem;
}

.supervisors li {
  color: #ccc;
}

.institution {
  margin-top: 1.5rem;
  text-align: center;
}

.hkb-logo {
  height: 60px;
  margin: 1rem 0;
  filter: brightness(0) invert(1);
}
</style>
