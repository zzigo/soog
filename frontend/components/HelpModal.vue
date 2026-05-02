<template>
  <Transition name="modal">
    <div v-if="modelValue" class="modal-overlay" @click="$emit('update:modelValue', false)">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <h2>Welcome to SOOG</h2>
          <button class="close-button" @click="$emit('update:modelValue', false)">×</button>
        </div>
        <div class="modal-body">
          <section v-if="featuredSketches.length" class="mosaic-section">
            <div class="mosaic-grid">
              <div 
                v-for="item in featuredSketches" 
                :key="item.basename"
                class="mosaic-item"
                @click="$emit('select-featured', item.basename)"
                @mouseenter="playAudio(item)"
                @mouseleave="stopAudio"
              >
                <img :src="assetHref(item.sketch_url)" :alt="item.title" class="mosaic-img" />
                <div class="mosaic-overlay">
                  <h1 class="mosaic-title">{{ item.title || item.basename }}</h1>
                  <p class="mosaic-prompt">{{ item.prompt }}</p>
                  <div v-if="item.sound_samples?.length" class="sound-indicator">
                    <span class="icon">🔊</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

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

          <h4>Reversioning:</h4>
          <ul>
            <li>Start your prompt with <kbd>*</kbd> or <kbd>+</kbd> to trigger reversion mode.</li>
            <li>You can also use a header like <kbd>[REFACT source=&lt;basename&gt; group=&lt;group_id&gt; title=&lt;name&gt;]</kbd>.</li>
            <li>Use <strong>Gallery → Load Code</strong> to preload previous organogram + geometry, then append your corrections.</li>
            <li>During generation, the processing HUD will show <code>[reversion]</code> when reversion mode is active.</li>
            <li>New iterations are saved in the same grouped name with incremental versions (<code>1, 2, 3...</code>).</li>
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

          <div class="credits" v-if="false">
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
import { ref, onMounted, watch } from 'vue'
import { useRuntimeConfig } from '#app'

const props = defineProps({
  modelValue: Boolean
});

defineEmits(['update:modelValue', 'select-featured']);

const config = useRuntimeConfig()
const apiBase = config.public.apiBase || 'http://localhost:10000/api'

const featuredSketches = ref([])
const activeAudio = ref(null)
const fadeInterval = ref(null)
const currentPlayingBasename = ref(null)

async function loadFeatured() {
  try {
    const res = await fetch(`${apiBase}/gallery/list`)
    const data = await res.json()
    // Filter items that have a sketch_url AND are marked as featured
    featuredSketches.value = (data.items || [])
      .filter(item => item.featured && item.sketch_url)
      // Limit to a reasonable number for the mosaic
      .slice(0, 24)
  } catch (e) {
    console.error('Failed to load featured sketches:', e)
  }
}

function playAudio(item) {
  if (!item.sound_samples || item.sound_samples.length === 0) return
  
  const sample = item.sound_samples[0]
  const url = assetHref(sample.ogg_url || sample.url)
  if (!url) return

  // If already playing this one, do nothing
  if (currentPlayingBasename.value === item.basename) return

  // Stop any existing playback immediately
  clearPlaybackState()

  currentPlayingBasename.value = item.basename
  const audio = new Audio(url)
  audio.volume = 0
  audio.loop = true
  audio.preload = 'auto'
  activeAudio.value = audio
  
  const playPromise = audio.play()
  if (playPromise !== undefined) {
    playPromise.then(() => {
      // Only continue if we are still supposed to be playing this item
      if (currentPlayingBasename.value !== item.basename) {
        audio.pause()
        return
      }

      // Fade in: 300ms
      let vol = 0
      const duration = 300
      const interval = 20
      const steps = duration / interval
      const stepValue = 1 / steps
      
      fadeInterval.value = setInterval(() => {
        vol += stepValue
        if (vol >= 1) {
          audio.volume = 1
          clearInterval(fadeInterval.value)
          fadeInterval.value = null
        } else {
          audio.volume = vol
        }
      }, interval)
    }).catch(e => {
      if (e.name !== 'AbortError') {
        console.warn('Playback failed:', e)
      }
      if (currentPlayingBasename.value === item.basename) {
        currentPlayingBasename.value = null
        activeAudio.value = null
      }
    })
  }
}

function stopAudio() {
  const audio = activeAudio.value
  const basename = currentPlayingBasename.value
  
  if (!audio) {
    currentPlayingBasename.value = null
    return
  }

  // Clear current target so playAudio can trigger again if needed
  currentPlayingBasename.value = null
  
  if (fadeInterval.value) {
    clearInterval(fadeInterval.value)
    fadeInterval.value = null
  }
  
  // Fade out: 300ms
  let vol = audio.volume
  const duration = 300
  const interval = 20
  const steps = duration / interval
  const stepValue = vol / steps

  fadeInterval.value = setInterval(() => {
    vol -= stepValue
    if (vol <= 0) {
      audio.volume = 0
      audio.pause()
      audio.src = '' // Help garbage collection
      audio.load()
      clearInterval(fadeInterval.value)
      fadeInterval.value = null
      if (activeAudio.value === audio) {
        activeAudio.value = null
      }
    } else {
      audio.volume = Math.max(0, vol)
    }
  }, interval)
}

function clearPlaybackState() {
  if (fadeInterval.value) {
    clearInterval(fadeInterval.value)
    fadeInterval.value = null
  }
  if (activeAudio.value) {
    activeAudio.value.pause()
    activeAudio.value.src = ''
    activeAudio.value.load()
    activeAudio.value = null
  }
  currentPlayingBasename.value = null
}

function assetHref(url) {
  if (!url) return ''
  if (url.startsWith('http')) return url
  const offloadApiBase = apiBase.endsWith('/api') ? apiBase.slice(0, -4) : apiBase
  if (url.startsWith('/offload')) return offloadApiBase + url
  if (apiBase.endsWith('/api') && url.startsWith('/api/')) return apiBase + url.substring(4)
  return apiBase + url
}

watch(() => props.modelValue, (val) => {
  if (val) {
    loadFeatured()
  } else {
    // Stop audio when closing modal
    if (activeAudio.value) {
      activeAudio.value.pause()
      activeAudio.value = null
    }
    if (fadeInterval.value) {
      clearInterval(fadeInterval.value)
      fadeInterval.value = null
    }
  }
})

onMounted(() => {
  if (props.modelValue) loadFeatured()
})
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
  max-width: 80%;
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

.mosaic-section {
  margin: -1.5rem -1.5rem 1.5rem -1.5rem;
  border-bottom: 1px solid #333;
}

.mosaic-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(255px, 1fr));
  gap: 0;
  background: #000;
  padding: 0;
}

.mosaic-item {
  aspect-ratio: 1;
  overflow: hidden;
  background: #111;
  border: none;
  transition: all 0.3s ease;
  cursor: pointer;
  position: relative;
}

.mosaic-item:hover {
  transform: scale(1.02);
  z-index: 10;
  box-shadow: 0 0 20px rgba(76, 175, 80, 0.4);
}

.mosaic-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.mosaic-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.75);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  opacity: 0;
  transition: opacity 0.3s ease;
  padding: 15px;
  text-align: center;
  backdrop-filter: blur(2px);
}

.mosaic-item:hover .mosaic-overlay {
  opacity: 1;
}

.mosaic-title {
  font-size: 1.1rem;
  margin: 0 0 8px 0;
  color: #4CAF50;
  font-weight: 700;
  line-height: 1.2;
}

.mosaic-prompt {
  font-size: 0.75rem;
  font-style: italic;
  margin: 0;
  color: rgba(255, 255, 255, 0.8);
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
  line-height: 1.4;
}

.sound-indicator {
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(0, 0, 0, 0.5);
  border-radius: 50%;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.sound-indicator .icon {
  font-size: 14px;
  color: #98ff5f;
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
