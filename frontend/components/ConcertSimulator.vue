<template>
  <div class="concert-container" :class="{ 'fullscreen-cursor': isFullscreen }" @touchstart="handleTouchStart" @touchmove="handleTouchMove" @touchend="handleTouchEnd">
    <div ref="canvasContainer" class="canvas-container"></div>

    <!-- HUD (Right Sidebar) -->
    <Transition name="slide">
      <div v-if="showHUD" class="hud-sidebar">
        <h3 class="hud-title">CONCERT HUD</h3>
        
        <div class="hud-scroll-content">
          <div class="hud-section">
            <label>MASTER VOLUME</label>
            <input type="range" v-model.number="params.masterVolume" min="0" max="2" step="0.1" @input="updateAudio" />
          </div>
          <div class="hud-section">
            <label>REVERB MIX</label>
            <input type="range" v-model.number="params.reverbMix" min="0" max="1" step="0.1" @input="updateAudio" />
          </div>
          <div class="hud-section">
            <label>NEIGHBOR SPACING</label>
            <input type="range" v-model.number="params.neighborSpacing" min="0" max="1000" step="10" />
          </div>
          <div class="hud-section">
            <label>BOUNCE SPEED</label>
            <input type="range" v-model.number="params.bounceSpeed" min="0" max="5" step="0.1" />
          </div>
          <div class="hud-section">
            <label>SPIN SPEED</label>
            <input type="range" v-model.number="params.spinSpeed" min="0" max="5" step="0.1" />
          </div>
          <div class="hud-section">
            <label>LIGHT INTENSITY</label>
            <input type="range" v-model.number="params.lightIntensity" min="0" max="10000" step="100" @input="updateLights" />
          </div>
          <div class="hud-section">
            <label>AUTO ROTATE</label>
            <input type="checkbox" v-model="params.autoRotate" @change="updateControls" />
          </div>

          <div class="hud-section instruments-list">
            <label>ACTIVE INSTRUMENTS</label>
            <div class="instrument-scroller">
              <div 
                v-for="inst in instruments" 
                :key="inst.id" 
                class="instrument-item"
                :class="{ disabled: !inst.enabled }"
              >
                <label class="custom-checkbox">
                  <input type="checkbox" v-model="inst.enabled" @change="toggleInstrument(inst)" />
                  <span class="checkmark"></span>
                </label>
                <span class="inst-name" :title="inst.description">{{ inst.name }}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="hud-footer">
          <p v-if="!isMobileOrTablet">[H] Toggle HUD</p>
          <p v-if="!isMobileOrTablet">[F] Fullscreen</p>
          <p v-if="!isMobileOrTablet">[ESC] Exit Fullscreen</p>
          <NuxtLink to="/" class="exit-btn">EXIT CONCERT</NuxtLink>
        </div>
      </div>
    </Transition>

    <!-- Initial Overlay to Start Audio -->
    <div v-if="!audioStarted" class="audio-overlay" @click="startAudio">
      <div class="start-card">
        <h2>SOOG EXPERIMENTAL CONCERT ROOM</h2>
        <p>Click anywhere to activate spatial audio</p>
        <div class="icon-pulse">
           <svg viewBox="0 0 24 24"><path fill="currentColor" d="M14,3.23V5.29C16.89,6.15 19,8.83 19,12C19,15.17 16.89,17.85 14,18.71V20.77C18,19.86 21,16.28 21,12C21,7.72 18,4.14 14,3.23M16.5,12C16.5,10.23 15.5,8.71 14,7.97V16.04C15.5,15.29 16.5,13.77 16.5,12M3,9V15H7L12,20V4L7,9H3Z"/></svg>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, reactive, watch } from 'vue'
import { useApi } from '~/composables/useApi'

const { apiBase } = useApi()
const canvasContainer = ref(null)
const showHUD = ref(true)
const isFullscreen = ref(false)
const audioStarted = ref(false)
const isMobileOrTablet = ref(false)

// --- Gesture Handling ---
let touchStartX = 0
let touchEndX = 0

function handleTouchStart(e) {
  touchStartX = e.touches[0].clientX
}

function handleTouchMove(e) {
  touchEndX = e.touches[0].clientX
}

function handleTouchEnd() {
  const swipeDistance = touchEndX - touchStartX
  const threshold = 50 // px

  // If swiping from the right margin to left, show HUD
  if (swipeDistance < -threshold && touchStartX > window.innerWidth * 0.8) {
    showHUD.value = true
  }
  // If swiping from HUD to right, hide HUD
  if (swipeDistance > threshold && touchStartX < window.innerWidth && showHUD.value) {
    showHUD.value = false
  }
  
  touchStartX = 0
  touchEndX = 0
}

const params = reactive({
  masterVolume: 1.0,
  reverbMix: 0.4,
  neighborSpacing: 300,
  bounceSpeed: 1.0,
  spinSpeed: 1.0,
  lightIntensity: 5000,
  autoRotate: true
})

let scene, camera, renderer, controls, clock, audioListener, convolver, reverbGain
const instruments = ref([])
let spotLights = []
let animId

async function initThree() {
  const THREE = await import('three')
  const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js')
  
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x020202)
  scene.fog = new THREE.FogExp2(0x020202, 0.001)

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 5000)
  camera.position.set(400, 300, 400)

  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setPixelRatio(window.devicePixelRatio)
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.0
  canvasContainer.value.appendChild(renderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.05
  controls.autoRotate = params.autoRotate
  controls.autoRotateSpeed = 0.5

  clock = new THREE.Clock()
  audioListener = new THREE.AudioListener()
  camera.add(audioListener)

  // --- Reverb Setup ---
  const ctx = audioListener.context
  convolver = ctx.createConvolver()
  reverbGain = ctx.createGain()
  reverbGain.gain.value = params.reverbMix

  // Synthesize Hall Impulse Response
  const length = ctx.sampleRate * 3.5 // 3.5 seconds
  const impulse = ctx.createBuffer(2, length, ctx.sampleRate)
  for (let i = 0; i < 2; i++) {
    const channel = impulse.getChannelData(i)
    for (let j = 0; j < length; j++) {
      channel[j] = (Math.random() * 2 - 1) * Math.pow(1 - j / length, 4.0)
    }
  }
  convolver.buffer = impulse
  
  convolver.connect(audioListener.gain)
  reverbGain.connect(convolver)

  // 20 Spotlights
  const colors = [0xffffff, 0xff0000, 0x00ff00]
  for (let i = 0; i < 20; i++) {
    const color = colors[Math.floor(Math.random() * colors.length)]
    const spot = new THREE.SpotLight(color, params.lightIntensity)
    spot.position.set((Math.random() - 0.5) * 1000, 800, (Math.random() - 0.5) * 1000)
    spot.angle = Math.PI / 8
    spot.penumbra = 0.5
    spot.decay = 1.5
    spot.distance = 2000
    scene.add(spot)
    spotLights.push(spot)
  }

  const ambient = new THREE.AmbientLight(0xffffff, 0.1)
  scene.add(ambient)

  checkDevice()
  loadInstruments()
  animate()
}

function checkDevice() {
  isMobileOrTablet.value = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
}

async function loadInstruments() {
  const { STLLoader } = await import('three/examples/jsm/loaders/STLLoader.js')
  const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js')
  const THREE = await import('three')
  
  const stlLoader = new STLLoader()
  const glbLoader = new GLTFLoader()
  const audioLoader = new THREE.AudioLoader()

  try {
    const res = await fetch(`${apiBase.value}/gallery/list`)
    const data = await res.json()
    const candidates = (data.items || []).filter(item => 
      (item.stl_url || (item.item && item.item.stl_url)) && 
      (item.sound_samples && item.sound_samples.length > 0)
    )

    for (const item of candidates) {
      const url = item.stl_url || item.item.stl_url
      const isGlb = url.toLowerCase().endsWith('.glb')
      const loader = isGlb ? glbLoader : stlLoader

      loader.load(assetHref(url), (result) => {
        let modelMesh
        if (isGlb) {
          modelMesh = result.scene
          result.scene.traverse(child => {
            if (child.isMesh && child.geometry.attributes.color) {
              child.material.vertexColors = true
            }
          })
        } else {
          const material = new THREE.MeshStandardMaterial({ color: 0x66ccff, metalness: 0.2, roughness: 0.5 })
          modelMesh = new THREE.Mesh(result, material)
        }

        modelMesh.position.set((Math.random() - 0.5) * 600, Math.random() * 200 + 100, (Math.random() - 0.5) * 600)
        
        const box = new THREE.Box3().setFromObject(modelMesh)
        const size = box.getSize(new THREE.Vector3()).length()
        const scale = 80 / size
        modelMesh.scale.set(scale, scale, scale)

        const sound = new THREE.PositionalAudio(audioListener)
        sound.getOutput().connect(reverbGain)
        modelMesh.add(sound)

        const samples = item.sound_samples || []
        const soundUrls = samples.map(s => assetHref(s.url))
        
        const instData = reactive({
          id: item.basename,
          name: (item.basename.split('_').slice(1).join(' ') || item.basename).toUpperCase(),
          description: item.answer || item.summary || item.prompt || '',
          enabled: true,
          mesh: modelMesh,
          sound: sound,
          soundUrls: soundUrls,
          offset: Math.random() * Math.PI * 2,
          bounceAmp: Math.random() * 50 + 20,
          rotationAxis: new THREE.Vector3(Math.random(), Math.random(), Math.random()).normalize(),
          velocity: new THREE.Vector3(0, 0, 0)
        })
        
        instruments.value.push(instData)
        scene.add(modelMesh)

        if (audioStarted.value) {
          setTimeout(() => startInstrumentAudio(instData), Math.random() * 5000)
        }
      })
    }
  } catch (e) {
    console.error('Failed to load instruments for concert:', e)
  }
}

async function startInstrumentAudio(inst) {
  if (!audioStarted.value || !inst.enabled) return
  
  const THREE = await import('three')
  const audioLoader = new THREE.AudioLoader()
  
  const playNext = () => {
    if (!inst.enabled || !audioStarted.value) return
    const url = inst.soundUrls[Math.floor(Math.random() * inst.soundUrls.length)]
    
    audioLoader.load(url, (buffer) => {
      if (!inst.enabled || !audioStarted.value) return
      if (inst.sound.isPlaying) inst.sound.stop()
      inst.sound.setBuffer(buffer)
      inst.sound.setRefDistance(100)
      inst.sound.setVolume(params.masterVolume)
      inst.sound.play()
      
      inst.sound.source.onended = () => {
        const silence = Math.random() * 5000 + 1000
        setTimeout(playNext, silence)
      }
    })
  }
  playNext()
}

function toggleInstrument(inst) {
  if (inst.enabled) {
    scene.add(inst.mesh)
    if (audioStarted.value) startInstrumentAudio(inst)
  } else {
    scene.remove(inst.mesh)
    if (inst.sound.isPlaying) inst.sound.stop()
  }
}

function assetHref(url) {
  if (!url) return ''
  if (url.startsWith('http')) return url
  const base = apiBase.value
  const offloadApiBase = base.endsWith('/api') ? base.slice(0, -4) : base
  if (url.startsWith('/offload')) return offloadApiBase + url
  if (base.endsWith('/api') && url.startsWith('/api/')) return base + url.substring(4)
  return base + url
}

function startAudio() {
  audioStarted.value = true
  instruments.value.forEach(inst => startInstrumentAudio(inst))
}

function updateAudio() {
  if (reverbGain) reverbGain.gain.value = params.reverbMix
  instruments.value.forEach(inst => {
    if (inst.sound) inst.sound.setVolume(params.masterVolume)
  })
}

function updateLights() {
  spotLights.forEach(spot => {
    spot.intensity = params.lightIntensity
  })
}

function updateControls() {
  if (controls) controls.autoRotate = params.autoRotate
}

function animate() {
  animId = requestAnimationFrame(animate)
  const dt = clock.getElapsedTime()
  
  const activeInstruments = instruments.value.filter(i => i.enabled)
  activeInstruments.forEach(inst => {
    const centerForce = inst.mesh.position.clone().multiplyScalar(-0.0005)
    inst.velocity.add(centerForce)

    activeInstruments.forEach(other => {
      if (inst === other) return
      const diff = inst.mesh.position.clone().sub(other.mesh.position)
      const dist = diff.length()
      const threshold = params.neighborSpacing
      if (dist < threshold && dist > 0) {
        const repulsionStrength = Math.pow(1 - dist / threshold, 2) * 0.05
        inst.velocity.add(diff.normalize().multiplyScalar(repulsionStrength))
      }
    })

    inst.mesh.position.x += inst.velocity.x
    inst.mesh.position.z += inst.velocity.z
    inst.velocity.multiplyScalar(0.95)
    inst.mesh.position.y = 150 + Math.sin(dt * params.bounceSpeed + inst.offset) * inst.bounceAmp
    inst.mesh.rotateOnAxis(inst.rotationAxis, 0.01 * params.spinSpeed)
  })

  if (controls) controls.update()
  if (renderer && scene && camera) renderer.render(scene, camera)
}

function handleResize() {
  if (!camera || !renderer) return
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
}

function handleKeyDown(e) {
  if (e.key.toLowerCase() === 'h') showHUD.value = !showHUD.value
  if (e.key.toLowerCase() === 'f') toggleFullscreen()
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    const doc = document.documentElement
    if (doc.requestFullscreen) doc.requestFullscreen()
    else if (doc.webkitRequestFullscreen) doc.webkitRequestFullscreen()
    else if (doc.msRequestFullscreen) doc.msRequestFullscreen()
    isFullscreen.value = true
  } else {
    if (document.exitFullscreen) document.exitFullscreen()
    isFullscreen.value = false
  }
}

onMounted(() => {
  initThree()
  window.addEventListener('resize', handleResize)
  window.addEventListener('keydown', handleKeyDown)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  window.removeEventListener('keydown', handleKeyDown)
  if (animId) cancelAnimationFrame(animId)
  instruments.value.forEach(inst => {
    if (inst.sound.isPlaying) inst.sound.stop()
  })
  if (renderer) {
    renderer.dispose()
    renderer.domElement.remove()
  }
})
</script>

<style scoped>
.concert-container {
  width: 100vw;
  height: 100vh;
  position: relative;
  overflow: hidden;
  background: #000;
}

.canvas-container {
  width: 100%;
  height: 100%;
}

.fullscreen-cursor {
  cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' style='fill:none;stroke:white;stroke-width:2;'><path d='M10 2H2v8m0 12v8h8m12 0h8v-8m0-12V2h-8'/></svg>") 16 16, crosshair;
}

/* HUD Styling */
.hud-sidebar {
  position: absolute;
  top: 0;
  right: 0;
  width: 224px;
  height: 100%;
  background: rgba(0, 0, 0, 0.85);
  border-left: 1px solid #333;
  padding: 24px;
  color: #fff;
  z-index: 100;
  backdrop-filter: blur(10px);
  display: flex;
  flex-direction: column;
}

.hud-title {
  font-size: 11px;
  letter-spacing: 2.4px;
  margin-bottom: 32px;
  color: #00ff00;
  text-align: center;
}

.hud-scroll-content {
  flex: 1;
  overflow-y: auto;
  margin-bottom: 20px;
}

.hud-scroll-content::-webkit-scrollbar {
  width: 3px;
}

.hud-scroll-content::-webkit-scrollbar-thumb {
  background: #222;
  border-radius: 2px;
}

.hud-section {
  margin-bottom: 20px;
}

.hud-section label {
  display: block;
  font-size: 8px;
  letter-spacing: 0.8px;
  margin-bottom: 8px;
  color: #888;
}

.hud-section input[type='range'] {
  width: 100%;
  accent-color: #00ff00;
}

.hud-footer {
  font-size: 9px;
  color: #555;
  line-height: 1.6;
  border-top: 1px solid #222;
  padding-top: 15px;
}

.exit-btn {
  display: block;
  margin-top: 16px;
  padding: 10px;
  background: transparent;
  border: 1px solid #444;
  color: #fff;
  text-align: center;
  text-decoration: none;
  font-size: 9px;
  letter-spacing: 1px;
  transition: all 0.3s;
}

.exit-btn:hover {
  background: #fff;
  color: #000;
}

.instruments-list {
  display: flex;
  flex-direction: column;
}

.instrument-scroller {
  padding-right: 5px;
}

.instrument-item {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  transition: opacity 0.3s;
}

.instrument-item.disabled {
  opacity: 0.4;
}

.inst-name {
  font-size: 8px;
  letter-spacing: 0.8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  cursor: help;
  color: #ccc;
}

.inst-name:hover {
  color: #00ff00;
}

/* Custom Checkbox */
.custom-checkbox {
  display: block;
  position: relative;
  width: 11px;
  height: 11px;
  cursor: pointer;
}

.custom-checkbox input {
  position: absolute;
  opacity: 0;
  cursor: pointer;
  height: 0;
  width: 0;
}

.checkmark {
  position: absolute;
  top: 0;
  left: 0;
  height: 11px;
  width: 11px;
  background-color: transparent;
  border: 1px solid #444;
  border-radius: 2px;
}

.custom-checkbox:hover input ~ .checkmark {
  border-color: #666;
}

.custom-checkbox input:checked ~ .checkmark {
  background-color: #00ff00;
  border-color: #00ff00;
}

.checkmark:after {
  content: '';
  position: absolute;
  display: none;
}

.custom-checkbox input:checked ~ .checkmark:after {
  display: block;
}

.custom-checkbox .checkmark:after {
  left: 3px;
  top: 1px;
  width: 3px;
  height: 6px;
  border: solid black;
  border-width: 0 2px 2px 0;
  transform: rotate(45deg);
}

@media (max-width: 768px) {
  .hud-sidebar {
    width: 180px;
    padding: 15px;
  }
  .start-card h2 {
    font-size: 16px;
    letter-spacing: 4px;
  }
}

/* Audio Overlay */
.audio-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 200;
  cursor: pointer;
}

.start-card {
  text-align: center;
}

.start-card h2 {
  letter-spacing: 10px;
  font-size: 20px;
  margin-bottom: 10px;
  color: #fff;
}

.start-card p {
  color: #888;
  font-size: 10px;
  letter-spacing: 2px;
}

.icon-pulse {
  margin-top: 40px;
  width: 50px;
  height: 50px;
  display: inline-block;
  animation: pulse 2s infinite;
  color: #00ff00;
}

@keyframes pulse {
  0% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.2); opacity: 0.5; }
  100% { transform: scale(1); opacity: 1; }
}

.slide-enter-active, .slide-leave-active {
  transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
.slide-enter-from, .slide-leave-to {
  transform: translateX(100%);
}
</style>
