<template>
  <div 
    ref="worldContainer" 
    class="sorganoid-world"
    :class="{ 
      'is-active': active,
      'is-interacting': !uiVisible
    }"
  >
    <!-- Visual HUD for Sorganoid stats (only visible in Interacting mode) -->
    <div v-if="active && !uiVisible" class="world-hud">
      <div class="stat">WORLD: {{ worldName.toUpperCase() }}</div>
      <div class="stat">GEN: {{ generation }}</div>
      <div class="stat">POP: {{ entities.length }}</div>
      <div class="stat">AVG NRG: {{ avgEnergy }}%</div>
      <div class="stat">BEST FIT: {{ bestFitness }}</div>
      <div class="stat">GOAL: {{ optimizationGoal.toUpperCase() }}</div>
      <div v-if="hatching" class="status-msg">METAMORPHOSIS IN PROGRESS...</div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, reactive } from 'vue'
import * as THREE from 'three'
import * as CANNON from 'cannon-es'

const props = defineProps({
  command: { type: String, default: '' },
  active: { type: Boolean, default: false },
  result: { type: Object, default: null },
  loading: { type: Boolean, default: false },
  apiBase: { type: String, default: '' },
  uiVisible: { type: Boolean, default: true }
})

const emit = defineEmits(['print-log', 'exit'])

// --- REACTIVE STATE ---
const worldContainer = ref(null)
const hatching = ref(false)
const generation = ref(1)
const bestFitness = ref(0)
const avgEnergy = ref(100)
const optimizationGoal = ref('speed')
const worldName = ref('void')

// --- NON-REACTIVE ENGINE VARIABLES ---
let scene, camera, renderer, controls, physicsWorld, audioListener, convolver, reverbGain, masterGain
let entities = []
let hatchingEgg = null
let galleryInstruments = []
let lastTime = performance.now()
let animId

const params = reactive({
  masterVolume: 1.0,
  reverbMix: 0.4,
  neighborSpacing: 300,
  bounceSpeed: 1.0,
  spinSpeed: 1.0,
  lightIntensity: 5000,
  autoRotate: true,
  friction: 0.1
})

// --- INITIALIZATION ---

async function init() {
  if (!worldContainer.value) return
  console.log('🌌 Sorganoid World Initializing...')

  const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js')

  // 1. Three.js Scene Setup
  scene = new THREE.Scene()
  scene.background = null 

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 5000)
  camera.position.set(400, 400, 400)
  camera.lookAt(0, 50, 0)

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setPixelRatio(window.devicePixelRatio)
  renderer.setClearColor(0x000000, 0) 
  worldContainer.value.appendChild(renderer.domElement)

  // 2. Audio Layer (Spatial Audio)
  audioListener = new THREE.AudioListener()
  camera.add(audioListener)
  const ctx = audioListener.context

  convolver = ctx.createConvolver()
  const impulseLength = ctx.sampleRate * 2.5
  const impulse = ctx.createBuffer(2, impulseLength, ctx.sampleRate)
  for (let i = 0; i < 2; i++) {
    const channel = impulse.getChannelData(i)
    for (let j = 0; j < impulseLength; j++) {
      channel[j] = (Math.random() * 2 - 1) * Math.pow(1 - j / impulseLength, 4.0)
    }
  }
  convolver.buffer = impulse

  reverbGain = ctx.createGain()
  reverbGain.gain.value = params.reverbMix
  reverbGain.connect(convolver)
  convolver.connect(audioListener.gain)

  // 3. Controls
  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.05

  // 4. Physics Engine
  physicsWorld = new CANNON.World({
    allowSleep: true,
    gravity: new CANNON.Vec3(0, -20.0, 0)
  })

  // Ground Plane
  const grid = new THREE.GridHelper(2000, 40, 0x00ff00, 0x222222)
  scene.add(grid)

  const groundBody = new CANNON.Body({
    type: CANNON.Body.STATIC,
    shape: new CANNON.Plane()
  })
  groundBody.quaternion.setFromEuler(-Math.PI / 2, 0, 0)
  physicsWorld.addBody(groundBody)

  // 5. Lighting
  const light = new THREE.DirectionalLight(0x00ff00, 2.0)
  light.position.set(100, 500, 100)
  scene.add(light)
  scene.add(new THREE.AmbientLight(0xffffff, 0.2))

  // 6. Egg
  const eggGeo = new THREE.SphereGeometry(20, 32, 32)
  eggGeo.scale(0.8, 1.2, 0.8)
  const eggMat = new THREE.MeshStandardMaterial({ 
    color: 0x00ff00, emissive: 0x00ff00, emissiveIntensity: 0.5,
    wireframe: true, transparent: true, opacity: 0
  })
  hatchingEgg = new THREE.Mesh(eggGeo, eggMat)
  hatchingEgg.position.y = 30
  scene.add(hatchingEgg)

  animate()
  console.log('✅ Sorganoid World Ready.')
}

// --- LOGGING & COMMANDS ---

function log(msg, type = 'info') {
  console.log(`[Sorganoid] ${msg}`)
  emit('print-log', { text: msg, type })
}

async function listWorlds() {
  try {
    const res = await fetch(`${props.apiBase}/sorganoids/worlds`)
    const data = await res.json()
    if (data.ok) log(`Worlds: ${data.worlds.join(', ')}`)
  } catch (e) { log(`Error listing worlds: ${e.message}`, 'error') }
}

async function listInstruments(silent = false) {
  try {
    const res = await fetch(`${props.apiBase}/gallery/list`)
    const data = await res.json()
    const candidates = (data.items || []).filter(item => 
      (item.stl_url || (item.item && item.item.stl_url)) && 
      (item.sound_samples && item.sound_samples.length > 0)
    )
    galleryInstruments = candidates
    const names = candidates.map(item => (item.basename.split('_').slice(1).join('-') || item.basename).replace(/[^a-zA-Z0-9-]/g, '').toLowerCase())
    log(`Instruments: ${names.join(', ')}`)
  } catch (e) { log(`Error listing instruments: ${e.message}`, 'error') }
}

async function loadInstrument(shortname, position = null) {
  if (!galleryInstruments.length) await listInstruments(true)
  const item = galleryInstruments.find(inst => 
    (inst.basename.split('_').slice(1).join('-') || inst.basename)
      .replace(/[^a-zA-Z0-9-]/g, '').toLowerCase() === shortname.toLowerCase()
  )
  if (item) {
    const url = item.stl_url || item.item?.stl_url
    if (url) {
      const soundUrls = (item.sound_samples || []).map(s => assetHref(s.url))
      await loadNeuralBeing(url, item.basename, soundUrls, position)
    }
  } else {
    log(`Instrument '${shortname}' not found.`, 'error')
  }
}

// --- AUDIO & SONIFICATION ---

function playCollisionSound(impactVelocity, mass, position, soundUrls = []) {
  if (!audioListener || audioListener.context.state !== 'running') return
  const ctx = audioListener.context
  const now = ctx.currentTime

  if (soundUrls && soundUrls.length > 0) {
    const url = soundUrls[Math.floor(Math.random() * soundUrls.length)]
    const audioLoader = new THREE.AudioLoader()
    audioLoader.load(url, (buffer) => {
      const pSound = new THREE.PositionalAudio(audioListener)
      pSound.setBuffer(buffer)
      pSound.setRefDistance(200)
      pSound.setVolume(Math.min(1.0, impactVelocity / 15))
      pSound.getOutput().connect(reverbGain)
      pSound.getOutput().connect(audioListener.gain)
      
      const source = new THREE.Object3D()
      source.position.copy(position)
      scene.add(source)
      source.add(pSound)
      pSound.play()
      setTimeout(() => { 
        scene.remove(source)
        pSound.disconnect() 
      }, buffer.duration * 1000 + 100)
    })
    return
  }

  // Fallback synthetic grain
  const pSound = new THREE.PositionalAudio(audioListener)
  const osc = ctx.createOscillator()
  const gain = ctx.createGain()
  const freq = (80 / (mass || 1)) + (impactVelocity * 15)
  osc.type = impactVelocity > 12 ? 'square' : 'sine'
  osc.frequency.setValueAtTime(freq, now)
  osc.frequency.exponentialRampToValueAtTime(freq * 0.01, now + 0.4)
  const vol = Math.min(0.25, (impactVelocity / 25) * 0.4)
  gain.gain.setValueAtTime(0, now)
  gain.gain.linearRampToValueAtTime(vol, now + 0.005)
  gain.gain.exponentialRampToValueAtTime(0.001, now + 0.4)
  osc.connect(gain)
  pSound.setNodeSource(gain)
  pSound.setRefDistance(100)
  pSound.getOutput().connect(reverbGain)
  pSound.getOutput().connect(audioListener.gain)
  const source = new THREE.Object3D()
  source.position.copy(position)
  scene.add(source)
  source.add(pSound)
  osc.start()
  osc.stop(now + 0.5)
  setTimeout(() => scene.remove(source), 600)
}

// --- PHYSICS & ENTITIES ---


function createEntity(mesh, body, genome = {}) {
  const soundNode = new THREE.PositionalAudio(audioListener)
  mesh.add(soundNode)
  
  const entity = reactive({
    id: Math.random().toString(36).substr(2, 9),
    mesh, body, soundNode,
    genome: {
      activity: 0.5, jumpForce: 15, impulseRate: 4000, 
      metabolism: 0.08, color: 0x00ff00, soundUrls: [],
      w: 10, h: 10, d: 10, ...genome
    },
    energy: 1.0, isAlive: true, lastAction: performance.now(),
    nextActionDelay: 2000 + Math.random() * 5000,
    startPos: mesh.position.clone(), createdAt: performance.now(),
    energySpent: 0, distanceTraveled: 0, fitness: 0
  })

  body.addEventListener('collide', (event) => {
    if (!entity.isAlive) return
    const impact = event.contact.getImpactVelocityAlongNormal()
    if (impact > 1.5) {
      playCollisionSound(impact, body.mass, mesh.position, entity.genome.soundUrls, soundNode)
      const drain = (impact / 100) * entity.genome.metabolism
      entity.energy -= drain
      entity.energySpent += drain
      mesh.traverse(c => {
        if (c.isMesh && c.material && c.material.emissive) c.material.emissiveIntensity = Math.min(2, impact / 5)
      })

      const otherBody = event.body
      const other = entities.find(e => e.body === otherBody)
      if (other && other.isAlive && impact > 10) {
        const temp = entity.genome.activity
        entity.genome.activity = THREE.MathUtils.lerp(entity.genome.activity, other.genome.activity, 0.1)
        other.genome.activity = THREE.MathUtils.lerp(other.genome.activity, temp, 0.1)
        log('🧬 Gene Traits Exchanged')
      }
    }
  })
  entities.push(entity)
}


function spawnRectangulata(params = {}) {
  if (!scene || !physicsWorld) return
  const count = params.count || 1
  const gO = params.genome || {}
  
  for (let i = 0; i < count; i++) {
    const w = gO.w || 5 + Math.random() * 15
    const h = gO.h || 5 + Math.random() * 15
    const d = gO.d || 5 + Math.random() * 15
    const color = gO.color || 0x00ff00
    
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(w, h, d),
      new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0, wireframe: true, transparent: true, opacity: 0.9 })
    )
    mesh.position.set((Math.random() - 0.5) * 150, 100 + Math.random() * 200, (Math.random() - 0.5) * 150)
    scene.add(mesh)

    const body = new CANNON.Body({
      mass: (w * h * d) / 1000,
      shape: new CANNON.Box(new CANNON.Vec3(w/2, h/2, d/2)),
      position: new CANNON.Vec3(mesh.position.x, mesh.position.y, mesh.position.z),
      linearDamping: 0.1, angularDamping: 0.2
    })
    physicsWorld.addBody(body)
    createEntity(mesh, body, { ...gO, w, h, d, color })
  }
}

async function loadNeuralBeing(url, basename, soundUrls = [], position = null) {
  const isGlb = url.toLowerCase().endsWith('.glb')
  const { GLTFLoader } = isGlb ? await import('three/examples/jsm/loaders/GLTFLoader.js') : { GLTFLoader: null }
  const { STLLoader } = !isGlb ? await import('three/examples/jsm/loaders/STLLoader.js') : { STLLoader: null }
  const loader = isGlb ? new GLTFLoader() : new STLLoader()
  
  loader.load(assetHref(url), (result) => {
    let mesh
    if (isGlb) {
      mesh = result.scene
      mesh.traverse(c => { 
        if (c.isMesh) { 
          if (c.geometry.attributes.color) c.material.vertexColors = true
          c.material.emissive = new THREE.Color(0x00ff00); c.material.emissiveIntensity = 0 
        } 
      })
    } else {
      mesh = new THREE.Mesh(result, new THREE.MeshStandardMaterial({ color: 0x66ccff, emissive: 0x66ccff, wireframe: true, transparent: true, opacity: 0.9 }))
    }

    
    const pos = position || { x: (Math.random() - 0.5) * 150, y: 250, z: (Math.random() - 0.5) * 150 }
    mesh.position.set(pos.x, pos.y, pos.z)
    
    const box = new THREE.Box3().setFromObject(mesh)
    const size = new THREE.Vector3()
    box.getSize(size)
    const totalSize = size.length()
    const scale = 80 / (totalSize || 1)
    mesh.scale.set(scale, scale, scale)
    scene.add(mesh)

    // Re-calculate size after scaling for accurate physics box
    const scaledSize = size.multiplyScalar(scale)
    const shape = new CANNON.Box(new CANNON.Vec3(scaledSize.x / 2, scaledSize.y / 2, scaledSize.z / 2))
    
    const body = new CANNON.Body({
      mass: 5, 
      shape: shape, 
      position: new CANNON.Vec3(pos.x, pos.y, pos.z),
      linearDamping: 0.1, 
      angularDamping: 0.2
    })
    physicsWorld.addBody(body)

    createEntity(mesh, body, { species: basename, soundUrls, color: 0x66ccff })
    hatching.value = false
  })
}

// --- EVOLUTION ---

function breedBeings(p1, p2) {
  const m = (v, r = 0.2) => Math.random() < 0.1 ? v + (Math.random() - 0.5) * r : v
  return {
    activity: Math.min(1, Math.max(0, m(Math.random() < 0.5 ? p1.activity : p2.activity))),
    jumpForce: Math.max(1, m(Math.random() < 0.5 ? p1.jumpForce : p2.jumpForce, 5)),
    impulseRate: Math.max(500, m(Math.random() < 0.5 ? p1.impulseRate : p2.impulseRate, 1000)),
    color: Math.random() < 0.05 ? Math.random() * 0xffffff : (Math.random() < 0.5 ? p1.color : p2.color),
    w: m(Math.random() < 0.5 ? p1.w : p2.w, 2), h: m(Math.random() < 0.5 ? p1.h : p2.h, 2), d: m(Math.random() < 0.5 ? p1.d : p2.d, 2)
  }
}

async function runEvolutionCycle() {
  if (entities.length < 2) return
  const now = performance.now()
  entities.forEach(e => {
    const dist = e.mesh.position.distanceTo(e.startPos)
    const lifetime = (now - e.createdAt) / 1000
    if (optimizationGoal.value === 'speed') e.fitness = dist
    else if (optimizationGoal.value === 'survival') e.fitness = lifetime * 10 + (e.isAlive ? 100 : 0)
    else if (optimizationGoal.value === 'efficiency') e.fitness = dist / (e.energySpent + 0.1)
    else e.fitness = dist
  })
  const sorted = [...entities].sort((a, b) => b.fitness - a.fitness)
  bestFitness.value = Math.round(sorted[0].fitness)
  const survivors = sorted.slice(0, Math.ceil(entities.length * 0.25))
  const newGenomes = []
  while (newGenomes.length < entities.length) {
    newGenomes.push(breedBeings(survivors[Math.floor(Math.random() * survivors.length)].genome, survivors[Math.floor(Math.random() * survivors.length)].genome))
  }
  const count = entities.length
  entities.forEach(e => { scene.remove(e.mesh); physicsWorld.removeBody(e.body) })
  entities = []
  generation.value++
  for (let i = 0; i < count; i++) spawnRectangulata({ count: 1, genome: newGenomes[i] })
}

// --- DATA PERSISTENCE ---

async function saveWorld(name) {
  const data = {
    name, generation: generation.value, bestFitness: bestFitness.value, optimizationGoal: optimizationGoal.value,
    entities: entities.map(e => ({
      genome: e.genome, position: { x: e.body.position.x, y: e.body.position.y, z: e.body.position.z },
      quaternion: { x: e.body.quaternion.x, y: e.body.quaternion.y, z: e.body.quaternion.z, w: e.body.quaternion.w }
    }))
  }
  const res = await fetch(`${props.apiBase}/sorganoids/world/${name}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })
  if (res.ok) log(`World '${name}' Saved.`)
}

async function loadWorld(name) {
  try {
    const res = await fetch(`${props.apiBase}/sorganoids/world/${name}`)
    if (!res.ok) { log(`World '${name}' not found.`, 'error'); return }
    const payload = await res.json()
    const data = payload.world
    entities.forEach(e => { if(e.mesh) scene.remove(e.mesh); if(e.body) physicsWorld.removeBody(e.body) })
    entities = []
    worldName.value = data.name; generation.value = data.generation; bestFitness.value = data.bestFitness; optimizationGoal.value = data.optimizationGoal || 'speed'
    for (const e of data.entities) {
      if (e.genome.species) {
         await loadInstrument(e.genome.species, e.position)
      } else {
         spawnRectangulata({ count: 1, genome: e.genome })
         const newEnt = entities[entities.length-1]
         if (newEnt) {
            newEnt.body.position.set(e.position.x, e.position.y, e.position.z)
            newEnt.body.quaternion.set(e.quaternion.x, e.quaternion.y, e.quaternion.z, e.quaternion.w)
         }
      }
    }
    log(`World '${name}' Loaded.`)
  } catch (e) { log(`Load Error: ${e.message}`, 'error') }
}

// --- COMMAND PARSING ---

function parseCommand(cmd) {
  const text = cmd.trim().toLowerCase()
  if (audioListener) {
    const ctx = audioListener.context;
    if (ctx.state === 'suspended') ctx.resume().then(() => log('Audio Resumed'));
    if (masterGain) {
      const now = ctx.currentTime;
      masterGain.gain.cancelScheduledValues(now);
      masterGain.gain.exponentialRampToValueAtTime(0.5, now + 0.1);
    }
  }

  const parts = text.split(' ')
  const verb = parts[1]; const entity = parts[2]
  
  if (verb === 'create' && entity === 'world') {
    entities.forEach(e => { scene.remove(e.mesh); physicsWorld.removeBody(e.body) })
    entities = []; generation.value = 1; bestFitness.value = 0; worldName.value = parts[3] || 'void'; log(`World '${worldName.value}' Created.`)
  }
  if (verb === 'exit') {
    if (masterGain) {
      const now = audioListener.context.currentTime;
      masterGain.gain.exponentialRampToValueAtTime(0.001, now + 1.0);
    }
    setTimeout(() => { emit('exit'); }, 1000);
    log('Exiting Sorganoid World...')
  }
  if (verb === 'save' && entity === 'world') saveWorld(parts[3] || worldName.value)
  if (verb === 'load' && entity === 'world') loadWorld(parts[3] || 'void')
  if (text.startsWith('§ list worlds')) listWorlds()
  if (text.startsWith('§ list instruments')) listInstruments()
  
  if (text.startsWith('§ load instrument') || text.startsWith('§ spawn')) {
    const countMatch = text.match(/(\d+)\s+individuals/)
    const count = countMatch ? parseInt(countMatch[1]) : 1
    const nameMatch = text.match(/(?:of|instrument)\s+([a-zA-Z0-9-]+)/)
    const targetName = nameMatch ? nameMatch[1] : (text.includes('rectangulata') ? 'rectangulata' : null)
    
    if (targetName === 'rectangulata') {
       spawnRectangulata({ count })
    } else if (targetName) {
       for(let i=0; i<count; i++) setTimeout(() => loadInstrument(targetName), i * 300)
    }
  }

  if (text.includes('species') || text.includes('mutate') || text.includes('evolve being')) hatching.value = true
  
  if (verb === 'evolve' && !text.includes('being')) {
    if (text.includes('speed')) optimizationGoal.value = 'speed'
    else if (text.includes('survival')) optimizationGoal.value = 'survival'
    else if (text.includes('efficiency')) optimizationGoal.value = 'efficiency'
    const genMatch = text.match(/for\s+(\d+)\s+generations/)
    let currentGenCount = 0; const genLimit = genMatch ? parseInt(genMatch[1]) : 1
    const nextGen = () => { if (currentGenCount < genLimit) { runEvolutionCycle(); currentGenCount++; setTimeout(nextGen, 5000) } }
    nextGen()
  }
  if (verb === 'mutate' && entity === 'gravity') {
    const val = parseFloat(parts[3]); if (!isNaN(val)) physicsWorld.gravity.set(0, -val * 9.82, 0)
  }
}

function assetHref(url) {
  if (!url) return ''
  if (url.startsWith('http')) return url
  const base = props.apiBase
  const offloadApiBase = base.endsWith('/api') ? base.slice(0, -4) : base
  if (url.startsWith('/offload')) return offloadApiBase + url
  if (base.endsWith('/api') && url.startsWith('/api/')) return base + url.substring(4)
  return base + url
}

function animate() {
  animId = requestAnimationFrame(animate)
  const now = performance.now()
  const dt = (now - lastTime) / 1000
  lastTime = now

  if (hatchingEgg) {
    if (hatching.value || props.loading) {
      hatchingEgg.material.opacity = THREE.MathUtils.lerp(hatchingEgg.material.opacity, 0.8, 0.1)
      hatchingEgg.rotation.y += 0.05
      hatchingEgg.scale.setScalar(1 + Math.sin(now * 0.01) * 0.1)
    } else {
      hatchingEgg.material.opacity = THREE.MathUtils.lerp(hatchingEgg.material.opacity, 0, 0.1)
    }
  }

  if (physicsWorld) {
    physicsWorld.fixedStep()
    let totalEnergy = 0
    entities.forEach(e => {
      if (!e.isAlive) return
      e.energy = Math.min(1.2, e.energy + 0.0005)
      const targetScale = Math.max(0.2, e.energy)
      e.mesh.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.05)
      
      // Social Avoidance
      entities.forEach(other => {
        if (e === other || !other.isAlive) return
        if (e.mesh.position.distanceTo(other.mesh.position) < 40 && e.energy > 0.3) {
          const dir = e.mesh.position.clone().sub(other.mesh.position).normalize()
          e.body.applyImpulse(new CANNON.Vec3(dir.x * 5, 5, dir.z * 5), new CANNON.Vec3(0, 0, 0))
          e.energy -= 0.01
        }
      })

      // Genome actions
      if (now - e.lastAction > e.nextActionDelay && e.energy > 0.2) {
        const force = e.genome.jumpForce * e.genome.activity
        e.body.applyImpulse(new CANNON.Vec3((Math.random() - 0.5) * force, force, (Math.random() - 0.5) * force), new CANNON.Vec3(0, 0, 0))
        e.body.angularVelocity.set(Math.random() * 5, Math.random() * 5, Math.random() * 5)
        const d = (force / 100) * e.genome.metabolism; e.energy -= d; e.energySpent += d
        e.lastAction = now; e.nextActionDelay = e.genome.impulseRate * (1.5 - e.genome.activity)
      }
      
      if (e.energy <= 0.2) { e.isAlive = false; e.mesh.traverse(c => { if(c.isMesh) { c.material.wireframe = false; c.material.transparent = true; c.material.opacity = 0.2 } }); }
      e.mesh.position.copy(e.body.position); e.mesh.quaternion.copy(e.body.quaternion)
      if (e.body.position.y < -500) { e.body.position.set((Math.random()-0.5)*100, 300, (Math.random()-0.5)*100); e.body.velocity.set(0,0,0); }
      e.mesh.traverse(c => { if(c.isMesh && c.material.emissiveIntensity > 0) c.material.emissiveIntensity *= 0.95 })
      totalEnergy += e.energy
    })
    if (entities.length > 0) avgEnergy.value = Math.round((totalEnergy / entities.length) * 100)
  }

  if (controls) controls.update()
  if (renderer && scene && camera) renderer.render(scene, camera)
}

function handleResize() {
  if (!camera || !renderer) return
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
}

onMounted(() => { init(); window.addEventListener('resize', handleResize) })
onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  if (animId) cancelAnimationFrame(animId)
  if (renderer) { renderer.dispose(); renderer.domElement.remove() }
  if (audioListener && audioListener.context) audioListener.context.close()
})

watch(() => props.command, (cmd) => { if (cmd) parseCommand(cmd) })
watch(() => props.result, (newResult) => {
  if (newResult && newResult.gallery) {
    const item = newResult.gallery; if (item.stl_url) loadNeuralBeing(item.stl_url, item.basename, (item.sound_samples || []).map(s => assetHref(s.url)))
  }
})
</script>

<style scoped>
.sorganoid-world {
  position: fixed;
  top: 0; left: 0; width: 100vw; height: 100vh;
  z-index: 5; pointer-events: none; opacity: 0; transition: opacity 1s ease;
}
.sorganoid-world.is-active { opacity: 1; }
.sorganoid-world.is-interacting { z-index: 25; pointer-events: auto; }
.world-hud {
  position: absolute; top: 20px; right: 20px; text-align: right;
  font-family: monospace; color: #00ff00; pointer-events: none; z-index: 30;
  display: flex; flex-direction: column; align-items: flex-end;
}
.stat {
  font-size: 10px; letter-spacing: 1.5px; margin-bottom: 4px;
  background: rgba(0,0,0,0.6); padding: 3px 8px; border-right: 2px solid #00ff00;
}
.status-msg {
  font-size: 9px; color: #fff; margin-top: 10px; animation: blink 1.5s infinite;
}
@keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.2; } 100% { opacity: 1; } }
</style>
