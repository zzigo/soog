<template>
  <div ref="container" class="stl-container"></div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'

const props = defineProps({
  url: { type: String, required: true }
})

let scene, camera, renderer, controls, mesh
const container = ref(null)
let animId

function disposeThree() {
  if (animId) cancelAnimationFrame(animId)
  animId = null
  if (controls && controls.dispose) controls.dispose()
  if (renderer) {
    renderer.dispose()
    renderer.forceContextLoss && renderer.forceContextLoss()
    renderer.domElement && renderer.domElement.remove()
  }
  scene = camera = renderer = controls = mesh = null
}

async function init() {
  if (!container.value) return
  const THREE = await import('three')
  const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js')
  const { STLLoader } = await import('three/examples/jsm/loaders/STLLoader.js')
  const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js')

  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)

  const width = container.value.clientWidth || 640
  const height = container.value.clientHeight || 480

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 10000)
  camera.position.set(0, 0, 300)

  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(width, height)
  renderer.setPixelRatio(window.devicePixelRatio || 1)
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.2
  container.value.appendChild(renderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)

  // --- Cinematic Lighting ---
  
  // 1. Godray Backlight (Very strong from behind)
  const backLight = new THREE.DirectionalLight(0xffffff, 8.0)
  backLight.position.set(0, 100, -400)
  scene.add(backLight)

  // 2. High-Intensity Side Spots (Top to Down)
  const spotLeft = new THREE.SpotLight(0xffffff, 15.0)
  spotLeft.position.set(-300, 500, 200)
  spotLeft.angle = Math.PI / 6
  spotLeft.penumbra = 0.3
  spotLeft.decay = 1
  spotLeft.distance = 2000
  scene.add(spotLeft)

  const spotRight = new THREE.SpotLight(0xffffff, 15.0)
  spotRight.position.set(300, 500, 200)
  spotRight.angle = Math.PI / 6
  spotRight.penumbra = 0.3
  spotRight.decay = 1
  spotRight.distance = 2000
  scene.add(spotRight)

  // 3. Soft Ambient Fill
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4)
  scene.add(ambientLight)

  // 4. Rim Light (to enhance edges)
  const rimLight = new THREE.PointLight(0xffffff, 10.0)
  rimLight.position.set(0, -200, 100)
  scene.add(rimLight)

  const isGlb = props.url.toLowerCase().endsWith('.glb')
  const loader = isGlb ? new GLTFLoader() : new STLLoader()

  loader.load(props.url, (result) => {
    let geometry
    let material = new THREE.MeshStandardMaterial({ color: 0x66ccff, metalness: 0.1, roughness: 0.6 })

    if (isGlb) {
      // result is a GLTF object
      mesh = result.scene
      // Find the first mesh to get geometry for auto-scale
      result.scene.traverse((child) => {
        if (child.isMesh) {
          geometry = child.geometry
          // If GLB has vertex colors, use them
          if (geometry.attributes.color) {
            child.material.vertexColors = true
          }
        }
      })
    } else {
      // result is a BufferGeometry
      geometry = result
      geometry.computeVertexNormals && geometry.computeVertexNormals()
      mesh = new THREE.Mesh(geometry, material)
    }

    if (geometry) {
      geometry.center && geometry.center()
      geometry.computeBoundingSphere()
      const bs = geometry.boundingSphere
      const R = (bs && bs.radius) ? bs.radius : 50
      const targetDist = R * 3.0
      camera.position.set(targetDist, targetDist, targetDist)
      camera.lookAt(0, 0, 0)
    }

    scene.add(mesh)
  }, 
  (xhr) => { /* progress */ },
  (error) => { console.error('Error loading 3D model:', error) })

  function onResize() {
    if (!container.value) return
    const w = container.value.clientWidth || 640
    const h = container.value.clientHeight || 480
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h)
  }
  window.addEventListener('resize', onResize)

  const animate = () => {
    if (!renderer || !scene || !camera) return
    animId = requestAnimationFrame(animate)
    renderer.render(scene, camera)
  }
  animate()

  onUnmounted(() => {
    window.removeEventListener('resize', onResize)
    disposeThree()
  })
}

onMounted(() => {
  if (process.server) return
  init()
})

watch(() => props.url, () => {
  // Reload on url change: simplest is to recreate
  disposeThree()
  init()
})
</script>

<style scoped>
.stl-container {
  width: 100%;
  height: 100%;
  background: #000;
}
</style>
