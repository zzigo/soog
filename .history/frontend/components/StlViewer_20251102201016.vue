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

  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)

  const width = container.value.clientWidth || 640
  const height = container.value.clientHeight || 480

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 10000)
  camera.position.set(0, 0, 300)

  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(width, height)
  renderer.setPixelRatio(window.devicePixelRatio || 1)
  container.value.appendChild(renderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)

  const light1 = new THREE.DirectionalLight(0xffffff, 0.9)
  light1.position.set(1, 1, 1)
  scene.add(light1)
  const light2 = new THREE.AmbientLight(0x555555)
  scene.add(light2)

  const loader = new STLLoader()
  loader.load(props.url, (geometry) => {
    geometry.computeVertexNormals && geometry.computeVertexNormals()
    const material = new THREE.MeshStandardMaterial({ color: 0x66ccff, metalness: 0.1, roughness: 0.6 })
    mesh = new THREE.Mesh(geometry, material)
    geometry.center && geometry.center()

    // Autoscale to fit
    geometry.computeBoundingSphere()
    const bs = geometry.boundingSphere
    const R = (bs && bs.radius) ? bs.radius : 50
    const targetDist = R * 3.0
    camera.position.set(targetDist, targetDist, targetDist)
    camera.lookAt(0, 0, 0)

    scene.add(mesh)
  })

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
