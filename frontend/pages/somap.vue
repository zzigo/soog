<template>
  <div class="somap-root">
    <div ref="graphContainer" class="graph-surface"></div>

    <section class="hud hud-left">
      <div class="hud-title">
        <span class="title">SOMAP</span>
        <span class="subtitle">Obsidian Knowledge Graph</span>
      </div>
      <div class="stats">
        <span class="chip">nodes {{ graphStats.nodes }}</span>
        <span class="chip">links {{ graphStats.links }}</span>
        <span class="chip">notes {{ graphStats.notes }}</span>
        <span class="chip">tags {{ graphStats.tags }}</span>
      </div>
      <div class="legend">
        <span class="legend-item folder">folders</span>
        <span class="legend-item note">notes</span>
        <span class="legend-item tag">tags</span>
        <span class="legend-item ghost">missing refs</span>
      </div>
    </section>

    <section class="hud hud-right">
      <input
        v-model="finderQuery"
        class="finder"
        type="text"
        placeholder="Find node"
        @keydown.enter.prevent="focusFinderNode"
      />
      <div class="actions">
        <button class="hud-btn" @click="focusFinderNode">Focus</button>
        <button class="hud-btn" @click="refreshGraph">Reload</button>
        <button class="hud-btn" :class="{ active: autoRotate }" @click="toggleAutoRotate">
          Orbit {{ autoRotate ? 'on' : 'off' }}
        </button>
        <button class="hud-btn" :class="{ active: labelMode }" @click="toggleLabelMode">
          Labels {{ labelMode ? 'on' : 'off' }}
        </button>
        <button class="hud-btn" :class="{ active: showGhosts }" @click="toggleGhosts">
          Missing {{ showGhosts ? 'on' : 'off' }}
        </button>
      </div>
      <div class="shortcut-note">s labels | alt+1 soog | alt+2 somap</div>
    </section>

    <section class="inspector" v-if="activeNode">
      <div class="inspector-head">
        <span class="inspector-type">{{ activeNode.type }}</span>
        <span class="inspector-degree">degree {{ activeNode.degree || 0 }}</span>
      </div>
      <h3 class="inspector-title">{{ activeNode.title || activeNode.id }}</h3>
      <p class="inspector-path" v-if="activeNode.path">{{ activeNode.path }}</p>
      <p class="inspector-preview" v-if="activeNode.preview">{{ activeNode.preview }}</p>
      <p class="inspector-preview" v-else>No preview available for this node.</p>
    </section>

    <div v-if="loading" class="status">Loading knowledge graph...</div>
    <div v-if="error" class="status error">{{ error }}</div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRuntimeConfig } from '#app'
import { useRoute, useRouter } from 'vue-router'
import * as THREE from 'three'

const config = useRuntimeConfig()
const router = useRouter()
const route = useRoute()
const apiBase = config.public.apiBase || 'http://127.0.0.1:10000/api'

const graphContainer = ref(null)
const loading = ref(false)
const error = ref('')
const finderQuery = ref('')
const autoRotate = ref(true)
const showGhosts = ref(true)
const labelMode = ref(false)
const selectedNode = ref(null)
const hoveredNode = ref(null)
const rawGraph = ref({ nodes: [], links: [], stats: {} })
const preparedGraph = ref({ nodes: [], links: [] })

const graphStats = computed(() => rawGraph.value.stats || { nodes: 0, links: 0, notes: 0, tags: 0 })
const activeNode = computed(() => selectedNode.value || hoveredNode.value)

let graphInstance = null
let ForceGraph3D = null
let resizeHandler = null
let keyHandler = null
let rotateRaf = null

const highlightNodes = new Set()
const highlightLinks = new Set()
const materialCache = new Map()
const geometryMap = {
  folder: new THREE.BoxGeometry(1, 1, 1),
  note: new THREE.OctahedronGeometry(1),
  tag: new THREE.TetrahedronGeometry(1),
  ghost: new THREE.SphereGeometry(1, 12, 12)
}

function getNodeColor(node) {
  switch (node?.type) {
    case 'folder':
      return '#6cb9ff'
    case 'tag':
      return '#ffbf66'
    case 'ghost':
      return '#ff6a6a'
    case 'note':
    default:
      return '#97ff79'
  }
}

function getLinkColor(link) {
  if (highlightLinks.has(link)) return '#ffffff'
  switch (link?.type) {
    case 'folder_contains':
      return 'rgba(110, 193, 255, 0.42)'
    case 'tagged':
      return 'rgba(255, 191, 102, 0.38)'
    case 'wikilink':
      return 'rgba(151, 255, 121, 0.38)'
    case 'wikilink_unresolved':
      return 'rgba(255, 106, 106, 0.36)'
    default:
      return 'rgba(255, 255, 255, 0.2)'
  }
}

function getLinkOpacity(link) {
  if (highlightLinks.has(link)) return 0.92
  switch (link?.type) {
    case 'folder_contains':
      return 0.38
    case 'tagged':
      return 0.34
    case 'wikilink':
      return 0.34
    case 'wikilink_unresolved':
      return 0.3
    default:
      return 0.18
  }
}

function getLinkWidth(link) {
  const base = link?.type === 'folder_contains' ? 1.2 : 0.75
  return highlightLinks.has(link) ? base + 1.3 : base
}

function getNodeSize(node) {
  const val = Number.isFinite(Number(node?.val)) ? Number(node.val) : 2
  const base = Math.max(1.2, val)
  const selectedBoost = selectedNode.value?.id === node?.id ? 1.28 : 1
  const hoveredBoost = hoveredNode.value?.id === node?.id ? 1.16 : 1
  const focusBoost = highlightNodes.size > 0 && highlightNodes.has(node) ? 1.08 : 1
  return base * selectedBoost * hoveredBoost * focusBoost
}

function materialKey(color, opacity, wireframe) {
  return `${color}|${opacity}|${wireframe ? 1 : 0}`
}

function getMaterial(color, opacity = 0.9, wireframe = false) {
  const key = materialKey(color, opacity, wireframe)
  if (!materialCache.has(key)) {
    materialCache.set(
      key,
      new THREE.MeshLambertMaterial({
        color,
        transparent: true,
        opacity,
        wireframe
      })
    )
  }
  return materialCache.get(key)
}

function createNodeObject(node) {
  const type = node?.type || 'note'
  if (labelMode.value && type === 'note') {
    const text = String(node?.title || node?.id || 'note')
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    const fontSize = 36
    const paddingX = 22
    const paddingY = 12
    ctx.font = `600 ${fontSize}px Courier New, monospace`
    const width = Math.ceil(ctx.measureText(text).width) + paddingX * 2
    canvas.width = Math.max(128, Math.min(1200, width))
    canvas.height = fontSize + paddingY * 2

    const drawCtx = canvas.getContext('2d')
    drawCtx.font = `600 ${fontSize}px Courier New, monospace`
    drawCtx.textAlign = 'center'
    drawCtx.textBaseline = 'middle'
    drawCtx.fillStyle = getNodeColor(node)
    drawCtx.fillText(text, canvas.width / 2, canvas.height / 2)

    const texture = new THREE.CanvasTexture(canvas)
    texture.colorSpace = THREE.SRGBColorSpace
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
      depthTest: true
    })
    const sprite = new THREE.Sprite(material)
    const aspect = canvas.width / canvas.height
    const height = Math.max(6, getNodeSize(node) * 2.2)
    sprite.scale.set(height * aspect, height, 1)
    return sprite
  }

  const geometry = geometryMap[type] || geometryMap.note
  const mesh = new THREE.Mesh(
    geometry,
    getMaterial(getNodeColor(node), 0.9, type === 'folder')
  )
  const size = getNodeSize(node)
  mesh.scale.setScalar(size)
  return mesh
}

function resolveLinkNodeId(value) {
  return typeof value === 'object' ? value?.id : value
}

function prepareGraph(payload) {
  const sourceNodes = Array.isArray(payload?.nodes) ? payload.nodes : []
  const sourceLinks = Array.isArray(payload?.links) ? payload.links : []

  const nodes = sourceNodes
    .filter((node) => showGhosts.value || node.type !== 'ghost')
    .map((node) => ({ ...node, neighbors: [], links: [] }))

  const nodeById = new Map(nodes.map((node) => [node.id, node]))
  const links = []

  for (const rawLink of sourceLinks) {
    const sourceId = resolveLinkNodeId(rawLink.source)
    const targetId = resolveLinkNodeId(rawLink.target)
    const sourceNode = nodeById.get(sourceId)
    const targetNode = nodeById.get(targetId)
    if (!sourceNode || !targetNode) continue

    const edge = {
      source: sourceNode.id,
      target: targetNode.id,
      type: rawLink.type || 'relation'
    }
    links.push(edge)
    sourceNode.neighbors.push(targetNode)
    targetNode.neighbors.push(sourceNode)
    sourceNode.links.push(edge)
    targetNode.links.push(edge)
  }

  return { nodes, links }
}

function applyControlsConfig() {
  const controls = graphInstance?.controls?.()
  if (!controls) return
  if ('enableDamping' in controls) {
    controls.enableDamping = true
    controls.dampingFactor = 0.12
  }
  if ('rotateSpeed' in controls) controls.rotateSpeed = 0.45
  if ('zoomSpeed' in controls) controls.zoomSpeed = 0.8
  if ('panSpeed' in controls) controls.panSpeed = 0.65
}

function updateHighlight() {
  if (!graphInstance) return
  const current = graphInstance.graphData() || { nodes: [], links: [] }
  for (const node of current.nodes || []) {
    const obj = node.__threeObj
    if (!obj) continue
    const nodeType = node?.type || 'note'
    const dimmed = highlightNodes.size > 0 && !highlightNodes.has(node)
    if (obj.isSprite) {
      obj.material.opacity = dimmed ? 0.16 : 1
      const aspect = obj.scale.y > 0 ? obj.scale.x / obj.scale.y : 1
      const height = Math.max(6, getNodeSize(node) * 2.2)
      obj.scale.set(height * aspect, height, 1)
      continue
    }
    const color = dimmed ? '#181818' : getNodeColor(node)
    const opacity = dimmed ? 0.12 : 0.9
    obj.material = getMaterial(color, opacity, nodeType === 'folder')
    obj.scale.setScalar(getNodeSize(node))
  }
  graphInstance
    .linkColor(graphInstance.linkColor())
    .linkOpacity(graphInstance.linkOpacity())
    .linkWidth(graphInstance.linkWidth())
}

function setFocusNode(node) {
  highlightNodes.clear()
  highlightLinks.clear()

  if (node) {
    highlightNodes.add(node)
    for (const neighbor of node.neighbors || []) highlightNodes.add(neighbor)
    for (const edge of node.links || []) highlightLinks.add(edge)
  }

  updateHighlight()
}

function flyToNode(node) {
  if (!graphInstance || !node) return
  const norm = Math.max(1, Math.hypot(node.x || 0, node.y || 0, node.z || 0))
  const distance = 110
  const ratio = 1 + distance / norm
  graphInstance.cameraPosition(
    { x: (node.x || 0) * ratio, y: (node.y || 0) * ratio, z: (node.z || 0) * ratio },
    { x: node.x || 0, y: node.y || 0, z: node.z || 0 },
    900
  )
}

function initGraph() {
  if (!graphContainer.value || graphInstance || !ForceGraph3D) return

  graphInstance = ForceGraph3D()(graphContainer.value)
    .backgroundColor('#000000')
    .showNavInfo(false)
    .numDimensions(3)
    .nodeLabel((node) => {
      const path = node?.path ? `<div class="lbl-path">${node.path}</div>` : ''
      return `<div class="lbl"><strong>${node.title || node.id}</strong>${path}</div>`
    })
    .nodeThreeObject((node) => createNodeObject(node))
    .linkColor((link) => getLinkColor(link))
    .linkOpacity((link) => getLinkOpacity(link))
    .linkWidth((link) => getLinkWidth(link))
    .linkDirectionalParticles(0)
    .onNodeHover((node) => {
      hoveredNode.value = node || null
      if (!selectedNode.value) setFocusNode(node || null)
      document.body.style.cursor = node ? 'pointer' : ''
    })
    .onNodeClick((node) => {
      selectedNode.value = node || null
      setFocusNode(node || null)
      flyToNode(node)
    })
    .onBackgroundClick(() => {
      selectedNode.value = null
      setFocusNode(hoveredNode.value || null)
    })

  graphInstance.d3Force('charge').strength(-280)
  graphInstance.d3Force('link').distance((link) => {
    if (link.type === 'folder_contains') return 55
    if (link.type === 'tagged') return 72
    if (link.type === 'wikilink_unresolved') return 90
    return 105
  }).strength(0.55)
  graphInstance.d3Force('center').strength(0.08)
  graphInstance.d3VelocityDecay(0.34)
  graphInstance.d3AlphaDecay(0.03)
  graphInstance.cooldownTicks(220)

  const lightA = new THREE.AmbientLight(0xffffff, 0.7)
  const lightB = new THREE.DirectionalLight(0xffffff, 0.58)
  lightB.position.set(100, 80, 50)
  graphInstance.scene().add(lightA)
  graphInstance.scene().add(lightB)
  graphInstance.cameraPosition({ x: 0, y: 0, z: 560 })

  applyControlsConfig()

  resizeHandler = () => {
    if (!graphInstance || !graphContainer.value) return
    const { clientWidth, clientHeight } = graphContainer.value
    graphInstance.width(clientWidth || window.innerWidth)
    graphInstance.height(clientHeight || window.innerHeight)
  }
  resizeHandler()
  window.addEventListener('resize', resizeHandler)
}

function startAutoRotate() {
  stopAutoRotate()
  if (!autoRotate.value) return
  const step = () => {
    if (!autoRotate.value || !graphInstance) return
    const camera = graphInstance.camera()
    if (camera) {
      const angle = 0.001
      const cos = Math.cos(angle)
      const sin = Math.sin(angle)
      const { x, z } = camera.position
      camera.position.x = x * cos + z * sin
      camera.position.z = z * cos - x * sin
      camera.lookAt(0, 0, 0)
    }
    rotateRaf = requestAnimationFrame(step)
  }
  rotateRaf = requestAnimationFrame(step)
}

function stopAutoRotate() {
  if (rotateRaf) {
    cancelAnimationFrame(rotateRaf)
    rotateRaf = null
  }
}

function toggleAutoRotate() {
  autoRotate.value = !autoRotate.value
  if (autoRotate.value) startAutoRotate()
  else stopAutoRotate()
}

function toggleLabelMode() {
  labelMode.value = !labelMode.value
  if (!graphInstance) return
  graphInstance.nodeThreeObject((node) => createNodeObject(node))
  graphInstance.d3ReheatSimulation()
  setFocusNode(selectedNode.value || hoveredNode.value || null)
}

async function refreshGraph() {
  loading.value = true
  error.value = ''
  try {
    const response = await fetch(`${apiBase}/somap/graph`, {
      headers: { Accept: 'application/json' }
    })
    const payload = await response.json()
    if (!response.ok) throw new Error(payload?.error || `Failed (${response.status})`)
    rawGraph.value = payload
    preparedGraph.value = prepareGraph(payload)
    const nodeById = new Map((preparedGraph.value.nodes || []).map((node) => [node.id, node]))
    selectedNode.value = selectedNode.value ? (nodeById.get(selectedNode.value.id) || null) : null
    hoveredNode.value = hoveredNode.value ? (nodeById.get(hoveredNode.value.id) || null) : null

    if (graphInstance) {
      graphInstance.graphData(preparedGraph.value)
      graphInstance.d3ReheatSimulation()
      setFocusNode(selectedNode.value || hoveredNode.value || null)
    }
  } catch (err) {
    error.value = err?.message || 'Could not load graph data.'
  } finally {
    loading.value = false
  }
}

function toggleGhosts() {
  showGhosts.value = !showGhosts.value
  preparedGraph.value = prepareGraph(rawGraph.value)
  const nodeById = new Map((preparedGraph.value.nodes || []).map((node) => [node.id, node]))
  selectedNode.value = selectedNode.value ? (nodeById.get(selectedNode.value.id) || null) : null
  hoveredNode.value = hoveredNode.value ? (nodeById.get(hoveredNode.value.id) || null) : null
  if (graphInstance) {
    graphInstance.graphData(preparedGraph.value)
    graphInstance.d3ReheatSimulation()
    setFocusNode(selectedNode.value || hoveredNode.value || null)
  }
}

function focusFinderNode() {
  const query = finderQuery.value.trim().toLowerCase()
  if (!query || !graphInstance) return
  const node = (preparedGraph.value.nodes || []).find((candidate) => {
    return (
      String(candidate.title || '').toLowerCase().includes(query) ||
      String(candidate.path || '').toLowerCase().includes(query) ||
      String(candidate.id || '').toLowerCase().includes(query)
    )
  })
  if (!node) return
  selectedNode.value = node
  setFocusNode(node)
  flyToNode(node)
}

function handleShortcuts(event) {
  const targetTag = String(event?.target?.tagName || '').toLowerCase()
  const isTypingTarget =
    targetTag === 'input' || targetTag === 'textarea' || Boolean(event?.target?.isContentEditable)

  const isAltDigit = (digit) =>
    event.altKey && (event.code === `Digit${digit}` || event.key === String(digit))

  if (isAltDigit(1)) {
    event.preventDefault()
    if (route.path !== '/') router.push('/')
    return
  }
  if (isAltDigit(2)) {
    event.preventDefault()
    if (route.path !== '/somap') router.push('/somap')
    return
  }
  if (isTypingTarget) return

  if (!event.altKey && event.key.toLowerCase() === 's') {
    event.preventDefault()
    toggleLabelMode()
  }
}

onMounted(async () => {
  const module = await import('3d-force-graph')
  ForceGraph3D = module.default
  initGraph()
  await refreshGraph()
  startAutoRotate()

  keyHandler = (event) => handleShortcuts(event)
  window.addEventListener('keydown', keyHandler)
})

onUnmounted(() => {
  stopAutoRotate()
  if (resizeHandler) window.removeEventListener('resize', resizeHandler)
  if (keyHandler) window.removeEventListener('keydown', keyHandler)
  if (graphContainer.value) graphContainer.value.innerHTML = ''
  materialCache.clear()
})
</script>

<style scoped>
.somap-root {
  position: relative;
  width: 100vw;
  height: 100vh;
  background: #000;
  color: rgba(255, 255, 255, 0.88);
  overflow: hidden;
  font-family: 'Courier New', monospace;
}

.graph-surface {
  position: absolute;
  inset: 0;
}

.hud {
  position: absolute;
  z-index: 20;
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: rgba(0, 0, 0, 0.44);
  padding: 10px 12px;
}

.hud-left {
  top: 12px;
  left: 12px;
  width: min(420px, calc(100vw - 24px));
}

.hud-right {
  top: 12px;
  right: 12px;
  width: min(320px, calc(100vw - 24px));
  display: grid;
  gap: 8px;
}

.hud-title {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
}

.title {
  letter-spacing: 0.12em;
  font-size: 13px;
}

.subtitle {
  opacity: 0.62;
  font-size: 11px;
}

.stats {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}

.chip {
  border: 1px solid rgba(255, 255, 255, 0.16);
  padding: 2px 6px;
  font-size: 11px;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 9px;
  font-size: 11px;
}

.legend-item {
  opacity: 0.86;
}

.legend-item.folder { color: #6cb9ff; }
.legend-item.note { color: #97ff79; }
.legend-item.tag { color: #ffbf66; }
.legend-item.ghost { color: #ff6a6a; }

.finder {
  width: 100%;
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: transparent;
  color: #fff;
  padding: 7px 8px;
  outline: none;
  font-family: inherit;
  font-size: 12px;
}

.finder::placeholder {
  color: rgba(255, 255, 255, 0.42);
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.hud-btn {
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: transparent;
  color: rgba(255, 255, 255, 0.9);
  padding: 5px 8px;
  font-family: inherit;
  font-size: 11px;
  cursor: pointer;
}

.hud-btn:hover {
  border-color: rgba(255, 255, 255, 0.42);
}

.hud-btn.active {
  border-color: #97ff79;
  color: #97ff79;
}

.shortcut-note {
  font-size: 10px;
  opacity: 0.56;
  letter-spacing: 0.04em;
}

.inspector {
  position: absolute;
  left: 12px;
  bottom: 12px;
  z-index: 20;
  width: min(480px, calc(100vw - 24px));
  max-height: 38vh;
  overflow: auto;
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: rgba(0, 0, 0, 0.58);
  backdrop-filter: blur(4px);
  padding: 10px 12px;
}

.inspector-head {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  opacity: 0.7;
}

.inspector-title {
  margin: 6px 0 4px 0;
  font-size: 15px;
  color: #fff;
}

.inspector-path {
  margin: 0;
  font-size: 11px;
  color: rgba(108, 185, 255, 0.9);
}

.inspector-preview {
  margin: 8px 0 0 0;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  color: rgba(255, 255, 255, 0.84);
}

.status {
  position: absolute;
  left: 50%;
  bottom: 16px;
  transform: translateX(-50%);
  z-index: 25;
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: rgba(0, 0, 0, 0.68);
  padding: 6px 10px;
  font-size: 12px;
}

.status.error {
  border-color: rgba(255, 96, 96, 0.6);
  color: #ff8e8e;
}

@media (max-width: 900px) {
  .hud-left,
  .hud-right,
  .inspector {
    width: calc(100vw - 24px);
  }
  .hud-right {
    top: auto;
    bottom: 120px;
    right: 12px;
  }
}
</style>
