<template>
  <div ref="stageEl" class="stage3d" aria-label="Three-dimensional graph of registry notes; facets sit on the vertices of a pentagon"></div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { FKEY, GHOST_COLOR, MAIN_COLUMNS, colorOf, isInstrument, linkColor, radiusOf } from '~/composables/useSomRegistry'

const props = defineProps({ reg: { type: Object, required: true } })
const emit = defineEmits(['select'])
const reg = props.reg

const stageEl = ref(null)
let graph = null
let resizeObserver = null
let nodes = []
let links = []

const PENT_R = 150
// The five MOAIE facets sit on a regular pentagon in the XZ plane (the 2D view uses columns).
const vertex = key => {
  const i = MAIN_COLUMNS.indexOf(key)
  const a = -Math.PI / 2 + (i * 2 * Math.PI) / 5
  return { x: Math.cos(a) * PENT_R, z: Math.sin(a) * PENT_R }
}
const idOf = e => (typeof e === 'object' ? e.id : e)

// One wireframe solid per facet; instruments are rings.
const GEOMETRY = {
  '1mat': () => new THREE.TetrahedronGeometry(1),
  '2obj': () => new THREE.BoxGeometry(1.4, 1.4, 1.4),
  '3agn': () => new THREE.OctahedronGeometry(1),
  '4int': () => new THREE.DodecahedronGeometry(1),
  '5env': () => new THREE.IcosahedronGeometry(1),
  '0pri': () => new THREE.ConeGeometry(0.9, 1.8, 5),
  other: () => new THREE.SphereGeometry(1, 8, 6),
  ghost: () => new THREE.SphereGeometry(1, 6, 4)
}
const geoCache = new Map()
const matCache = new Map()
function geometryFor(n) {
  const key = n.ghost ? 'ghost' : isInstrument(n) ? 'instrument' : n.facet
  if (!geoCache.has(key)) {
    const base = key === 'instrument' ? new THREE.TorusGeometry(1, 0.28, 5, 14) : (GEOMETRY[key] || GEOMETRY.other)()
    geoCache.set(key, new THREE.EdgesGeometry(base, 1))
    base.dispose()
  }
  return geoCache.get(key)
}
function lineMaterial(color, opacity) {
  const k = `${color}|${opacity}`
  if (!matCache.has(k)) matCache.set(k, new THREE.LineBasicMaterial({ color, transparent: true, opacity }))
  return matCache.get(k)
}
function labelSprite(text, opacity) {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  const font = '28px "IBM Plex Mono", monospace'
  ctx.font = font
  const w = Math.ceil(ctx.measureText(text).width) + 12
  canvas.width = w
  canvas.height = 40
  ctx.font = font
  ctx.fillStyle = '#ffffff'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, 6, 21)
  const tex = new THREE.CanvasTexture(canvas)
  tex.minFilter = THREE.LinearFilter
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, opacity, depthWrite: false }))
  sprite.scale.set(w / 6, 40 / 6, 1)
  return sprite
}

function neighborhood() {
  const id = reg.selId.value
  return id ? new Set([id, ...(reg.adjacency.value.get(id) || [])]) : null
}

function makeObject(n) {
  const nb = neighborhood()
  const dim = nb && !nb.has(n.id)
  const selected = n.id === reg.selId.value
  const group = new THREE.Group()
  const s = n.ghost ? 2.2 : 3 + Math.sqrt(n.deg) * 1.15
  const lines = new THREE.LineSegments(geometryFor(n), lineMaterial(selected ? '#ffffff' : colorOf(n), dim ? 0.12 : n.ghost ? 0.5 : 0.95))
  lines.scale.setScalar(s)
  group.add(lines)
  if (!n.ghost && (selected || (nb && nb.has(n.id)) || n.deg >= 8 || (reg.query.value.trim() && reg.matches(n)))) {
    const label = labelSprite(n.title.length > 28 ? n.title.slice(0, 27) + '…' : n.title, dim ? 0.1 : 0.95)
    label.position.set(0, s + 4, 0)
    group.add(label)
  }
  return group
}

function linkStyle(l) {
  const id = reg.selId.value
  const on = id && (idOf(l.source) === id || idOf(l.target) === id)
  return { on, dim: id && !on }
}

function restyle() {
  if (!graph) return
  graph.nodeThreeObject(makeObject)
  graph.linkColor(l => (linkStyle(l).dim ? 'rgba(255,255,255,0.05)' : linkColor(l.type)))
  graph.linkOpacity(0.5)
  graph.linkWidth(l => (linkStyle(l).on ? 1.6 : l.type.startsWith('moaie') ? 0.5 : 0))
  graph.refresh()
}

function focus(id) {
  if (!graph) return
  const ids = id ? new Set([id, ...(reg.adjacency.value.get(id) || [])]) : null
  const pts = nodes.filter(n => reg.isVisible(n) && (!ids || ids.has(n.id)) && Number.isFinite(n.x))
  if (!pts.length) return
  const c = { x: 0, y: 0, z: 0 }
  pts.forEach(n => { c.x += n.x; c.y += n.y || 0; c.z += n.z || 0 })
  c.x /= pts.length; c.y /= pts.length; c.z /= pts.length
  const r = Math.max(25, ...pts.map(n => Math.hypot(n.x - c.x, (n.y || 0) - c.y, (n.z || 0) - c.z))) + 14
  const cam = graph.camera().position
  const dir = new THREE.Vector3(cam.x - c.x, cam.y - c.y, cam.z - c.z)
  if (dir.length() < 1e-3) dir.set(0, 0.35, 1)
  dir.normalize()
  const fov = (graph.camera().fov * Math.PI) / 180
  const dist = Math.max(70, (r / Math.sin(fov / 2)) * 1.05)
  graph.cameraPosition({ x: c.x + dir.x * dist, y: c.y + dir.y * dist, z: c.z + dir.z * dist }, c, 900)
}

function target(n) {
  if (n.ghost && !n.xKeys.length) return { x: 0, y: 0, z: 0 }
  const keys = n.xKeys.length ? n.xKeys : ['2obj']
  const vs = keys.map(vertex)
  return {
    x: vs.reduce((s, v) => s + v.x, 0) / vs.length,
    y: n.ghost ? 0 : isInstrument(n) ? -55 : 35,
    z: vs.reduce((s, v) => s + v.z, 0) / vs.length
  }
}

function addFacetFrame(THREEns, scene) {
  // Wireframe pentagon with a post and a letter at each facet vertex.
  const pts = MAIN_COLUMNS.map(k => { const v = vertex(k); return new THREEns.Vector3(v.x, -90, v.z) })
  pts.push(pts[0].clone())
  scene.add(new THREEns.Line(new THREEns.BufferGeometry().setFromPoints(pts), new THREEns.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.2 })))
  MAIN_COLUMNS.forEach(k => {
    const v = vertex(k)
    const f = FKEY.get(k)
    const post = new THREEns.Line(
      new THREEns.BufferGeometry().setFromPoints([new THREEns.Vector3(v.x, -90, v.z), new THREEns.Vector3(v.x, 90, v.z)]),
      new THREEns.LineBasicMaterial({ color: f.color, transparent: true, opacity: 0.35 }))
    scene.add(post)
    const sp = labelSprite(`${f.short} ${f.label.toUpperCase()}`, 0.75)
    sp.position.set(v.x * 1.12, 102, v.z * 1.12)
    scene.add(sp)
  })
}

async function init() {
  const mod = await import('3d-force-graph')
  const ForceGraph3D = mod.default
  nodes = reg.nodes.value.map(n => ({ ...n }))
  links = reg.links.value.map(l => ({ ...l }))
  graph = ForceGraph3D()(stageEl.value)
    .backgroundColor('#000000')
    .showNavInfo(false)
    .width(stageEl.value.clientWidth)
    .height(stageEl.value.clientHeight)
    .graphData({ nodes, links })
    .nodeId('id')
    .nodeLabel(n => `<div class="tip3d">${String(n.title).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]))}<br><span>${n.ghost ? 'pending note' : n.category || 'uncategorised'} · ${n.deg} links</span></div>`)
    .nodeVisibility(n => reg.isVisible(n))
    .linkVisibility(l => reg.isVisible(l.source) && reg.isVisible(l.target))
    .linkDirectionalParticles(0)
    .onNodeClick(n => emit('select', n.id))
    .onBackgroundClick(() => emit('select', null))
    .cooldownTicks(260)
  graph.d3Force('charge').strength(n => (n.ghost ? -14 : -70))
  graph.d3Force('link').distance(l => (l.type.startsWith('moaie') ? 26 : 46)).strength(0.25)
  graph.d3Force('center', null)
  graph.d3Force('facets', alpha => {
    for (const n of nodes) {
      if (n.x === undefined) continue
      const t = target(n)
      const k = (n.ghost ? 0.03 : 0.12) * alpha
      n.vx += (t.x - n.x) * k
      n.vy += (t.y - (n.y || 0)) * k * 0.7
      n.vz += (t.z - (n.z || 0)) * k
    }
  })
  graph.cameraPosition({ x: 0, y: 150, z: 430 })
  addFacetFrame(THREE, graph.scene())
  restyle()
  graph.onEngineStop(() => { if (reg.selId.value) focus(reg.selId.value) })

  resizeObserver = new ResizeObserver(() => {
    if (graph && stageEl.value) graph.width(stageEl.value.clientWidth).height(stageEl.value.clientHeight)
  })
  resizeObserver.observe(stageEl.value)
}

watch([reg.hidden, reg.showGhosts, reg.query], () => {
  if (!graph) return
  graph.nodeVisibility(n => reg.isVisible(n))
  graph.linkVisibility(l => reg.isVisible(l.source) && reg.isVisible(l.target))
  restyle()
})
watch(reg.selId, id => {
  restyle()
  focus(id)
})

defineExpose({ fit: () => focus(null) })

onMounted(init)
onUnmounted(() => {
  resizeObserver?.disconnect()
  if (graph) {
    graph.pauseAnimation()
    graph._destructor?.()
    graph = null
  }
})
</script>

<style scoped>
.stage3d { position: absolute; inset: 0; background: #000; overflow: hidden; }
.stage3d :deep(canvas) { outline: none; }
.stage3d :deep(.tip3d) { background: #000; border: 1px solid rgba(255, 255, 255, 0.3); padding: 4px 7px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #fff; }
.stage3d :deep(.tip3d span) { color: rgba(255, 255, 255, 0.55); font-size: 10px; }
</style>
