<template>
  <div ref="stageEl" class="stage">
    <svg ref="svgEl" role="img" aria-label="Force-directed graph of registry notes grouped by MOAIE facet"></svg>
    <div v-if="tip" class="tip" :style="{ left: tip.x + 'px', top: tip.y + 'px' }">
      <div class="t">{{ tip.node.title }}</div>
      <div class="m">{{ tipMeta(tip.node) }}</div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { FKEY, GHOST_COLOR, colorOf, isInstrument, linkColor, radiusOf, COLUMN_LETTER } from '~/composables/useSomRegistry'

const props = defineProps({ reg: { type: Object, required: true } })
const emit = defineEmits(['select'])
const reg = props.reg

const svgEl = ref(null)
const stageEl = ref(null)
const tip = ref(null)

let d3 = null
let sim = null
let zoom = null
let resizeObserver = null
let sel = {}
let W = 800
let H = 560
let nodes = []
let links = []

const colX = key => {
  const cols = reg.columns.value
  const i = cols.indexOf(key)
  const n = Math.max(cols.length, 2)
  return ((i < 0 ? (n - 1) / 2 : i) / (n - 1)) * (W * 0.8) + W * 0.1
}
const targetX = n => (n.xKeys.length ? n.xKeys.reduce((s, k) => s + colX(k), 0) / n.xKeys.length : n.ghost ? W / 2 : colX('2obj'))
const targetY = n => (n.ghost ? H * 0.5 : isInstrument(n) ? H * 0.64 : H * 0.36)
const idOf = e => (typeof e === 'object' ? e.id : e)

function tipMeta(n) {
  if (n.ghost) return `pending note · ${n.indeg} incoming links`
  return `${n.category || 'uncategorised'} · ${n.type || n.kind || ''} · ${n.deg} links`
}

function draw() {
  // The simulation works on copies: the shared graph in `reg` stays untouched by d3.
  nodes = reg.nodes.value.map(n => ({ ...n }))
  links = reg.links.value.map(l => ({ ...l }))
  const svg = d3.select(svgEl.value)
  svg.selectAll('*').remove()
  const root = svg.append('g')
  const axisG = root.append('g').attr('class', 'axis')
  const linkG = root.append('g')
  const nodeG = root.append('g')
  const labelG = root.append('g')
  const short = (s, k = 26) => (s.length > k ? s.slice(0, k - 1) + '…' : s)

  sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(l => (l.type.startsWith('moaie') ? 24 : 34)).strength(0.3))
    .force('charge', d3.forceManyBody().strength(n => (n.ghost ? -9 : -42)).distanceMax(320))
    .force('collide', d3.forceCollide().radius(n => radiusOf(n) + 1.5))

  sel.link = linkG.selectAll('line').data(links).join('line')
    .attr('class', d => 'lk' + (d.type.startsWith('moaie') ? ' moaie' : ''))
    .style('--c', d => linkColor(d.type))
  sel.node = nodeG.selectAll('circle').data(nodes).join('circle')
    .attr('class', d => 'nd' + (d.ghost ? ' ghost' : '') + (isInstrument(d) ? ' indiv' : ''))
    .attr('r', radiusOf)
    .style('--c', colorOf)
    .attr('tabindex', d => (d.ghost ? null : 0))
    .attr('role', 'button')
    .attr('aria-label', d => `${d.title}, ${d.ghost ? 'pending note' : d.category || 'uncategorised'}`)
    .on('click', (e, d) => { e.stopPropagation(); emit('select', d.id) })
    .on('keydown', (e, d) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); emit('select', d.id) }
    })
    .on('mouseenter', showTip)
    .on('mousemove', showTip)
    .on('mouseleave', () => { tip.value = null })
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.25).restart(); d.fx = d.x; d.fy = d.y })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y })
      .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null }))
  // Instruments get a small core so a hollow ring with a dot reads as "tuple of facets".
  sel.core = nodeG.selectAll('circle.core').data(nodes.filter(isInstrument)).join('circle')
    .attr('class', 'core').attr('r', 1.8).style('--c', colorOf)
  sel.label = labelG.selectAll('text').data(nodes.filter(n => !n.ghost)).join('text')
    .attr('class', 'lb').attr('text-anchor', 'middle').text(d => short(d.title))
  sel.axis = axisG

  sim.on('tick', () => {
    sel.link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y)
    sel.node.attr('cx', d => d.x).attr('cy', d => d.y)
    sel.core.attr('cx', d => d.x).attr('cy', d => d.y)
    sel.label.attr('x', d => d.x).attr('y', d => d.y - radiusOf(d) - 4)
  })

  // Labels keep a constant on-screen size while the graph zooms.
  zoom = d3.zoom().scaleExtent([0.3, 6]).on('zoom', e => {
    root.attr('transform', e.transform)
    sel.label.style('font-size', `${10 / e.transform.k}px`).style('stroke-width', `${3 / e.transform.k}px`)
    sel.link.style('stroke-width', null)
  })
  svg.call(zoom).on('dblclick.zoom', null)
  svg.on('click', () => emit('select', null))

  resizeObserver = new ResizeObserver(() => resize())
  resizeObserver.observe(svgEl.value)
  resize()
  applyVisibility()
}

function drawAxes() {
  sel.axis.selectAll('*').remove()
  reg.columns.value.forEach(key => {
    const f = FKEY.get(key)
    sel.axis.append('line').attr('x1', colX(key)).attr('x2', colX(key)).attr('y1', 34).attr('y2', H - 8)
    sel.axis.append('text').attr('x', colX(key)).attr('y', 22).attr('text-anchor', 'middle')
      .text(key === 'other' ? 'OTHER' : `${f.short} ${f.label}`.toUpperCase())
  })
}

function resize() {
  const r = svgEl.value.getBoundingClientRect()
  W = Math.max(320, r.width)
  H = Math.max(320, r.height)
  d3.select(svgEl.value).attr('viewBox', `0 0 ${W} ${H}`)
  sim.force('x', d3.forceX(targetX).strength(n => (n.ghost ? (n.xKeys.length ? 0.04 : 0.012) : 0.12)))
  sim.force('y', d3.forceY(targetY).strength(n => (n.ghost ? 0.03 : 0.06)))
  drawAxes()
  sim.alpha(0.5).restart()
}

function applyVisibility() {
  if (!sel.node) return
  const id = reg.selId.value
  const nb = id ? new Set([id, ...(reg.adjacency.value.get(id) || [])]) : null
  const q = reg.query.value.trim()
  const touchesSel = l => idOf(l.source) === id || idOf(l.target) === id
  const dim = n => (nb && !nb.has(n.id)) || !reg.matches(n)
  sel.node.style('display', n => (reg.isVisible(n) ? null : 'none'))
    .classed('dim', dim).classed('sel', n => n.id === id)
  sel.core.style('display', n => (reg.isVisible(n) ? null : 'none')).classed('dim', dim)
  sel.label.style('display', n => (reg.isVisible(n) ? null : 'none')).classed('dim', dim)
    .style('opacity', n => (n.deg >= 7 || (nb && nb.has(n.id)) || (q && reg.matches(n)) ? 1 : 0))
  sel.link.style('display', l => (reg.isVisible(l.source) && reg.isVisible(l.target) ? null : 'none'))
    .classed('on', l => !!nb && touchesSel(l))
    .classed('dim', l => !!nb && !touchesSel(l))
}

// Centre and zoom so the selected note and everything it links to fill the view.
function focus(id, animate = true) {
  if (!sel.node) return
  const ids = id ? new Set([id, ...(reg.adjacency.value.get(id) || [])]) : null
  const pts = nodes.filter(n => reg.isVisible(n) && (!ids || ids.has(n.id)) && Number.isFinite(n.x))
  if (!pts.length) return
  const x0 = Math.min(...pts.map(n => n.x - radiusOf(n)))
  const x1 = Math.max(...pts.map(n => n.x + radiusOf(n)))
  const y0 = Math.min(...pts.map(n => n.y - radiusOf(n)))
  const y1 = Math.max(...pts.map(n => n.y + radiusOf(n)))
  const pad = id ? 110 : 40
  const k = Math.max(0.35, Math.min(id ? 2.6 : 2, (W - pad * 2) / Math.max(x1 - x0, 30), (H - pad * 2) / Math.max(y1 - y0, 30)))
  const t = d3.zoomIdentity.translate(W / 2 - (k * (x0 + x1)) / 2, H / 2 - (k * (y0 + y1)) / 2).scale(k)
  const svg = d3.select(svgEl.value)
  if (animate) svg.transition().duration(700).call(zoom.transform, t)
  else svg.call(zoom.transform, t)
}

function showTip(e, d) {
  const r = stageEl.value.getBoundingClientRect()
  tip.value = { node: d, x: Math.min(r.width - 270, e.clientX - r.left + 14), y: Math.min(r.height - 60, e.clientY - r.top + 14) }
}

watch([reg.hidden, reg.showGhosts, reg.query, reg.selId], () => {
  applyVisibility()
})
watch(reg.selId, id => focus(id))

defineExpose({ fit: () => focus(null) })

onMounted(async () => {
  d3 = await import('d3')
  draw()
  // Let the layout settle, then frame the whole graph, or the deep-linked note.
  setTimeout(() => focus(reg.selId.value, false), 900)
})

onUnmounted(() => {
  sim?.stop()
  resizeObserver?.disconnect()
})
</script>

<style scoped>
.stage { position: absolute; inset: 0; background: #000; overflow: hidden; }
svg { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
svg:active { cursor: grabbing; }
.stage :deep(.axis text) { font-size: 10px; fill: rgba(255, 255, 255, 0.42); letter-spacing: 0.16em; }
.stage :deep(.axis line) { stroke: rgba(255, 255, 255, 0.14); stroke-dasharray: 2 5; }
.stage :deep(.lk), .stage :deep(.nd), .stage :deep(.axis line) { vector-effect: non-scaling-stroke; }
.stage :deep(.lk) { stroke: #fff; stroke-opacity: 0.16; stroke-width: 0.6; }
.stage :deep(.lk.moaie) { stroke: var(--c); stroke-opacity: 0.55; stroke-width: 1; }
.stage :deep(.lk.on) { stroke: var(--c); stroke-opacity: 0.95; stroke-width: 1.4; }
.stage :deep(.nd) { fill: #000; stroke: var(--c); stroke-width: 1.3; cursor: pointer; }
.stage :deep(.nd.ghost) { stroke: #6a6a6a; stroke-dasharray: 2 2; stroke-width: 1; }
.stage :deep(.nd.indiv) { stroke-width: 1.9; }
.stage :deep(.core) { fill: var(--c); pointer-events: none; }
.stage :deep(.nd:hover) { fill: rgba(255, 255, 255, 0.12); }
.stage :deep(.nd:focus-visible) { outline: none; stroke: #fff; stroke-width: 2.4; }
.stage :deep(.nd.dim), .stage :deep(.core.dim), .stage :deep(.lb.dim), .stage :deep(.lk.dim) { opacity: 0.1; }
.stage :deep(.nd.sel) { stroke: #fff; stroke-width: 2.6; fill: rgba(255, 255, 255, 0.16); }
.stage :deep(.lb) { font-size: 10px; fill: rgba(255, 255, 255, 0.9); paint-order: stroke; stroke: #000; stroke-width: 3px; stroke-linejoin: round; pointer-events: none; letter-spacing: 0.02em; }
.tip { position: absolute; pointer-events: none; background: #000; border: 1px solid rgba(255, 255, 255, 0.3); padding: 5px 8px; font-size: 11px; max-width: 260px; z-index: 3; }
.tip .t { color: #fff; }
.tip .m { color: rgba(255, 255, 255, 0.55); font-size: 10px; }
</style>
