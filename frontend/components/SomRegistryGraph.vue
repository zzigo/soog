<template>
  <div class="srg">
    <header class="srg-head">
      <div>
        <h1>SOM registry</h1>
        <p class="sub">
          Notes of the Speculative Organology registry and the links between them.
          Columns are the MOAIE facets, solid dots are classes, rings are instruments.
        </p>
      </div>
      <div class="stats" aria-live="polite">
        <span><b>{{ real.length }}</b> notes</span>
        <span><b>{{ instrumentCount }}</b> instruments</span>
        <span><b>{{ ghosts.length }}</b> pending</span>
        <span><b>{{ links.length }}</b> links</span>
      </div>
    </header>

    <div class="bar">
      <div class="legend" role="group" aria-label="Filter by facet">
        <button
          v-for="f in facetChips"
          :key="f.key"
          type="button"
          class="chip"
          :aria-pressed="String(!hidden.has(f.key))"
          @click="toggle(f.key)"
        >
          <span class="dot" :style="{ background: `var(${f.css})` }"></span>{{ f.short }} {{ f.label }}
          <span class="n">{{ f.count }}</span>
        </button>
        <button type="button" class="chip" :aria-pressed="String(!hidden.has('indiv'))" @click="toggle('indiv')">
          <span class="dot ring"></span>Instruments (rings)<span class="n">{{ instrumentCount }}</span>
        </button>
      </div>
      <div class="legend" role="group" aria-label="Filter instruments by origin">
        <button
          v-for="o in originChips"
          :key="o.key"
          type="button"
          class="chip"
          :aria-pressed="String(!hidden.has('origin:' + o.key))"
          @click="toggle('origin:' + o.key)"
        >
          {{ o.label }}<span class="n">{{ o.count }}</span>
        </button>
      </div>
    </div>

    <div class="bar">
      <div class="ghost-note">
        <button type="button" class="btn" :aria-pressed="String(showGhosts)" @click="showGhosts = !showGhosts">
          {{ showGhosts ? 'Hide' : 'Show' }} pending notes ({{ ghosts.length }})
        </button>
        <span>Grey dots are pending notes: other notes link to them, but they do not exist in the registry yet.</span>
      </div>
      <div class="tools">
        <input v-model="query" type="search" placeholder="Find a note" aria-label="Find a note" />
        <button type="button" class="btn" :aria-pressed="String(tableOn)" @click="tableOn = !tableOn">Table view</button>
        <button type="button" class="btn" @click="reset">Reset view</button>
      </div>
    </div>

    <div class="main">
      <div ref="stageEl" class="stage">
        <svg ref="svgEl" role="img" aria-label="Force-directed graph of registry notes grouped by MOAIE facet"></svg>
        <div v-if="tip" class="tip" :style="{ left: tip.x + 'px', top: tip.y + 'px' }">
          <div class="t">{{ tip.node.title }}</div>
          <div class="m">{{ tipMeta(tip.node) }}</div>
        </div>
        <div v-if="loading" class="status">Loading registry…</div>
        <div v-if="error" class="status error">{{ error }}</div>
      </div>

      <aside aria-live="polite">
        <template v-if="selected">
          <h2>{{ selected.title }}</h2>
          <div class="kv">
            <span>{{ selected.ghost ? 'Pending note' : facetLabel(selected) }}</span>
            <span v-if="selected.kind">{{ selected.kind }}</span>
            <span v-if="selected.type">{{ selected.type }}</span>
            <span v-if="selected.origin">{{ selected.origin }}</span>
          </div>
          <div class="kv">
            <span>{{ selected.deg }} links</span>
            <span v-if="!selected.ghost">{{ selected.n_crit }} criteria recorded</span>
          </div>
          <p v-if="selected.def">{{ selected.def }}</p>
          <p v-else-if="selected.ghost">Other notes link here, but the note does not exist in the registry yet.</p>
          <p v-else>No definition (def) in the frontmatter.</p>

          <template v-if="selected.crit">
            <h3>Criteria</h3>
            <dl class="crit">
              <template v-for="(value, key) in selected.crit" :key="key">
                <dt>{{ key }}</dt>
                <dd v-if="key === 'O.elec'">
                  <ol class="elec">
                    <li v-for="step in elecSteps(value)" :key="step.label" :class="{ on: step.on }">
                      <span class="lbl">{{ step.label }}</span><span class="val">{{ step.value }}</span>
                    </li>
                  </ol>
                </dd>
                <dd v-else>{{ formatCrit(value) }}</dd>
              </template>
            </dl>
          </template>

          <h3>Linked to</h3>
          <ul class="nl">
            <li v-for="n in neighborList" :key="n.id">
              <button type="button" class="nb" @click="select(n.id)">
                <span class="dot" :style="{ background: colorOf(n) }"></span><span>{{ n.title }}</span>
                <span class="n">{{ n.ghost ? n.indeg + ' in' : n.deg }}</span>
              </button>
            </li>
          </ul>
        </template>
        <template v-else>
          <h2>How to read it</h2>
          <p>Columns are the MOAIE facets. Instruments sit between the facets their links reach. Dot size is the number of links. Select a note to see what it connects to.</p>
          <h3>Most connected notes</h3>
          <ul class="nl">
            <li v-for="n in hubs" :key="n.id">
              <button type="button" class="nb" @click="select(n.id)">
                <span class="dot" :style="{ background: colorOf(n) }"></span><span>{{ n.title }}</span><span class="n">{{ n.deg }}</span>
              </button>
            </li>
          </ul>
          <h3>Most linked pending notes</h3>
          <ul class="nl">
            <li v-for="n in topMissing" :key="n.id">
              <button type="button" class="nb" @click="select(n.id)">
                <span class="dot" :style="{ background: colorOf(n) }"></span><span>{{ n.title }}</span><span class="n">{{ n.indeg }} in</span>
              </button>
            </li>
          </ul>
        </template>
      </aside>
    </div>

    <div v-if="tableOn" class="tablewrap">
      <table>
        <thead>
          <tr>
            <th>Note</th><th>Facet</th><th>Kind</th><th>Type</th><th>Origin</th>
            <th class="num">Links</th><th class="num">Criteria</th><th>Definition</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="n in tableRows" :key="n.id">
            <td>
              <button type="button" class="nb" @click="select(n.id)">
                <span class="dot" :style="{ background: colorOf(n) }"></span><span>{{ n.title }}</span>
              </button>
            </td>
            <td>{{ n.category || '—' }}</td>
            <td>{{ n.kind }}</td>
            <td>{{ n.type }}</td>
            <td>{{ n.origin }}</td>
            <td class="num">{{ n.deg }}</td>
            <td class="num">{{ n.n_crit }}</td>
            <td class="def">{{ n.def }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <footer v-if="meta" class="foot">
      <span>
        registry
        <a :href="`https://github.com/zzigo/som-registry/commit/${meta.commit}`" target="_blank" rel="noopener">
          {{ meta.commit ? meta.commit.slice(0, 7) : 'unknown commit' }}
        </a>
      </span>
      <span>built {{ meta.built_at }}</span>
      <span>{{ meta.counts.real }} notes · {{ meta.counts.ghost }} pending · {{ meta.counts.links }} links</span>
      <span v-if="meta.skipped && meta.skipped.length" :title="meta.skipped.map(s => s.file).join('\n')">
        {{ meta.skipped.length }} skipped (broken frontmatter)
      </span>
    </footer>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, shallowRef, watch } from 'vue'
import { useApi } from '~/composables/useApi'

const FACETS = [
  { key: '1mat', label: 'Materials', short: 'M', css: '--srg-mat' },
  { key: '2obj', label: 'Objects', short: 'O', css: '--srg-obj' },
  { key: '3agn', label: 'Agents', short: 'A', css: '--srg-agn' },
  { key: '4int', label: 'Interfaces', short: 'I', css: '--srg-int' },
  { key: '5env', label: 'Environments', short: 'E', css: '--srg-env' },
  { key: '0pri', label: 'Principles', short: 'P', css: '--srg-pri' },
  { key: 'other', label: 'Dictionary, uncategorised', short: '·', css: '--srg-other' }
]
const FKEY = new Map(FACETS.map(f => [f.key, f]))
const MOAIE_COLUMN = { M: '1mat', O: '2obj', A: '3agn', I: '4int', E: '5env' }
const ORIGINS = ['traditional', 'contemporary', 'fictional', 'generated']
const ELEC_STEPS = ['none', 'material', 'component', 'circuit']

const { apiBase } = useApi()
const svgEl = ref(null)
const stageEl = ref(null)
const loading = ref(true)
const error = ref('')
const meta = ref(null)
const nodes = shallowRef([])
const links = shallowRef([])
const hidden = ref(new Set())
const showGhosts = ref(true)
const query = ref('')
const selId = ref(null)
const tableOn = ref(false)
const tip = ref(null)

const real = computed(() => nodes.value.filter(n => !n.ghost))
const ghosts = computed(() => nodes.value.filter(n => n.ghost))
const byId = computed(() => new Map(nodes.value.map(n => [n.id, n])))
const isInstrument = n => !n.ghost && (n.type === 'instrument' || n.kind === 'individual')
const instrumentCount = computed(() => real.value.filter(isInstrument).length)
const originOf = n => (ORIGINS.includes(n.origin) ? n.origin : 'unset')

const facetChips = computed(() => FACETS
  .map(f => ({ ...f, count: real.value.filter(n => n.facet === f.key).length }))
  .filter(f => f.count > 0 || ['1mat', '2obj', '3agn', '4int', '5env'].includes(f.key)))
const originChips = computed(() => [...ORIGINS, 'unset'].map(key => ({
  key,
  label: key === 'unset' ? 'origin unset' : key,
  count: real.value.filter(n => isInstrument(n) && originOf(n) === key).length
})))

const selected = computed(() => (selId.value ? byId.value.get(selId.value) : null))
const adjacency = computed(() => {
  const map = new Map(nodes.value.map(n => [n.id, new Set()]))
  links.value.forEach(l => {
    const s = l.source.id ?? l.source
    const t = l.target.id ?? l.target
    map.get(s)?.add(t)
    map.get(t)?.add(s)
  })
  return map
})
const neighborList = computed(() => {
  if (!selected.value) return []
  return [...(adjacency.value.get(selected.value.id) || [])]
    .map(id => byId.value.get(id)).filter(Boolean).sort((a, b) => b.deg - a.deg)
})
const hubs = computed(() => [...real.value].sort((a, b) => b.deg - a.deg).slice(0, 8))
const topMissing = computed(() => [...ghosts.value].sort((a, b) => b.indeg - a.indeg).slice(0, 10))
const matches = n => {
  const q = query.value.trim().toLowerCase()
  return !q || n.title.toLowerCase().includes(q) || n.id.toLowerCase().includes(q)
}
const tableRows = computed(() => real.value.filter(matches)
  .sort((a, b) => (a.category || 'zz').localeCompare(b.category || 'zz') || b.deg - a.deg))

function facetOf(n) {
  if (n.ghost) return 'ghost'
  return FKEY.has(n.category) ? n.category : 'other'
}
function facetLabel(n) {
  const f = FKEY.get(n.facet)
  return n.category ? `${f.short} ${f.label} (${n.category})` : 'Uncategorised'
}
function colorOf(n) {
  return n.ghost ? 'var(--srg-ghost)' : `var(${FKEY.get(n.facet).css})`
}
function tipMeta(n) {
  if (n.ghost) return `pending note · ${n.indeg} incoming links`
  return `${n.category || 'uncategorised'} · ${n.type || n.kind || ''} · ${n.deg} links`
}
function formatCrit(value) {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}
// O.elec may arrive as one level, a list of four values or a map keyed by level.
function elecSteps(value) {
  if (Array.isArray(value)) return ELEC_STEPS.map((label, i) => ({ label, value: formatCrit(value[i]), on: false }))
  if (value && typeof value === 'object') return ELEC_STEPS.map(label => ({ label, value: formatCrit(value[label]), on: false }))
  return ELEC_STEPS.map(label => ({ label, value: label === String(value) ? '●' : '', on: label === String(value) }))
}
function toggle(key) {
  const next = new Set(hidden.value)
  next.has(key) ? next.delete(key) : next.add(key)
  hidden.value = next
}
function select(id) {
  selId.value = id
}

let d3 = null
let sim = null
let zoom = null
let resizeObserver = null
let sel = {}
let W = 800
let H = 560
let columns = []

const colX = key => {
  const i = columns.indexOf(key)
  const n = Math.max(columns.length, 2)
  return ((i < 0 ? (n - 1) / 2 : i) / (n - 1)) * (W * 0.8) + W * 0.1
}
const radius = n => (n.ghost ? 2.6 : 4.5 + Math.sqrt(n.deg) * 1.35)
const targetX = n => (n.xKeys.length ? n.xKeys.reduce((s, k) => s + colX(k), 0) / n.xKeys.length : n.ghost ? W / 2 : colX('2obj'))
const targetY = n => (n.ghost ? H * 0.5 : isInstrument(n) ? H * 0.64 : H * 0.36)

function isVisible(n) {
  if (n.ghost) return showGhosts.value
  if (hidden.value.has(n.facet)) return false
  if (isInstrument(n) && (hidden.value.has('indiv') || hidden.value.has('origin:' + originOf(n)))) return false
  return true
}

function prepare(data) {
  const ns = data.nodes.map(n => ({ ...n, facet: facetOf(n), deg: 0, indeg: 0, xKeys: [] }))
  const map = new Map(ns.map(n => [n.id, n]))
  const ls = data.links.filter(l => map.has(l.source) && map.has(l.target)).map(l => ({ ...l }))
  ls.forEach(l => {
    const s = map.get(l.source)
    const t = map.get(l.target)
    s.deg++
    t.deg++
    t.indeg++
    // Instruments are drawn between the facet columns their resolved links reach.
    if (isInstrument(s) && !t.ghost && MOAIE_COLUMN[FKEY.get(t.facet)?.short]) s.xKeys.push(t.facet)
    // A pending note reached through moaie:X sits in column X.
    if (t.ghost && l.type.startsWith('moaie:')) t.xKeys.push(MOAIE_COLUMN[l.type.slice(6)])
  })
  ns.forEach(n => {
    if (!n.ghost && !isInstrument(n)) n.xKeys = [n.facet]
  })
  const present = new Set(ns.filter(n => !n.ghost).map(n => n.facet))
  columns = FACETS.map(f => f.key).filter(k => ['1mat', '2obj', '3agn', '4int', '5env'].includes(k) || present.has(k))
  nodes.value = ns
  links.value = ls
}

function draw() {
  const svg = d3.select(svgEl.value)
  svg.selectAll('*').remove()
  const root = svg.append('g')
  const axisG = root.append('g').attr('class', 'axis')
  const linkG = root.append('g')
  const nodeG = root.append('g')
  const labelG = root.append('g')
  const short = (s, k = 26) => (s.length > k ? s.slice(0, k - 1) + '…' : s)

  sim = d3.forceSimulation(nodes.value)
    .force('link', d3.forceLink(links.value).id(d => d.id)
      .distance(l => (l.type.startsWith('moaie') ? 24 : 34)).strength(0.3))
    .force('charge', d3.forceManyBody().strength(n => (n.ghost ? -9 : -42)).distanceMax(320))
    .force('collide', d3.forceCollide().radius(n => radius(n) + 1.5))

  sel.link = linkG.selectAll('line').data(links.value).join('line')
    .attr('class', d => 'lk' + (d.type.startsWith('moaie') ? ' moaie' : ''))
  sel.node = nodeG.selectAll('circle').data(nodes.value).join('circle')
    .attr('class', d => 'nd' + (d.ghost ? ' ghost' : '') + (isInstrument(d) ? ' indiv' : ''))
    .attr('r', radius)
    .style('fill', colorOf)
    .style('--ring', colorOf)
    .attr('tabindex', d => (d.ghost ? null : 0))
    .attr('role', d => (d.ghost ? null : 'button'))
    .attr('aria-label', d => `${d.title}, ${d.ghost ? 'pending note' : d.category || 'uncategorised'}`)
    .on('click', (e, d) => { e.stopPropagation(); select(d.id) })
    .on('keydown', (e, d) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(d.id) }
    })
    .on('mouseenter', (e, d) => showTip(e, d))
    .on('mousemove', (e, d) => showTip(e, d))
    .on('mouseleave', () => { tip.value = null })
    .call(d3.drag()
      .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.25).restart(); d.fx = d.x; d.fy = d.y })
      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y })
      .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null }))
  sel.label = labelG.selectAll('text').data(nodes.value.filter(n => !n.ghost)).join('text')
    .attr('class', 'lb').attr('text-anchor', 'middle').text(d => short(d.title))
  sel.axis = axisG

  sim.on('tick', () => {
    sel.link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y)
    sel.node.attr('cx', d => d.x).attr('cy', d => d.y)
    sel.label.attr('x', d => d.x).attr('y', d => d.y - radius(d) - 4)
  })

  zoom = d3.zoom().scaleExtent([0.4, 6]).on('zoom', e => root.attr('transform', e.transform))
  svg.call(zoom).on('dblclick.zoom', null)
  svg.on('click', () => select(null))

  resizeObserver = new ResizeObserver(() => resize())
  resizeObserver.observe(svgEl.value)
  resize()
  applyVisibility()
}

function drawAxes() {
  sel.axis.selectAll('*').remove()
  columns.forEach(key => {
    const f = FKEY.get(key)
    sel.axis.append('line').attr('x1', colX(key)).attr('x2', colX(key)).attr('y1', 28).attr('y2', H - 8)
    sel.axis.append('text').attr('x', colX(key)).attr('y', 18).attr('text-anchor', 'middle')
      .text(key === 'other' ? 'Other' : `${f.short} ${f.label}`)
  })
}

function resize() {
  const r = svgEl.value.getBoundingClientRect()
  W = Math.max(320, r.width)
  H = Math.max(360, r.height)
  d3.select(svgEl.value).attr('viewBox', `0 0 ${W} ${H}`)
  sim.force('x', d3.forceX(targetX).strength(n => (n.ghost ? (n.xKeys.length ? 0.04 : 0.012) : 0.12)))
  sim.force('y', d3.forceY(targetY).strength(n => (n.ghost ? 0.03 : 0.06)))
  drawAxes()
  sim.alpha(0.5).restart()
}

function applyVisibility() {
  if (!sel.node) return
  const nb = selId.value ? new Set([selId.value, ...(adjacency.value.get(selId.value) || [])]) : null
  const q = query.value.trim()
  const touchesSel = l => l.source.id === selId.value || l.target.id === selId.value
  sel.node.style('display', n => (isVisible(n) ? null : 'none'))
    .classed('dim', n => (nb && !nb.has(n.id)) || !matches(n))
    .classed('sel', n => n.id === selId.value)
  sel.label.style('display', n => (isVisible(n) ? null : 'none'))
    .classed('dim', n => (nb && !nb.has(n.id)) || !matches(n))
    .style('opacity', n => (n.deg >= 7 || (nb && nb.has(n.id)) || (q && matches(n)) ? 1 : 0))
  sel.link.style('display', l => (isVisible(l.source) && isVisible(l.target) ? null : 'none'))
    .classed('on', l => !!nb && touchesSel(l))
    .classed('dim', l => !!nb && !touchesSel(l))
}

function showTip(e, d) {
  const r = stageEl.value.getBoundingClientRect()
  tip.value = {
    node: d,
    x: Math.min(r.width - 270, e.clientX - r.left + 14),
    y: Math.min(r.height - 60, e.clientY - r.top + 14)
  }
}

function reset() {
  selId.value = null
  query.value = ''
  hidden.value = new Set()
  showGhosts.value = true
  if (zoom) d3.select(svgEl.value).transition().duration(300).call(zoom.transform, d3.zoomIdentity)
}

watch([hidden, showGhosts, query, selId], applyVisibility)

onMounted(async () => {
  try {
    const [mod, res] = await Promise.all([import('d3'), fetch(`${apiBase.value}/somap/registry`)])
    d3 = mod
    const data = await res.json()
    if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`)
    meta.value = data.meta
    prepare(data)
    loading.value = false
    draw()
  } catch (e) {
    loading.value = false
    error.value = `Could not load the registry graph: ${e.message}`
  }
})

onUnmounted(() => {
  sim?.stop()
  resizeObserver?.disconnect()
})
</script>

<style scoped>
.srg {
  color-scheme: light;
  --srg-bg: #f4f5f6;
  --srg-surface: #fcfcfb;
  --srg-ink: #0b0b0b;
  --srg-ink-2: #52514e;
  --srg-ink-3: #6b6a65;
  --srg-line: #dcdcd8;
  --srg-focus: #0072b2;
  /* Okabe-Ito hues: validated for deuteranopia, protanopia and tritanopia (worst pair ΔE 7.6, secondary encoding by column). */
  --srg-mat: #0072b2;
  --srg-obj: #d55e00;
  --srg-agn: #009e73;
  --srg-int: #e69f00;
  --srg-env: #cc79a7;
  --srg-pri: #56b4e9;
  --srg-other: #6b6a66;
  --srg-ghost: #b9b8b3;
  --srg-link: #8d8c87;
  background: var(--srg-bg);
  color: var(--srg-ink);
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
  font-size: 14px;
  line-height: 1.5;
  padding: 20px 16px 28px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 100vh;
  box-sizing: border-box;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme='light'])) .srg {
    color-scheme: dark;
    --srg-bg: #121211;
    --srg-surface: #1a1a19;
    --srg-ink: #f0efec;
    --srg-ink-2: #c3c2b7;
    --srg-ink-3: #9a9991;
    --srg-line: #34332f;
    --srg-focus: #56b4e9;
    --srg-mat: #2d8fd5;
    --srg-other: #a09f97;
    --srg-ghost: #5f5e59;
    --srg-link: #6e6d67;
  }
}
:root[data-theme='dark'] .srg {
  color-scheme: dark;
  --srg-bg: #121211;
  --srg-surface: #1a1a19;
  --srg-ink: #f0efec;
  --srg-ink-2: #c3c2b7;
  --srg-ink-3: #9a9991;
  --srg-line: #34332f;
  --srg-focus: #56b4e9;
  --srg-mat: #2d8fd5;
  --srg-other: #a09f97;
  --srg-ghost: #5f5e59;
  --srg-link: #6e6d67;
}
.srg * { box-sizing: border-box; }
.srg-head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 6px 24px; }
h1 { font-size: 22px; font-weight: 600; margin: 0; }
.sub { color: var(--srg-ink-2); margin: 2px 0 0; max-width: 70ch; }
.stats, .foot { display: flex; flex-wrap: wrap; gap: 6px 18px; font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: var(--srg-ink-2); font-variant-numeric: tabular-nums; }
.stats b { color: var(--srg-ink); font-weight: 500; }
.foot a { color: var(--srg-ink); }
.bar { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; justify-content: space-between; }
.legend { display: flex; flex-wrap: wrap; gap: 6px; }
button, input { font: inherit; color: inherit; }
.chip { display: inline-flex; align-items: center; gap: 7px; padding: 4px 10px 4px 8px; border: 1px solid var(--srg-line); background: var(--srg-surface); border-radius: 999px; cursor: pointer; font-size: 12.5px; }
.chip .n, .nb .n { font-family: ui-monospace, Menlo, monospace; font-size: 11.5px; color: var(--srg-ink-3); font-variant-numeric: tabular-nums; }
.chip[aria-pressed='false'] { opacity: 0.45; text-decoration: line-through; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; flex: none; }
.dot.ring { border: 2px solid var(--srg-ink-2); background: transparent; }
.chip:focus-visible, .btn:focus-visible, input:focus-visible, .nb:focus-visible, .foot a:focus-visible { outline: 2px solid var(--srg-focus); outline-offset: 2px; }
.ghost-note { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; color: var(--srg-ink-2); font-size: 12.5px; max-width: 760px; }
.tools { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
input[type='search'] { padding: 5px 10px; border: 1px solid var(--srg-line); background: var(--srg-surface); border-radius: 6px; width: 190px; max-width: 100%; }
.btn { padding: 5px 11px; border: 1px solid var(--srg-line); background: var(--srg-surface); border-radius: 6px; cursor: pointer; font-size: 12.5px; }
.btn[aria-pressed='true'] { border-color: var(--srg-ink-2); }
.main { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 14px; align-items: start; }
.stage { background: var(--srg-surface); border: 1px solid var(--srg-line); border-radius: 8px; overflow: hidden; position: relative; min-width: 0; }
svg { display: block; width: 100%; height: min(72vh, 700px); min-height: 440px; cursor: grab; touch-action: none; }
svg:active { cursor: grabbing; }
.stage :deep(.axis text) { font-size: 11px; fill: var(--srg-ink-3); letter-spacing: 0.06em; text-transform: uppercase; }
.stage :deep(.axis line) { stroke: var(--srg-line); stroke-dasharray: 2 4; }
.stage :deep(.lk) { stroke: var(--srg-link); stroke-opacity: 0.28; stroke-width: 0.8; }
.stage :deep(.lk.moaie) { stroke-opacity: 0.7; stroke-width: 1.6; }
.stage :deep(.lk.on) { stroke: var(--srg-ink); stroke-opacity: 0.75; }
.stage :deep(.nd) { stroke: var(--srg-surface); stroke-width: 2; cursor: pointer; }
.stage :deep(.nd.ghost) { fill: var(--srg-ghost); stroke-width: 1; }
.stage :deep(.nd.indiv) { stroke: var(--ring); stroke-width: 2.2; fill-opacity: 0.22; }
.stage :deep(.nd:focus-visible) { outline: none; stroke: var(--srg-focus); stroke-width: 3; }
.stage :deep(.nd.dim), .stage :deep(.lb.dim), .stage :deep(.lk.dim) { opacity: 0.12; }
.stage :deep(.nd.sel) { stroke: var(--srg-ink); stroke-width: 2.5; }
.stage :deep(.lb) { font-size: 11px; fill: var(--srg-ink); paint-order: stroke; stroke: var(--srg-surface); stroke-width: 3px; stroke-linejoin: round; pointer-events: none; }
.tip { position: absolute; pointer-events: none; background: var(--srg-surface); border: 1px solid var(--srg-line); border-radius: 6px; padding: 6px 9px; font-size: 12px; max-width: 260px; box-shadow: 0 2px 10px rgb(0 0 0 / 0.12); z-index: 3; }
.tip .t { font-weight: 600; }
.tip .m { font-family: ui-monospace, Menlo, monospace; font-size: 11px; color: var(--srg-ink-3); }
.status { position: absolute; inset: auto 12px 12px 12px; font-size: 12.5px; color: var(--srg-ink-2); }
.status.error { color: var(--srg-obj); }
aside { background: var(--srg-surface); border: 1px solid var(--srg-line); border-radius: 8px; padding: 14px; display: flex; flex-direction: column; gap: 10px; min-width: 0; max-height: min(72vh, 700px); overflow-y: auto; }
aside h2 { font-size: 15px; margin: 0; font-weight: 600; overflow-wrap: anywhere; }
aside h3 { font-size: 11.5px; margin: 6px 0 0; font-weight: 500; text-transform: uppercase; letter-spacing: 0.07em; color: var(--srg-ink-3); }
aside p { margin: 0; color: var(--srg-ink-2); overflow-wrap: anywhere; }
.kv { font-family: ui-monospace, Menlo, monospace; font-size: 11.5px; color: var(--srg-ink-3); display: flex; flex-wrap: wrap; gap: 4px 12px; }
.crit { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px; margin: 0; font-size: 12px; }
.crit dt { font-family: ui-monospace, Menlo, monospace; color: var(--srg-ink-3); }
.crit dd { margin: 0; overflow-wrap: anywhere; }
.elec { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 4px; }
.elec li { border: 1px solid var(--srg-line); border-radius: 4px; padding: 2px 4px; display: flex; flex-direction: column; font-size: 11px; }
.elec li.on { border-color: var(--srg-ink); font-weight: 600; }
.elec .lbl { color: var(--srg-ink-3); }
.nl { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }
.nb { display: flex; align-items: baseline; gap: 8px; width: 100%; text-align: left; padding: 3px 4px; border: 0; background: none; cursor: pointer; border-radius: 4px; overflow-wrap: anywhere; }
.nb:hover { background: var(--srg-bg); }
.nb .n { margin-left: auto; flex: none; }
.tablewrap { overflow-x: auto; background: var(--srg-surface); border: 1px solid var(--srg-line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--srg-line); vertical-align: top; }
th { font-weight: 500; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--srg-ink-3); white-space: nowrap; }
.num { font-family: ui-monospace, Menlo, monospace; font-variant-numeric: tabular-nums; text-align: right; }
td.def { color: var(--srg-ink-2); min-width: 260px; }
@media (max-width: 900px) {
  .main { grid-template-columns: minmax(0, 1fr); }
  svg { min-height: 380px; }
  aside { max-height: none; }
}
@media (prefers-reduced-motion: reduce) { .srg * { transition: none !important; } }
</style>
