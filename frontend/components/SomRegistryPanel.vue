<template>
  <div class="panel">
    <nav class="tabs" role="tablist" aria-label="Panel">
      <button v-for="t in TABS" :key="t.key" type="button" role="tab" :aria-selected="String(reg.tab.value === t.key)"
        :class="{ on: reg.tab.value === t.key }" @click="reg.tab.value = t.key">{{ t.label }}</button>
    </nav>

    <div class="body" aria-live="polite">
      <!-- NODE ---------------------------------------------------------- -->
      <template v-if="reg.tab.value === 'node'">
        <template v-if="sel">
          <header class="head">
            <h2>{{ sel.title }}</h2>
            <button type="button" class="x" aria-label="Clear selection" @click="reg.select(null)">×</button>
          </header>
          <p class="kv">
            <span class="sw" :style="{ borderColor: colorOf(sel) }"></span>
            <span>{{ sel.ghost ? 'pending note' : facetLabel(sel) }}</span>
            <span v-if="sel.kind">{{ sel.kind }}</span>
            <span v-if="sel.type">{{ sel.type }}</span>
            <span v-if="sel.origin">{{ sel.origin }}</span>
          </p>
          <p class="kv"><span>{{ sel.deg }} links</span><span v-if="!sel.ghost">{{ sel.n_crit }} criteria</span></p>
          <p v-if="sel.def" class="def">{{ sel.def }}</p>
          <p v-else-if="sel.ghost" class="def dim">Other notes link here; the note does not exist in the registry yet.</p>
          <p v-else class="def dim">No definition (def) in the frontmatter.</p>
          <button v-if="!sel.ghost" type="button" class="btn" @click="reg.tab.value = 'details'">Open details →</button>

          <template v-if="sel.crit">
            <h3>Criteria</h3>
            <dl class="crit">
              <template v-for="(value, key) in sel.crit" :key="key">
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

          <h3>Edges <span class="n">{{ edgeTotal }}</span></h3>
          <section v-for="g in reg.edges.value" :key="g.type" class="edge-group">
            <h4><span class="bar" :style="{ background: linkColor(g.type) }"></span>{{ g.label }}<span class="n">{{ g.items.length }}</span></h4>
            <ul>
              <li v-for="it in g.items" :key="g.type + it.node.id + it.dir">
                <button type="button" class="nb" @click="reg.select(it.node.id)">
                  <span class="arrow" :title="it.dir === 'out' ? 'this note links to it' : 'it links to this note'">{{ it.dir === 'out' ? '→' : '←' }}</span>
                  <span class="dot" :class="{ ghost: it.node.ghost }" :style="{ borderColor: colorOf(it.node) }"></span>
                  <span class="nm">{{ it.node.title }}</span>
                  <span class="n">{{ it.node.deg }}</span>
                </button>
              </li>
            </ul>
          </section>
        </template>

        <template v-else>
          <h2>How to read it</h2>
          <p class="def">Columns (or the pentagon, in 3D) are the five MOAIE facets. Instruments sit between the facets their links reach. Dot size is link count. Select a note: the view centres on it and its edges, and the Details tab shows its YAML and text.</p>
          <h3>Most connected notes</h3>
          <ul class="plain">
            <li v-for="n in hubs" :key="n.id">
              <button type="button" class="nb" @click="reg.select(n.id)">
                <span class="dot" :style="{ borderColor: colorOf(n) }"></span><span class="nm">{{ n.title }}</span><span class="n">{{ n.deg }}</span>
              </button>
            </li>
          </ul>
          <h3>Most linked pending notes</h3>
          <ul class="plain">
            <li v-for="n in topMissing" :key="n.id">
              <button type="button" class="nb" @click="reg.select(n.id)">
                <span class="dot ghost" :style="{ borderColor: colorOf(n) }"></span><span class="nm">{{ n.title }}</span><span class="n">{{ n.indeg }} in</span>
              </button>
            </li>
          </ul>
        </template>
      </template>

      <!-- DETAILS -------------------------------------------------------- -->
      <template v-else-if="reg.tab.value === 'details'">
        <p v-if="!sel" class="def dim">Select a note to read its YAML and text.</p>
        <template v-else>
          <header class="head"><h2>{{ sel.title }}</h2></header>
          <p v-if="sel.ghost" class="def dim">Pending note: other notes link here, but it has no file in the registry yet.</p>
          <template v-else>
          <p class="kv"><span>{{ reg.note.value?.file || sel.file }}</span></p>
          <p v-if="reg.noteLoading.value" class="def dim">Loading note…</p>
          <p v-else-if="reg.noteError.value" class="def err">Could not load the note: {{ reg.noteError.value }}</p>
          <template v-else-if="reg.note.value">
            <div class="wheels">
              <SomMoaieWheel title="Facet reach" :axes="reachAxes" :max="reachMax"
                :caption="reachCaption" />
              <SomMoaieWheel v-if="vectorAxes" title="MOAIE vector" :axes="vectorAxes" :max="1"
                caption="declared in moaie.vector" />
            </div>
            <div v-if="profileAxes" class="wheels one">
              <SomMoaieWheel title="Interface profile" :axes="profileAxes" :max="1" :size="250"
                caption="declared in interface_profile" />
            </div>
            <p v-if="!vectorAxes && !profileAxes" class="def dim">
              This note declares no numeric MOAIE vector or interface profile. The wheel above is measured from the graph: how many linked notes sit in each facet.
            </p>

            <h3>YAML</h3>
            <pre class="yaml" @click="onYamlClick"><template v-for="(ln, i) in yaml" :key="i"><span class="ln" :class="{ empty: ln.empty && ln.key, plain: ln.plain }"><span class="ind">{{ ln.indent }}</span><span v-if="ln.bullet" class="pun">- </span><span v-if="ln.key !== null" class="k">{{ ln.key }}</span><span v-if="ln.key !== null" class="pun">:</span><template v-for="(s, j) in ln.segs" :key="j"><span v-if="s.k === 'link'" class="v link" :class="{ miss: !linkId(s.target) }" role="button" tabindex="0" :data-id="linkId(s.target)" @keydown.enter="goTo(linkId(s.target))">{{ s.t }}</span><span v-else class="v" :class="s.k">{{ s.t }}</span></template></span></template></pre>

            <h3>Note <span v-if="reg.note.value.truncated" class="n">truncated at 300k characters</span></h3>
            <div v-if="bodyHtml" class="md" @click="onBodyClick" v-html="bodyHtml"></div>
            <p v-else class="def dim">The note has no text below the frontmatter.</p>
          </template>
          </template>
        </template>
      </template>

      <!-- FILTERS -------------------------------------------------------- -->
      <template v-else>
        <input v-model="reg.query.value" type="search" class="search" placeholder="Find a note" aria-label="Find a note"
          @keydown.enter.prevent="focusFirstMatch" />
        <h3>Facets</h3>
        <div class="chips" role="group" aria-label="Filter by facet">
          <button v-for="f in facetChips" :key="f.key" type="button" class="chip" :aria-pressed="String(!reg.hidden.value.has(f.key))" @click="reg.toggle(f.key)">
            <span class="dot" :style="{ borderColor: f.color }"></span>{{ f.short }} {{ f.label }}<span class="n">{{ f.count }}</span>
          </button>
          <button type="button" class="chip" :aria-pressed="String(!reg.hidden.value.has('indiv'))" @click="reg.toggle('indiv')">
            <span class="dot ring"></span>Instruments<span class="n">{{ reg.instrumentCount.value }}</span>
          </button>
        </div>
        <h3>Instrument origin</h3>
        <div class="chips" role="group" aria-label="Filter instruments by origin">
          <button v-for="o in originChips" :key="o.key" type="button" class="chip" :aria-pressed="String(!reg.hidden.value.has('origin:' + o.key))" @click="reg.toggle('origin:' + o.key)">
            {{ o.label }}<span class="n">{{ o.count }}</span>
          </button>
        </div>
        <h3>Pending notes</h3>
        <div class="chips">
          <button type="button" class="chip" :aria-pressed="String(reg.showGhosts.value)" @click="reg.showGhosts.value = !reg.showGhosts.value">
            <span class="dot ghost"></span>{{ reg.showGhosts.value ? 'Shown' : 'Hidden' }}<span class="n">{{ reg.ghosts.value.length }}</span>
          </button>
        </div>
        <p class="def dim">Dashed grey: other notes link to them, but they do not exist in the registry yet.</p>
        <div class="chips">
          <button type="button" class="btn" :aria-pressed="String(reg.tableOn.value)" @click="reg.tableOn.value = !reg.tableOn.value">Table view</button>
          <button type="button" class="btn" @click="reg.reset()">Reset filters</button>
        </div>
      </template>
    </div>

    <footer v-if="reg.meta.value" class="foot">
      registry
      <a :href="`https://github.com/zzigo/som-registry/commit/${reg.meta.value.commit}`" target="_blank" rel="noopener">{{ reg.meta.value.commit ? reg.meta.value.commit.slice(0, 7) : 'unknown' }}</a>
      · built {{ reg.meta.value.built_at }}
      <span v-if="reg.meta.value.skipped && reg.meta.value.skipped.length" :title="reg.meta.value.skipped.map(s => s.file).join('\n')"> · {{ reg.meta.value.skipped.length }} skipped</span>
    </footer>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { marked } from 'marked'
import SomMoaieWheel from '~/components/SomMoaieWheel.vue'
import {
  FACETS, FKEY, MOAIE, MOAIE_COLUMN, ORIGINS, ELEC_STEPS, INTERFACE_AXES,
  colorOf, linkColor, isInstrument, originOf, resolveWikilink, yamlLines
} from '~/composables/useSomRegistry'

const props = defineProps({ reg: { type: Object, required: true } })
const reg = props.reg

const TABS = [
  { key: 'node', label: 'Node' },
  { key: 'details', label: 'Details' },
  { key: 'filters', label: 'Filters' }
]

const sel = computed(() => reg.selected.value)
const hubs = computed(() => [...reg.real.value].sort((a, b) => b.deg - a.deg).slice(0, 8))
const topMissing = computed(() => [...reg.ghosts.value].sort((a, b) => b.indeg - a.indeg).slice(0, 8))
const edgeTotal = computed(() => reg.edges.value.reduce((s, g) => s + g.items.length, 0))

const facetChips = computed(() => FACETS
  .map(f => ({ ...f, count: reg.real.value.filter(n => n.facet === f.key).length }))
  .filter(f => f.count > 0 || Object.values(MOAIE_COLUMN).includes(f.key)))
const originChips = computed(() => [...ORIGINS, 'unset'].map(key => ({
  key,
  label: key === 'unset' ? 'origin unset' : key,
  count: reg.real.value.filter(n => isInstrument(n) && originOf(n) === key).length
})))

function facetLabel(n) {
  const f = FKEY.get(n.facet)
  return n.category ? `${f.short} ${f.label} (${n.category})` : 'uncategorised'
}
function formatCrit(value) {
  if (value === null || value === undefined) return '—'
  return typeof value === 'object' ? JSON.stringify(value) : String(value)
}
// O.elec may arrive as one level, a list of four values or a map keyed by level.
function elecSteps(value) {
  if (Array.isArray(value)) return ELEC_STEPS.map((label, i) => ({ label, value: formatCrit(value[i]), on: false }))
  if (value && typeof value === 'object') return ELEC_STEPS.map(label => ({ label, value: formatCrit(value[label]), on: false }))
  return ELEC_STEPS.map(label => ({ label, value: label === String(value) ? '●' : '', on: label === String(value) }))
}

// --- wheels -----------------------------------------------------------
const letterColor = l => FKEY.get(MOAIE_COLUMN[l]).color
const reachMax = computed(() => Math.max(1, ...Object.values(reg.reach.value || {})))
const reachAxes = computed(() => MOAIE.map(l => ({ label: l, value: reg.reach.value?.[l] || 0, color: letterColor(l) })))
const reachCaption = computed(() => `linked notes per facet · max ${reachMax.value}`)
const vectorAxes = computed(() => {
  const v = reg.note.value?.frontmatter?.moaie?.vector
  if (!Array.isArray(v) || v.length !== 5 || !v.some(x => Number(x) > 0)) return null
  return MOAIE.map((l, i) => ({ label: l, value: Number(v[i]) || 0, color: letterColor(l) }))
})
const profileAxes = computed(() => {
  const p = reg.note.value?.frontmatter?.interface_profile
  if (!p || typeof p !== 'object') return null
  const keys = INTERFACE_AXES.filter(k => k in p)
  if (!keys.some(k => Number(p[k]) > 0)) return null
  return keys.map(k => ({ label: k.replace('sensorimotor_scheme', 'sensorimotor'), value: Number(p[k]) || 0, color: '#d8d8d8' }))
})

// --- YAML and body --------------------------------------------------------
const yaml = computed(() => (reg.note.value ? yamlLines(reg.note.value.frontmatter_raw || '') : []))
const linkId = target => resolveWikilink(reg.lookup.value, target)
const goTo = id => { if (id) reg.select(id) }
function onYamlClick(e) {
  const el = e.target.closest('[data-id]')
  if (el && el.dataset.id) reg.select(el.dataset.id)
}

const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
const bodyHtml = ref('')
let purify = null
async function renderBody() {
  const body = reg.note.value?.body?.trim()
  if (!body) { bodyHtml.value = ''; return }
  if (!purify) {
    const mod = await import('dompurify')
    purify = mod.default
  }
  const withLinks = body.replace(/\[\[([^\]]+)\]\]/g, (_, inner) => {
    const [target, label] = inner.split('|')
    const id = resolveWikilink(reg.lookup.value, target)
    return `<a href="#" class="wl${id ? '' : ' miss'}" data-id="${id ? esc(id) : ''}">${esc((label || target).trim())}</a>`
  })
  bodyHtml.value = purify.sanitize(marked.parse(withLinks, { async: false, gfm: true }), {
    ADD_ATTR: ['data-id'],
    FORBID_TAGS: ['style', 'form', 'input', 'button', 'iframe']
  })
}
watch(() => reg.note.value, renderBody)
function onBodyClick(e) {
  const a = e.target.closest('a')
  if (!a) return
  if (a.classList.contains('wl')) {
    e.preventDefault()
    if (a.dataset.id) reg.select(a.dataset.id)
  } else if (a.href) {
    a.target = '_blank'
    a.rel = 'noopener noreferrer'
  }
}

function focusFirstMatch() {
  const hit = reg.real.value.find(reg.matches)
  if (hit) reg.select(hit.id)
}
</script>

<style scoped>
.panel { display: flex; flex-direction: column; height: 100%; min-height: 0; color: rgba(255, 255, 255, 0.88); font-size: 12px; line-height: 1.5; }
.tabs { display: flex; border-bottom: 1px solid rgba(255, 255, 255, 0.16); flex: none; }
.tabs button { flex: 1; background: none; border: 0; border-right: 1px solid rgba(255, 255, 255, 0.16); color: rgba(255, 255, 255, 0.55); padding: 9px 4px; font: inherit; text-transform: uppercase; letter-spacing: 0.12em; font-size: 11px; cursor: pointer; }
.tabs button:last-child { border-right: 0; }
.tabs button.on { color: #fff; box-shadow: inset 0 -1px 0 #fff; }
.tabs button:hover { color: #fff; }
.body { flex: 1; min-height: 0; overflow-y: auto; padding: 12px 14px 18px; display: flex; flex-direction: column; gap: 8px; scrollbar-width: thin; scrollbar-color: rgba(255, 255, 255, 0.25) #000; }
.body > * { flex-shrink: 0; }
.head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
h2 { margin: 0; font-size: 14px; font-weight: 500; letter-spacing: 0.04em; overflow-wrap: anywhere; }
h3 { margin: 10px 0 0; font-size: 10.5px; font-weight: 400; text-transform: uppercase; letter-spacing: 0.14em; color: rgba(255, 255, 255, 0.5); border-top: 1px solid rgba(255, 255, 255, 0.12); padding-top: 8px; display: flex; gap: 8px; align-items: baseline; }
h4 { margin: 8px 0 2px; font-size: 10.5px; font-weight: 400; letter-spacing: 0.08em; color: rgba(255, 255, 255, 0.72); display: flex; align-items: center; gap: 7px; }
h4 .bar { width: 14px; height: 1px; flex: none; }
.n { margin-left: auto; color: rgba(255, 255, 255, 0.45); font-variant-numeric: tabular-nums; font-size: 10.5px; letter-spacing: 0; }
.x { background: none; border: 1px solid rgba(255, 255, 255, 0.2); color: #fff; width: 22px; height: 22px; line-height: 1; cursor: pointer; flex: none; }
.kv { margin: 0; display: flex; flex-wrap: wrap; gap: 2px 12px; color: rgba(255, 255, 255, 0.55); font-size: 11px; align-items: center; overflow-wrap: anywhere; }
.sw { width: 9px; height: 9px; border: 1.5px solid; border-radius: 50%; display: inline-block; }
.def { margin: 0; color: rgba(255, 255, 255, 0.78); overflow-wrap: anywhere; }
.dim { color: rgba(255, 255, 255, 0.45); }
.err { color: #ff7a7a; }
.btn, .chip { font: inherit; font-size: 11px; color: rgba(255, 255, 255, 0.85); background: #000; border: 1px solid rgba(255, 255, 255, 0.22); padding: 3px 9px; cursor: pointer; letter-spacing: 0.04em; }
.btn:hover, .chip:hover { border-color: #fff; color: #fff; }
.btn[aria-pressed='true'] { border-color: #fff; }
.chips { display: flex; flex-wrap: wrap; gap: 5px; }
.chip { display: inline-flex; align-items: center; gap: 6px; }
.chip[aria-pressed='false'] { opacity: 0.4; text-decoration: line-through; }
.chip .n { margin-left: 2px; }
.dot { width: 9px; height: 9px; border: 1.5px solid #fff; border-radius: 50%; display: inline-block; flex: none; background: #000; }
.dot.ghost { border-style: dashed; }
.dot.ring { box-shadow: 0 0 0 1.5px #000, 0 0 0 2.5px rgba(255, 255, 255, 0.7); }
.search { background: #000; color: #fff; border: 1px solid rgba(255, 255, 255, 0.22); padding: 6px 9px; font: inherit; font-size: 12px; width: 100%; box-sizing: border-box; }
.search:focus { outline: 0; border-color: #fff; }
button:focus-visible, a:focus-visible, input:focus-visible, [role='button']:focus-visible { outline: 1px solid #fff; outline-offset: 2px; }
.crit { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px; margin: 0; }
.crit dt { color: rgba(255, 255, 255, 0.5); }
.crit dd { margin: 0; overflow-wrap: anywhere; }
.elec { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 3px; }
.elec li { border: 1px solid rgba(255, 255, 255, 0.18); padding: 1px 4px; display: flex; flex-direction: column; font-size: 10.5px; }
.elec li.on { border-color: #fff; }
.elec .lbl { color: rgba(255, 255, 255, 0.5); }
.edge-group ul, .plain { list-style: none; margin: 0; padding: 0; }
.nb { display: flex; align-items: baseline; gap: 7px; width: 100%; text-align: left; padding: 2px 3px; border: 0; background: none; color: inherit; font: inherit; cursor: pointer; }
.nb:hover { background: rgba(255, 255, 255, 0.08); }
.nb .nm { overflow-wrap: anywhere; }
.nb .arrow { color: rgba(255, 255, 255, 0.55); width: 12px; flex: none; }
.nb .dot { position: relative; top: 1px; }
.wheels { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; margin-top: 4px; }
.wheels.one { grid-template-columns: minmax(0, 1fr); justify-items: center; }
.yaml { margin: 0; padding: 8px 9px; border: 1px solid rgba(255, 255, 255, 0.16); background: #000; font: inherit; font-size: 11px; line-height: 1.55; overflow-x: auto; white-space: pre; tab-size: 2; }
.yaml .ln { display: block; }
.yaml .k { color: rgba(255, 255, 255, 0.42); }
.yaml .pun { color: rgba(255, 255, 255, 0.3); }
.yaml .ln.empty .k { color: rgba(255, 255, 255, 0.24); }
.yaml .v { color: #fff; }
.yaml .v.str { background: rgba(255, 255, 255, 0.09); padding: 0 3px; margin-left: 5px; }
.yaml .v.num { color: #000; background: #f5b82e; padding: 0 3px; margin-left: 5px; font-weight: 500; }
.yaml .v.bool { color: #000; background: #2fd0a0; padding: 0 3px; margin-left: 5px; }
.yaml .v.link { color: #7cc7f2; border-bottom: 1px solid #7cc7f2; cursor: pointer; margin-left: 5px; }
.yaml .v.link.miss { color: rgba(255, 255, 255, 0.5); border-bottom-style: dashed; border-color: rgba(255, 255, 255, 0.4); cursor: default; }
.yaml .v + .v { margin-left: 0; }
.yaml .ln.plain .v { margin-left: 0; background: none; color: rgba(255, 255, 255, 0.6); }
.md { font-size: 12px; overflow-wrap: anywhere; color: rgba(255, 255, 255, 0.85); }
.md :deep(h1), .md :deep(h2), .md :deep(h3), .md :deep(h4) { font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: #fff; margin: 14px 0 4px; border: 0; padding: 0; display: block; }
.md :deep(p) { margin: 0 0 8px; }
.md :deep(a) { color: #7cc7f2; }
.md :deep(a.wl) { text-decoration: none; border-bottom: 1px solid #7cc7f2; }
.md :deep(a.wl.miss) { color: rgba(255, 255, 255, 0.5); border-bottom: 1px dashed rgba(255, 255, 255, 0.4); }
.md :deep(code) { background: rgba(255, 255, 255, 0.1); padding: 0 3px; }
.md :deep(pre) { border: 1px solid rgba(255, 255, 255, 0.16); padding: 6px 8px; overflow-x: auto; }
.md :deep(pre code) { background: none; padding: 0; }
.md :deep(blockquote) { margin: 0 0 8px; padding-left: 10px; border-left: 1px solid rgba(255, 255, 255, 0.35); color: rgba(255, 255, 255, 0.7); }
.md :deep(table) { border-collapse: collapse; width: 100%; font-size: 11px; margin-bottom: 8px; }
.md :deep(th), .md :deep(td) { border: 1px solid rgba(255, 255, 255, 0.18); padding: 2px 5px; text-align: left; vertical-align: top; }
.md :deep(img) { max-width: 100%; height: auto; filter: grayscale(1) contrast(1.05); opacity: 0.85; border: 1px solid rgba(255, 255, 255, 0.16); }
.md :deep(hr) { border: 0; border-top: 1px solid rgba(255, 255, 255, 0.16); }
.md :deep(ul), .md :deep(ol) { padding-left: 18px; margin: 0 0 8px; }
.foot { flex: none; border-top: 1px solid rgba(255, 255, 255, 0.16); padding: 6px 14px; font-size: 10.5px; color: rgba(255, 255, 255, 0.45); }
.foot a { color: rgba(255, 255, 255, 0.8); }
</style>
