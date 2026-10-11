import { computed, ref, shallowRef, watch } from 'vue'

// Hues are the Okabe-Ito set already used in the registry view, lifted for a black background.
export const FACETS = [
  { key: '1mat', label: 'Materials', short: 'M', color: '#3b9fe0' },
  { key: '2obj', label: 'Objects', short: 'O', color: '#ff7a1a' },
  { key: '3agn', label: 'Agents', short: 'A', color: '#2fd0a0' },
  { key: '4int', label: 'Interfaces', short: 'I', color: '#f5b82e' },
  { key: '5env', label: 'Environments', short: 'E', color: '#e58bbd' },
  { key: '0pri', label: 'Principles', short: 'P', color: '#7cc7f2' },
  { key: 'other', label: 'Dictionary, other', short: '·', color: '#9a9a9a' }
]
export const FKEY = new Map(FACETS.map(f => [f.key, f]))
export const GHOST_COLOR = '#6a6a6a'
export const MOAIE = ['M', 'O', 'A', 'I', 'E']
export const MOAIE_COLUMN = { M: '1mat', O: '2obj', A: '3agn', I: '4int', E: '5env' }
export const COLUMN_LETTER = Object.fromEntries(Object.entries(MOAIE_COLUMN).map(([l, k]) => [k, l]))
export const MAIN_COLUMNS = Object.values(MOAIE_COLUMN)
export const ORIGINS = ['traditional', 'contemporary', 'fictional', 'generated']
export const ELEC_STEPS = ['none', 'material', 'component', 'circuit']
export const INTERFACE_AXES = ['affordance', 'liveness', 'playability', 'learnability', 'situatedness',
  'mediality', 'mapping', 'sensorimotor_scheme', 'ergonomics', 'expressivity']

const LINK_ORDER = ['moaie:M', 'moaie:O', 'moaie:A', 'moaie:I', 'moaie:E', 'hyper', 'hypo', 'connect', 'derives_from', 'principle']
export const LINK_LABEL = {
  'moaie:M': 'MOAIE · materials', 'moaie:O': 'MOAIE · objects', 'moaie:A': 'MOAIE · agents',
  'moaie:I': 'MOAIE · interfaces', 'moaie:E': 'MOAIE · environments',
  hyper: 'hyper (broader)', hypo: 'hypo (narrower)', connect: 'connect',
  derives_from: 'derives from', principle: 'principle'
}

export const isInstrument = n => !n.ghost && (n.type === 'instrument' || n.kind === 'individual')
export const originOf = n => (ORIGINS.includes(n.origin) ? n.origin : 'unset')
export const facetOf = n => (n.ghost ? 'ghost' : FKEY.has(n.category) ? n.category : 'other')
export const colorOf = n => (n.ghost ? GHOST_COLOR : (FKEY.get(n.facet) || FKEY.get('other')).color)
export const radiusOf = n => (n.ghost ? 2.6 : 4.5 + Math.sqrt(n.deg) * 1.35)
export const linkColor = type => {
  const letter = type.startsWith('moaie:') ? type.slice(6) : null
  return letter ? FKEY.get(MOAIE_COLUMN[letter]).color : '#ffffff'
}

const stemOf = file => (file || '').split('/').pop().replace(/\.md$/i, '')
const norm = s => String(s).normalize('NFC').trim().toLowerCase()

export function prepareGraph(data) {
  const nodes = data.nodes.map(n => ({ ...n, facet: facetOf(n), deg: 0, indeg: 0, xKeys: [] }))
  const byId = new Map(nodes.map(n => [n.id, n]))
  const links = data.links.filter(l => byId.has(l.source) && byId.has(l.target)).map(l => ({ ...l }))
  links.forEach(l => {
    const s = byId.get(l.source)
    const t = byId.get(l.target)
    s.deg++
    t.deg++
    t.indeg++
    // Instruments sit between the facet columns their resolved links reach.
    if (isInstrument(s) && !t.ghost && COLUMN_LETTER[t.facet]) s.xKeys.push(t.facet)
    // A pending note reached through moaie:X belongs to column X.
    if (t.ghost && l.type.startsWith('moaie:')) t.xKeys.push(MOAIE_COLUMN[l.type.slice(6)])
  })
  nodes.forEach(n => {
    if (!n.ghost && !isInstrument(n)) n.xKeys = [n.facet]
  })
  const present = new Set(nodes.filter(n => !n.ghost).map(n => n.facet))
  const columns = FACETS.map(f => f.key).filter(k => MAIN_COLUMNS.includes(k) || present.has(k))
  // Wikilink targets resolve by file stem, title or id, like the backend does.
  const lookup = new Map()
  for (const pick of [n => stemOf(n.file), n => n.title, n => n.id]) {
    nodes.forEach(n => {
      if (n.ghost) return
      const k = norm(pick(n))
      if (k && !lookup.has(k)) lookup.set(k, n.id)
    })
  }
  nodes.forEach(n => {
    if (n.ghost) lookup.set(norm(n.title), n.id)
  })
  return { nodes, links, columns, lookup }
}

export function resolveWikilink(lookup, target) {
  const t = String(target).split('|')[0].split('#')[0].replace(/\.md$/i, '').trim()
  return lookup.get(norm(t)) || lookup.get(norm(t.split('/').pop())) || null
}

/** Edges of one node grouped by relation type, with direction. */
export function edgesOf(id, links, byId) {
  const groups = new Map()
  links.forEach(l => {
    const s = l.source.id ?? l.source
    const t = l.target.id ?? l.target
    if (s !== id && t !== id) return
    const other = byId.get(s === id ? t : s)
    if (!other) return
    if (!groups.has(l.type)) groups.set(l.type, [])
    groups.get(l.type).push({ node: other, dir: s === id ? 'out' : 'in' })
  })
  const rank = type => (LINK_ORDER.indexOf(type) < 0 ? 99 : LINK_ORDER.indexOf(type))
  return [...groups.entries()]
    .sort((a, b) => rank(a[0]) - rank(b[0]))
    .map(([type, items]) => ({ type, label: LINK_LABEL[type] || type, items: items.sort((a, b) => b.node.deg - a.node.deg) }))
}

/** How many neighbours of a node sit in each MOAIE facet (a graph measure, not a declared value). */
export function reachOf(id, links, byId) {
  const counts = { M: 0, O: 0, A: 0, I: 0, E: 0 }
  links.forEach(l => {
    const s = l.source.id ?? l.source
    const t = l.target.id ?? l.target
    if (s !== id && t !== id) return
    const other = byId.get(s === id ? t : s)
    if (!other) return
    const letter = other.ghost
      ? (l.type.startsWith('moaie:') ? l.type.slice(6) : null)
      : COLUMN_LETTER[other.facet]
    if (letter) counts[letter]++
  })
  return counts
}

const NUM = /^\s*-?\d+(\.\d+)?\s*$/
const WIKI = /\[\[([^\]]+)\]\]/g

function valueSegments(rest) {
  const out = []
  let last = 0
  for (const m of rest.matchAll(WIKI)) {
    if (m.index > last) out.push({ k: 'str', t: rest.slice(last, m.index) })
    out.push({ k: 'link', t: m[0], target: m[1] })
    last = m.index + m[0].length
  }
  if (last < rest.length) out.push({ k: 'str', t: rest.slice(last) })
  return out.flatMap(seg => {
    if (seg.k !== 'str') return [seg]
    if (NUM.test(seg.t)) return [{ k: 'num', t: seg.t }]
    if (/^\s*\[[\d\s.,-]*\]\s*$/.test(seg.t)) {
      return seg.t.split(/(-?\d+(?:\.\d+)?)/).filter(Boolean).map(p => ({ k: NUM.test(p) ? 'num' : 'str', t: p }))
    }
    if (/^\s*(true|false)\s*$/i.test(seg.t)) return [{ k: 'bool', t: seg.t }]
    return [seg]
  })
}

/** Frontmatter text as lines of typed tokens, so values can be styled apart from keys. */
export function yamlLines(raw) {
  return raw.split('\n').map(line => {
    const key = line.match(/^(\s*)(-\s+)?([^\s:#\-][^:]*?):(?:\s+(.*))?$/)
    if (key) {
      const rest = key[4] ?? ''
      const empty = /^\s*(""|''|\[\]|\{\}|null|~)?\s*$/.test(rest)
      return { indent: key[1], bullet: !!key[2], key: key[3], segs: empty ? [] : valueSegments(rest), empty }
    }
    const item = line.match(/^(\s*)-\s+(.*)$/)
    if (item) return { indent: item[1], bullet: true, key: null, segs: valueSegments(item[2]), empty: false }
    return { indent: '', bullet: false, key: null, segs: line.trim() ? [{ k: 'str', t: line }] : [], empty: !line.trim(), plain: true }
  })
}

const noteCache = new Map()
export async function fetchNote(apiBase, id) {
  if (noteCache.has(id)) return noteCache.get(id)
  const res = await fetch(`${apiBase}/somap/registry/note?id=${encodeURIComponent(id)}`)
  const data = await res.json()
  if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`)
  noteCache.set(id, data)
  return data
}

export function useSomRegistryState(apiBase) {
  const loading = ref(true)
  const error = ref('')
  const meta = ref(null)
  const nodes = shallowRef([])
  const links = shallowRef([])
  const columns = shallowRef([])
  const lookup = shallowRef(new Map())
  const hidden = ref(new Set())
  const showGhosts = ref(true)
  const query = ref('')
  const selId = ref(null)
  const tab = ref('node')
  const tableOn = ref(false)
  const note = ref(null)
  const noteLoading = ref(false)
  const noteError = ref('')

  const real = computed(() => nodes.value.filter(n => !n.ghost))
  const ghosts = computed(() => nodes.value.filter(n => n.ghost))
  const byId = computed(() => new Map(nodes.value.map(n => [n.id, n])))
  const instrumentCount = computed(() => real.value.filter(isInstrument).length)
  const selected = computed(() => (selId.value ? byId.value.get(selId.value) || null : null))
  const adjacency = computed(() => {
    const map = new Map(nodes.value.map(n => [n.id, new Set()]))
    links.value.forEach(l => {
      map.get(l.source)?.add(l.target)
      map.get(l.target)?.add(l.source)
    })
    return map
  })
  const edges = computed(() => (selId.value ? edgesOf(selId.value, links.value, byId.value) : []))
  const reach = computed(() => (selId.value ? reachOf(selId.value, links.value, byId.value) : null))

  const matches = n => {
    const q = query.value.trim().toLowerCase()
    return !q || n.title.toLowerCase().includes(q) || n.id.toLowerCase().includes(q)
  }
  const isVisible = n => {
    if (n.ghost) return showGhosts.value
    if (hidden.value.has(n.facet)) return false
    if (isInstrument(n) && (hidden.value.has('indiv') || hidden.value.has('origin:' + originOf(n)))) return false
    return true
  }
  const toggle = key => {
    const next = new Set(hidden.value)
    next.has(key) ? next.delete(key) : next.add(key)
    hidden.value = next
  }
  const select = id => {
    selId.value = id && byId.value.has(id) ? id : null
    if (selId.value && tab.value === 'filters') tab.value = 'node'
  }
  const reset = () => {
    selId.value = null
    query.value = ''
    hidden.value = new Set()
    showGhosts.value = true
  }

  async function load() {
    loading.value = true
    error.value = ''
    try {
      const res = await fetch(`${apiBase.value}/somap/registry`)
      const data = await res.json()
      if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`)
      meta.value = data.meta
      const g = prepareGraph(data)
      nodes.value = g.nodes
      links.value = g.links
      columns.value = g.columns
      lookup.value = g.lookup
    } catch (e) {
      error.value = `Could not load the registry graph: ${e.message}`
    } finally {
      loading.value = false
    }
  }

  // The note is only fetched while the Details tab is open on a real node.
  watch([selId, tab], async () => {
    note.value = null
    noteError.value = ''
    const n = selected.value
    if (tab.value !== 'details' || !n || n.ghost) return
    noteLoading.value = true
    const wanted = n.id
    try {
      const data = await fetchNote(apiBase.value, wanted)
      if (selId.value === wanted) note.value = data
    } catch (e) {
      if (selId.value === wanted) noteError.value = e.message
    } finally {
      if (selId.value === wanted) noteLoading.value = false
    }
  })

  return {
    loading, error, meta, nodes, links, columns, lookup, hidden, showGhosts, query, selId, tab, tableOn,
    note, noteLoading, noteError, real, ghosts, byId, instrumentCount, selected, adjacency, edges, reach,
    matches, isVisible, toggle, select, reset, load
  }
}
