<template>
  <div class="sr">
    <aside class="sr-side" aria-label="Registry panel">
      <header class="brand">
        <span class="title">SOMAP</span>
        <span class="sub">speculative organology registry</span>
      </header>
      <SomRegistryPanel :reg="reg" />
    </aside>

    <section class="sr-main">
      <header class="sr-bar">
        <div class="views" role="group" aria-label="View">
          <button type="button" :class="{ on: mode === '2d' }" @click="setView('2d')">2D</button>
          <button type="button" :class="{ on: mode === '3d' }" @click="setView('3d')">3D</button>
        </div>
        <div class="stats" aria-live="polite" v-if="!reg.loading.value && !reg.error.value">
          <span><b>{{ reg.real.value.length }}</b> notes</span>
          <span><b>{{ reg.instrumentCount.value }}</b> instruments</span>
          <span><b>{{ reg.ghosts.value.length }}</b> pending</span>
          <span><b>{{ reg.links.value.length }}</b> links</span>
        </div>
        <div class="tools">
          <button type="button" @click="fit">Fit all</button>
          <button type="button" @click="reg.select(null)" :disabled="!reg.selId.value">Deselect</button>
        </div>
      </header>

      <div class="canvas">
        <template v-if="!reg.loading.value && !reg.error.value">
          <SomRegistryGraph v-if="mode === '2d'" ref="stage" :reg="reg" @select="reg.select" />
          <SomRegistry3D v-else ref="stage" :reg="reg" @select="reg.select" />
        </template>
        <div v-if="reg.loading.value" class="status">Loading registry…</div>
        <div v-if="reg.error.value" class="status error">{{ reg.error.value }}</div>

        <div v-if="reg.tableOn.value" class="table-overlay">
          <button type="button" class="close" @click="reg.tableOn.value = false">Close table ×</button>
          <table>
            <thead>
              <tr><th>Note</th><th>Facet</th><th>Kind</th><th>Type</th><th>Origin</th><th class="num">Links</th><th class="num">Criteria</th><th>Definition</th></tr>
            </thead>
            <tbody>
              <tr v-for="n in tableRows" :key="n.id">
                <td><button type="button" class="link" @click="pick(n.id)">{{ n.title }}</button></td>
                <td>{{ n.category || '—' }}</td><td>{{ n.kind }}</td><td>{{ n.type }}</td><td>{{ n.origin }}</td>
                <td class="num">{{ n.deg }}</td><td class="num">{{ n.n_crit }}</td><td class="def">{{ n.def }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useApi } from '~/composables/useApi'
import { useSomRegistryState } from '~/composables/useSomRegistry'
import SomRegistryPanel from '~/components/SomRegistryPanel.vue'
import SomRegistryGraph from '~/components/SomRegistryGraph.vue'
import SomRegistry3D from '~/components/SomRegistry3D.vue'

const route = useRoute()
const router = useRouter()
const { apiBase } = useApi()
const reg = useSomRegistryState(apiBase)
const stage = ref(null)

const mode = computed(() => (route.query.view === '3d' ? '3d' : '2d'))
const TABS = ['node', 'details', 'filters']

function setView(view) {
  router.replace({ query: { ...route.query, view } })
}
function fit() {
  stage.value?.fit?.()
}
function pick(id) {
  reg.select(id)
  reg.tableOn.value = false
}
const tableRows = computed(() => reg.real.value.filter(reg.matches)
  .sort((a, b) => (a.category || 'zz').localeCompare(b.category || 'zz') || b.deg - a.deg))

// Selection and tab live in the URL (?node=<id>&tab=details) so a link opens on that note.
// The initial values are read once, and the URL is only rewritten after the data has loaded.
const initial = { node: typeof route.query.node === 'string' ? route.query.node : null, tab: route.query.tab }
let urlReady = false
function syncUrl() {
  if (!urlReady) return
  const q = { ...route.query }
  if (reg.selId.value) q.node = reg.selId.value
  else delete q.node
  if (reg.tab.value !== 'node') q.tab = reg.tab.value
  else delete q.tab
  if (q.node !== route.query.node || q.tab !== route.query.tab) router.replace({ query: q })
}
watch([reg.selId, reg.tab], syncUrl)

onMounted(async () => {
  if (TABS.includes(initial.tab)) reg.tab.value = initial.tab
  await reg.load()
  if (initial.node) reg.select(initial.node)
  urlReady = true
})
</script>

<style scoped>
.sr { display: grid; grid-template-columns: minmax(300px, 390px) minmax(0, 1fr); height: 100vh; background: #000; color: rgba(255, 255, 255, 0.88); font-family: 'IBM Plex Mono', 'Courier New', monospace; overflow: hidden; }
.sr-side { display: flex; flex-direction: column; min-height: 0; border-right: 1px solid rgba(255, 255, 255, 0.16); background: #000; }
.brand { display: flex; align-items: baseline; gap: 10px; padding: 11px 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.16); flex: none; }
.title { letter-spacing: 0.16em; font-size: 13px; color: #fff; }
.sub { font-size: 10.5px; color: rgba(255, 255, 255, 0.45); }
.sr-main { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.sr-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 18px; padding: 8px 14px; border-bottom: 1px solid rgba(255, 255, 255, 0.16); font-size: 11px; flex: none; }
.views { display: flex; align-items: center; gap: 0; }
.views button, .views a, .tools button { background: #000; color: rgba(255, 255, 255, 0.65); border: 1px solid rgba(255, 255, 255, 0.22); padding: 3px 10px; font: inherit; font-size: 11px; letter-spacing: 0.08em; cursor: pointer; text-decoration: none; margin-right: -1px; }
.views a { margin-left: 12px; border-color: transparent; padding-left: 0; }
.views button.on { color: #000; background: #fff; border-color: #fff; }
.views button:hover, .views a:hover, .tools button:hover:not(:disabled) { color: #fff; border-color: #fff; }
.views button.on:hover { color: #000; }
.tools { display: flex; gap: 6px; margin-left: auto; }
.tools button { margin-right: 0; }
.tools button:disabled { opacity: 0.35; cursor: default; }
.stats { display: flex; flex-wrap: wrap; gap: 4px 16px; color: rgba(255, 255, 255, 0.55); font-variant-numeric: tabular-nums; }
.stats b { color: #fff; font-weight: 500; }
.canvas { position: relative; flex: 1; min-height: 0; }
.status { position: absolute; left: 14px; bottom: 12px; font-size: 12px; color: rgba(255, 255, 255, 0.6); }
.status.error { color: #ff7a7a; }
.table-overlay { position: absolute; inset: 0; overflow: auto; background: #000; padding: 10px 14px 20px; z-index: 5; }
.close { background: #000; color: #fff; border: 1px solid rgba(255, 255, 255, 0.3); font: inherit; font-size: 11px; padding: 3px 10px; cursor: pointer; margin-bottom: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 11.5px; }
th, td { text-align: left; padding: 5px 9px; border-bottom: 1px solid rgba(255, 255, 255, 0.12); vertical-align: top; }
th { font-weight: 400; font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(255, 255, 255, 0.5); white-space: nowrap; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
td.def { color: rgba(255, 255, 255, 0.6); min-width: 240px; }
.link { background: none; border: 0; color: #fff; font: inherit; cursor: pointer; padding: 0; text-align: left; border-bottom: 1px solid rgba(255, 255, 255, 0.3); }
button:focus-visible, a:focus-visible { outline: 1px solid #fff; outline-offset: 2px; }
@media (max-width: 860px) {
  .sr { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) minmax(0, 46vh); }
  .sr-side { order: 2; border-right: 0; border-top: 1px solid rgba(255, 255, 255, 0.16); }
  .sr-main { order: 1; }
}
</style>
