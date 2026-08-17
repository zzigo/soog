<template>
  <section v-if="modelValue" class="usage-shell" aria-labelledby="usage-title">
        <header class="usage-head">
          <div>
            <div class="usage-kicker">ADMIN · RESOURCE CONTROL</div>
            <h2 id="usage-title">Render usage</h2>
          </div>
          <div class="usage-head-actions">
            <button :class="{ active: period === 'day' }" @click="setPeriod('day')">DAY</button>
            <button :class="{ active: period === 'week' }" @click="setPeriod('week')">WEEK</button>
            <button class="usage-close" @click="emit('close')">CLOSE</button>
          </div>
        </header>

        <div v-if="loading" class="usage-state">Loading usage…</div>
        <div v-else-if="error" class="usage-state error">{{ error }}</div>
        <template v-else-if="data">
          <div class="usage-cards">
            <article><span>RENDERS</span><strong>{{ data.summary.renders }}</strong></article>
            <article><span>LLM CALLS</span><strong>{{ data.summary.llm_calls }}</strong></article>
            <article><span>TOKENS</span><strong>{{ compactNumber(data.summary.total_tokens) }}</strong></article>
            <article><span>ACTIVE USERS</span><strong>{{ data.summary.active_users }}</strong></article>
            <article><span>FAILED</span><strong>{{ data.summary.failed }}</strong></article>
          </div>

          <section class="usage-section usage-chart-section">
            <div class="section-heading">
              <h3>{{ period === 'day' ? 'HOURLY REGISTER' : 'DAILY REGISTER' }}</h3>
              <span>{{ data.timezone }}</span>
            </div>
            <div class="usage-chart">
              <div v-for="item in data.series" :key="item.label" class="chart-column" :title="`${item.label}: ${item.renders} renders`">
                <div class="chart-track">
                  <div class="chart-bar" :style="{ height: `${barHeight(item.renders)}%` }"></div>
                </div>
                <span>{{ item.label }}</span>
              </div>
            </div>
          </section>

          <section class="usage-section">
            <div class="section-heading">
              <h3>USER QUOTAS</h3>
              <span>blank = default · -1 = unlimited</span>
            </div>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>USER</th>
                    <th>PERIOD</th>
                    <th>CALLS</th>
                    <th>TOKENS</th>
                    <th>DAILY</th>
                    <th>WEEKLY</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="user in data.users" :key="user.subject">
                    <td>
                      <div class="user-name">
                        {{ user.name || user.email || user.subject }}
                        <span v-if="user.is_admin">ADMIN</span>
                      </div>
                      <small>{{ user.email || user.subject }}</small>
                    </td>
                    <td>{{ user.renders }}</td>
                    <td>{{ user.llm_calls }}</td>
                    <td>{{ compactNumber(user.total_tokens) }}</td>
                    <td>
                      <input v-model="drafts[user.subject].daily" inputmode="numeric" :disabled="user.is_admin" />
                      <small>{{ quotaUsage(user.quota.daily) }}</small>
                    </td>
                    <td>
                      <input v-model="drafts[user.subject].weekly" inputmode="numeric" :disabled="user.is_admin" />
                      <small>{{ quotaUsage(user.quota.weekly) }}</small>
                    </td>
                    <td>
                      <button class="save-btn" :disabled="saving === user.subject || user.is_admin" @click="saveQuota(user.subject)">
                        {{ saving === user.subject ? '…' : 'SAVE' }}
                      </button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section class="usage-section recent-section">
            <div class="section-heading"><h3>RECENT EVENTS</h3></div>
            <div v-if="!data.recent.length" class="empty">No renders in this period.</div>
            <div v-else class="recent-list">
              <div v-for="event in data.recent" :key="event.id" class="recent-row">
                <time>{{ formatTime(event.started_at) }}</time>
                <span>{{ event.name || event.email || event.subject }}</span>
                <span>{{ event.action }}</span>
                <span>{{ event.llm_calls }} calls</span>
                <span>{{ compactNumber(event.total_tokens) }} tok</span>
                <span :class="['event-status', event.status]">{{ event.status }}</span>
              </div>
            </div>
          </section>
        </template>
  </section>
</template>

<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import { useSoogAuth } from '~/composables/useSoogAuth'

const props = defineProps<{ modelValue: boolean; apiBase: string }>()
const emit = defineEmits<{ close: [] }>()
const { authHeaders } = useSoogAuth()
const period = ref<'day' | 'week'>('week')
const loading = ref(false)
const saving = ref('')
const error = ref('')
const data = ref<Record<string, any> | null>(null)
const drafts = reactive<Record<string, { daily: string; weekly: string }>>({})

const maxRenders = computed(() => Math.max(1, ...(data.value?.series || []).map((item: any) => Number(item.renders || 0))))
const barHeight = (value: number) => Math.max(value ? 8 : 1, Math.round((Number(value || 0) / maxRenders.value) * 100))
const compactNumber = (value: number) => new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(Number(value || 0))
const quotaUsage = (item: any) => `${item.used}/${item.limit === null ? '∞' : item.limit}`
const formatTime = (value: string) => new Intl.DateTimeFormat('en-GB', { dateStyle: 'short', timeStyle: 'short' }).format(new Date(value))

const hydrateDrafts = () => {
  for (const user of data.value?.users || []) {
    drafts[user.subject] = {
      daily: user.configured_quota.daily === null ? '' : String(user.configured_quota.daily),
      weekly: user.configured_quota.weekly === null ? '' : String(user.configured_quota.weekly),
    }
  }
}

const load = async () => {
  if (!props.modelValue || loading.value) return
  loading.value = true
  error.value = ''
  try {
    const response = await fetch(`${props.apiBase}/admin/usage?period=${period.value}`, {
      headers: { Accept: 'application/json', ...(await authHeaders()) },
    })
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(payload.error || `Usage request failed (${response.status})`)
    data.value = payload
    hydrateDrafts()
  } catch (cause: any) {
    error.value = cause?.message || 'Unable to load usage.'
  } finally {
    loading.value = false
  }
}

const setPeriod = async (value: 'day' | 'week') => {
  period.value = value
  await load()
}

const saveQuota = async (subject: string) => {
  saving.value = subject
  error.value = ''
  try {
    const draft = drafts[subject]
    const response = await fetch(`${props.apiBase}/admin/quotas/${encodeURIComponent(subject)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json', ...(await authHeaders()) },
      body: JSON.stringify({
        daily: draft.daily.trim() === '' ? null : Number(draft.daily),
        weekly: draft.weekly.trim() === '' ? null : Number(draft.weekly),
      }),
    })
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(payload.error || `Quota update failed (${response.status})`)
    await load()
  } catch (cause: any) {
    error.value = cause?.message || 'Unable to update quota.'
  } finally {
    saving.value = ''
  }
}

watch(() => props.modelValue, (open) => { if (open) void load() }, { immediate: true })
</script>

<style scoped>
.usage-shell {
  width: 100%;
  height: 100dvh;
  margin: 0;
  box-sizing: border-box;
  background: #070707;
  color: rgba(255, 255, 255, 0.72);
  font-family: 'IBM Plex Mono', monospace;
  overflow: auto;
}

.usage-head,
.section-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

.usage-head {
  padding: clamp(16px, 2vw, 28px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}

.usage-kicker,
h3,
th {
  font-size: 9px;
  letter-spacing: 0.16em;
  font-weight: 500;
}

.usage-kicker { color: #4caf50; }
h2 { margin: 6px 0 0; color: white; font-size: 20px; font-weight: 400; }
h3 { margin: 0; color: rgba(255, 255, 255, 0.72); }

.usage-head-actions { display: flex; gap: 6px; }
.usage-head-actions button,
.save-btn {
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: transparent;
  color: rgba(255, 255, 255, 0.52);
  padding: 7px 10px;
  font: inherit;
  font-size: 9px;
  letter-spacing: 0.09em;
  cursor: pointer;
}
.usage-head-actions button.active { border-color: #4caf50; color: #4caf50; }
.usage-close { margin-left: 8px; }

.usage-cards {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}
.usage-cards article { padding: clamp(16px, 2vw, 28px); }
.usage-cards span { display: block; margin-bottom: 10px; color: rgba(255, 255, 255, 0.35); font-size: 9px; letter-spacing: 0.12em; }
.usage-cards strong { color: white; font-size: 22px; font-weight: 400; }

.usage-section { padding: clamp(16px, 2vw, 28px); border-bottom: 1px solid rgba(255, 255, 255, 0.12); }
.section-heading span { color: rgba(255, 255, 255, 0.3); font-size: 9px; }
.usage-chart { height: 130px; display: flex; gap: 8px; margin-top: 18px; }
.chart-column { min-width: 0; flex: 1; display: flex; flex-direction: column; gap: 7px; text-align: center; }
.chart-track { position: relative; flex: 1; border-bottom: 1px solid rgba(255, 255, 255, 0.12); }
.chart-bar { position: absolute; right: 18%; bottom: 0; left: 18%; min-height: 1px; background: #4caf50; opacity: 0.72; }
.chart-column span { color: rgba(255, 255, 255, 0.34); font-size: 8px; white-space: nowrap; }

.table-wrap { overflow-x: auto; margin-top: 16px; }
table { width: 100%; border-collapse: collapse; font-size: 10px; }
th { color: rgba(255, 255, 255, 0.32); text-align: left; }
th, td { padding: 9px 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); }
.user-name { color: rgba(255, 255, 255, 0.78); }
.user-name span { margin-left: 6px; color: #4caf50; font-size: 8px; }
td small { display: block; margin-top: 3px; max-width: 240px; overflow: hidden; color: rgba(255, 255, 255, 0.3); text-overflow: ellipsis; }
input { width: 58px; padding: 5px; border: 1px solid rgba(255, 255, 255, 0.15); background: #020202; color: white; font: inherit; }
.save-btn:disabled { opacity: 0.3; cursor: default; }

.recent-list { margin-top: 12px; }
.recent-row { display: grid; grid-template-columns: 120px 1.5fr 1fr 70px 70px 80px; gap: 12px; padding: 8px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.07); font-size: 9px; }
.recent-row time, .recent-row span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.event-status.succeeded { color: #81c784; }
.event-status.failed { color: #ef9a9a; }
.usage-state, .empty { padding: 40px 22px; color: rgba(255, 255, 255, 0.42); font-size: 11px; }
.usage-state.error { color: #ef9a9a; }

@media (max-width: 760px) {
  .usage-cards { grid-template-columns: repeat(2, 1fr); }
  .usage-cards article:last-child { grid-column: 1 / -1; }
  .usage-head { align-items: flex-start; flex-direction: column; }
  .usage-head-actions { width: 100%; flex-wrap: wrap; }
  .usage-close { margin-left: auto; }
  .section-heading { align-items: flex-start; flex-direction: column; gap: 6px; }
  .usage-chart { gap: 4px; overflow-x: auto; }
  .chart-column { min-width: 38px; }
  .recent-row { grid-template-columns: 100px 1fr 70px; }
  .recent-row span:nth-child(3), .recent-row span:nth-child(4), .recent-row span:nth-child(5) { display: none; }
}
</style>
