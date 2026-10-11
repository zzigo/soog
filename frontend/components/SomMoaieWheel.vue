<template>
  <figure class="wheel">
    <svg :viewBox="`0 0 ${S} ${S}`" role="img" :aria-label="`${title}: ${summary}`">
      <g class="grid">
        <polygon v-for="r in rings" :key="r" :points="ring(r)" />
        <line v-for="(a, i) in axes" :key="'s' + i" :x1="C" :y1="C" :x2="px(i, 1)" :y2="py(i, 1)" />
      </g>
      <polygon class="area" :points="area" />
      <g v-for="(a, i) in axes" :key="'p' + i">
        <circle class="pt" :cx="px(i, frac(a))" :cy="py(i, frac(a))" r="3.2" :style="{ stroke: a.color || '#fff' }" />
        <text class="lbl" :x="px(i, 1.2)" :y="py(i, 1.2)" :text-anchor="anchor(i)" dominant-baseline="middle"
          :style="{ fill: a.color || '#fff' }">{{ a.label }}</text>
        <title>{{ a.label }}: {{ a.display ?? a.value }}</title>
      </g>
    </svg>
    <figcaption>
      <span class="t">{{ title }}</span>
      <span class="s">{{ caption }}</span>
    </figcaption>
  </figure>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: { type: String, required: true },
  caption: { type: String, default: '' },
  axes: { type: Array, required: true }, // [{ label, value, color?, display? }]
  max: { type: Number, default: 1 },
  size: { type: Number, default: 200 }
})

const S = computed(() => props.size)
const C = computed(() => props.size / 2)
const R = computed(() => props.size / 2 - 34)
const n = computed(() => props.axes.length)
const rings = [0.25, 0.5, 0.75, 1]
const angle = i => -Math.PI / 2 + (i * 2 * Math.PI) / n.value
const px = (i, f) => C.value + Math.cos(angle(i)) * R.value * f
const py = (i, f) => C.value + Math.sin(angle(i)) * R.value * f
const frac = a => Math.max(0, Math.min(1, (Number(a.value) || 0) / (props.max || 1)))
const ring = r => props.axes.map((_, i) => `${px(i, r).toFixed(1)},${py(i, r).toFixed(1)}`).join(' ')
const area = computed(() => props.axes.map((a, i) => `${px(i, frac(a)).toFixed(1)},${py(i, frac(a)).toFixed(1)}`).join(' '))
const anchor = i => {
  const c = Math.cos(angle(i))
  return c > 0.3 ? 'start' : c < -0.3 ? 'end' : 'middle'
}
const summary = computed(() => props.axes.map(a => `${a.label} ${a.display ?? a.value}`).join(', '))
</script>

<style scoped>
.wheel { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 2px; min-width: 0; }
svg { width: 100%; max-width: 220px; height: auto; overflow: visible; }
.grid polygon, .grid line { fill: none; stroke: rgba(255, 255, 255, 0.16); stroke-width: 0.7; }
.grid polygon:not(:last-of-type) { stroke-dasharray: 2 3; }
.area { fill: rgba(255, 255, 255, 0.07); stroke: #fff; stroke-width: 1.1; stroke-linejoin: round; }
.pt { fill: #000; stroke-width: 1.4; }
.lbl { font-size: 9.5px; letter-spacing: 0.04em; font-family: inherit; }
figcaption { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 1px; }
.t { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255, 255, 255, 0.78); }
.s { font-size: 10px; color: rgba(255, 255, 255, 0.45); }
</style>
