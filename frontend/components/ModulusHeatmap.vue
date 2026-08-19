<template>
  <div class="heatmap-container">
    <canvas ref="canvasRef" :width="size" :height="size" class="heatmap-canvas"></canvas>
    <div class="heatmap-legend">
      <span>-Max</span>
      <div class="gradient-bar"></div>
      <span>+Max</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import { acousticLegendGradient, sampleAcousticPalette } from '~/utils/acousticPalette';
import {
  normalizePlaneKey,
  projectMarkerToPlane,
  phaseDistance2d,
  temporalFieldValue,
} from '~/utils/acousticTemporal';

const props = defineProps({
  data: {
    type: Array,
    required: true
  },
  size: {
    type: Number,
    default: 400
  },
  markers: {
    type: Array,
    default: () => []
  },
  plane: {
    type: String,
    default: 'xy'
  },
  phase: {
    type: Number,
    default: 0
  }
});

const canvasRef = ref(null);
const legendGradient = acousticLegendGradient(0.95);

function projectCoord(value, span) {
  const numeric = Number(value);
  const clamped = Number.isFinite(numeric) ? Math.max(-1, Math.min(1, numeric)) : 0;
  return ((clamped + 1) / 2) * span;
}

function strokeMarker(ctx, marker, x, y) {
  const palette = {
    source: '#ff9a1f',
    probe: '#00f6ff',
    obstacle: '#ffffff'
  };
  const color = palette[marker?.type] || '#9e9e9e';
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.5;
  ctx.shadowBlur = 16;
  ctx.shadowColor = color;

  if (marker?.type === 'source') {
    ctx.beginPath();
    ctx.moveTo(x - 8, y);
    ctx.lineTo(x + 8, y);
    ctx.moveTo(x, y - 8);
    ctx.lineTo(x, y + 8);
    ctx.stroke();
  } else if (marker?.type === 'probe') {
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y, 1.8, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.strokeRect(x - 6, y - 6, 12, 12);
  }

  if (marker?.label) {
    ctx.font = '10px IBM Plex Mono, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(String(marker.label), x + 10, y - 8);
  }
  ctx.restore();
}

const drawHeatmap = () => {
  const canvas = canvasRef.value;
  if (!canvas || !props.data || props.data.length === 0) return;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  const rows = props.data.length;
  const cols = props.data[0].length;
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;
  const planeKey = normalizePlaneKey(props.plane);

  const backgroundGradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  backgroundGradient.addColorStop(0, 'rgba(4, 10, 24, 0.72)');
  backgroundGradient.addColorStop(0.5, 'rgba(13, 16, 38, 0.38)');
  backgroundGradient.addColorStop(1, 'rgba(3, 5, 18, 0.72)');
  ctx.fillStyle = backgroundGradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const transformed = Array.from({ length: rows }, () => new Array(cols).fill(0));
  let maxVal = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const distance = phaseDistance2d(r, c, rows, cols, props.markers, planeKey);
      const value = temporalFieldValue(props.data[r][c], props.phase, distance);
      transformed[r][c] = value;
      maxVal = Math.max(maxVal, Math.abs(value));
    }
  }
  if (maxVal === 0) maxVal = 1;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = transformed[r][c];
      const norm = val / maxVal; // -1 to 1
      const color = sampleAcousticPalette(norm, 0.92);
      ctx.fillStyle = color.css;
      ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  ctx.strokeStyle = 'rgba(60, 196, 255, 0.09)';
  ctx.lineWidth = 1;
  for (let col = 0; col <= cols; col += Math.max(1, Math.round(cols / 14))) {
    ctx.beginPath();
    ctx.moveTo(col * cellW, 0);
    ctx.lineTo(col * cellW, canvas.height);
    ctx.stroke();
  }
  for (let row = 0; row <= rows; row += Math.max(1, Math.round(rows / 14))) {
    ctx.beginPath();
    ctx.moveTo(0, row * cellH);
    ctx.lineTo(canvas.width, row * cellH);
    ctx.stroke();
  }

  for (const marker of props.markers || []) {
    const projected = projectMarkerToPlane(marker, planeKey);
    const x = projectCoord(projected.u, canvas.width);
    const y = canvas.height - projectCoord(projected.v, canvas.height);
    strokeMarker(ctx, marker, x, y);
  }

  ctx.strokeStyle = 'rgba(0, 246, 255, 0.34)';
  ctx.lineWidth = 1.25;
  ctx.strokeRect(0.625, 0.625, canvas.width - 1.25, canvas.height - 1.25);
  ctx.restore();
};

onMounted(drawHeatmap);
watch(() => props.data, drawHeatmap, { deep: true });
watch(() => props.markers, drawHeatmap, { deep: true });
watch(() => props.plane, drawHeatmap);
watch(() => props.phase, drawHeatmap);
</script>

<style scoped>
.heatmap-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  width: 100%;
}

.heatmap-canvas {
  background: transparent;
  border: 1px solid rgba(0, 246, 255, 0.22);
  box-shadow:
    inset 0 0 0 1px rgba(255, 65, 243, 0.06),
    0 0 28px rgba(0, 246, 255, 0.08);
  image-rendering: pixelated;
  max-width: 100%;
  height: auto;
}

.heatmap-legend {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 10px;
  color: rgba(202, 232, 255, 0.64);
  width: 200px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.gradient-bar {
  flex: 1;
  height: 8px;
  background: linear-gradient(to right, v-bind(legendGradient));
  border-radius: 999px;
  box-shadow: 0 0 16px rgba(0, 246, 255, 0.18);
}
</style>
