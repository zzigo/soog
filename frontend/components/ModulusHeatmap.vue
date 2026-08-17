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
  }
});

const canvasRef = ref(null);

function projectCoord(value, span) {
  const numeric = Number(value);
  const clamped = Number.isFinite(numeric) ? Math.max(-1, Math.min(1, numeric)) : 0;
  return ((clamped + 1) / 2) * span;
}

function strokeMarker(ctx, marker, x, y) {
  const palette = {
    source: '#ffb300',
    probe: '#00e5ff',
    obstacle: '#ffffff'
  };
  const color = palette[marker?.type] || '#9e9e9e';
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.5;

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
  const rows = props.data.length;
  const cols = props.data[0].length;
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;

  // Find max absolute value for normalization
  let maxVal = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      maxVal = Math.max(maxVal, Math.abs(props.data[r][c]));
    }
  }
  if (maxVal === 0) maxVal = 1;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = props.data[r][c];
      const norm = val / maxVal; // -1 to 1

      // Blue-White-Red Diverging Colormap
      let r8, g8, b8;
      if (norm > 0) {
        // Red intensity
        r8 = 255;
        g8 = b8 = Math.floor(255 * (1 - norm));
      } else {
        // Blue intensity
        b8 = 255;
        r8 = g8 = Math.floor(255 * (1 + norm));
      }

      ctx.fillStyle = `rgb(${r8}, ${g8}, ${b8})`;
      ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  for (const marker of props.markers || []) {
    const x = projectCoord(marker?.x, canvas.width);
    const y = canvas.height - projectCoord(marker?.y, canvas.height);
    strokeMarker(ctx, marker, x, y);
  }
};

onMounted(drawHeatmap);
watch(() => props.data, drawHeatmap, { deep: true });
watch(() => props.markers, drawHeatmap, { deep: true });
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
  background: #000;
  border: 1px solid rgba(255, 255, 255, 0.14);
  image-rendering: pixelated;
  max-width: 100%;
  height: auto;
}

.heatmap-legend {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 10px;
  color: rgba(255, 255, 255, 0.6);
  width: 200px;
}

.gradient-bar {
  flex: 1;
  height: 8px;
  background: linear-gradient(to right, #00f, #fff, #f00);
  border-radius: 4px;
}
</style>
