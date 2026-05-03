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
  }
});

const canvasRef = ref(null);

const drawHeatmap = () => {
  const canvas = canvasRef.value;
  if (!canvas || !props.data || props.data.length === 0) return;

  const ctx = canvas.getContext('2d');
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

  // Draw Source and Obstacle indicators if possible
  // (Heuristic: look for extreme values or just keep it clean)
};

onMounted(drawHeatmap);
watch(() => props.data, drawHeatmap, { deep: true });
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
  border: 1px solid rgba(255, 255, 255, 0.2);
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
