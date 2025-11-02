<template>
  <div class="knowledge-graph-3d">
    <button @click="initGraph">Init 3D Graph</button>
    <div ref="threeContainer" class="threejs-canvas"></div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
let threeInstance = null
const threeContainer = ref(null)

function initGraph() {
  if (threeInstance) return
  // Placeholder: Aquí irá la inicialización de three.js
  // Puedes importar three.js y renderizar un grafo básico
}

onMounted(() => {
  // Placeholder para inicializar si es necesario
})
</script>

<style scoped>
.knowledge-graph-3d {
  background: #111;
  color: #fff;
  min-height: 400px;
  position: relative;
  border-radius: 8px;
  margin-bottom: 2rem;
}

.threejs-canvas {
  width: 100%;
  height: 600px;
  background: #000;
  margin-top: 1rem;
  border-radius: 8px;
}
</style>
