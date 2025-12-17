<template>
  <div class="somap-container">
    <header>
      <select v-model="activePerspective" @change="loadPerspective">
        <option v-for="p in perspectives" :key="p" :value="p">{{ p }}</option>
      </select>
      <input v-model="newColumn" placeholder="Nueva columna" @keyup.enter="addColumn" style="width: 120px;"/>
      <button @click="addColumn">Agregar Columna</button>
      <button @click="addRow">Nuevo</button>
      <button @click="saveTable">Guardar</button>
      <button @click="reloadTable">Recargar</button>
    </header>
    <div class="sheet-wrapper">
      <KnowledgeSheet
        :rows="rows"
        :columns="columns"
        @update="onSheetUpdate"
        @deleteRow="onDeleteRow"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import KnowledgeSheet from '~/components/KnowledgeSheet.vue'

const perspectives = [
  'materials', 'objects', 'agents', 'interactions', 'environments', 'concepts', 'procedures'
]
const activePerspective = ref(perspectives[0])
const rows = ref([])
const columns = ref([])
const newColumn = ref('')

// Cargar datos
async function loadPerspective() {
  const res = await fetch(`/api/somap/${activePerspective.value}`)
  const data = await res.json()
  // Mantén columnas aunque no haya filas
  columns.value = data.columns && data.columns.length ? data.columns : columns.value.length ? columns.value : []
  rows.value = data.rows || []
}

// Añadir columna
function addColumn() {
  const colName = newColumn.value.trim()
  if (colName && !columns.value.includes(colName)) {
    columns.value.push(colName)
    // Añade la columna vacía a todas las filas existentes
    rows.value.forEach(row => { row[colName] = '' })
    newColumn.value = ''
  }
}

// Añadir fila vacía con _key único
function addRow() {
  const empty = {}
  columns.value.forEach(col => { empty[col] = '' })
  empty._key = Date.now().toString(36) + Math.random().toString(36).substr(2, 5)
  rows.value.push(empty)
}

// Guardar cambios
async function saveTable() {
  await fetch(`/api/somap/${activePerspective.value}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rows: rows.value })
  })
  await loadPerspective()
}

// Recargar la tabla
function reloadTable() {
  loadPerspective()
}

// Actualizar filas editadas
function onSheetUpdate(updatedRows) {
  rows.value = updatedRows
}

// Borrar una fila (petición DELETE al backend)
async function onDeleteRow(rowIdx) {
  const row = rows.value[rowIdx]
  if (row && row._key) {
    await fetch(`/api/somap/${activePerspective.value}/${row._key}`, { method: 'DELETE' })
  }
  rows.value.splice(rowIdx, 1)
}

onMounted(loadPerspective)
</script>

<style scoped>
/* Hereda el look general de index.vue */
.somap-container header {
  margin-bottom: 2rem;
}
</style>