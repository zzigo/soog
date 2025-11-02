<template>
  <div class="knowledge-sheet">
    <table>
      <thead>
        <tr>
          <th v-for="col in columns" :key="col">{{ col }}</th>
          <th>Acciones</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, rowIdx) in rows" :key="rowIdx">
          <td v-for="col in columns" :key="col">
            <input v-model="editRows[rowIdx][col]" @change="emitUpdate" />
          </td>
          <td>
            <button @click="emitDelete(rowIdx)">🗑️</button>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
import { watch, reactive, toRefs } from 'vue'
const props = defineProps({
  rows: { type: Array, default: () => [] },
  columns: { type: Array, default: () => [] },
})
const emit = defineEmits(['update', 'deleteRow'])

const editRows = reactive(props.rows.map(row => ({ ...row })))

watch(() => props.rows, (newRows) => {
  // Deep copy to avoid mutating parent
  editRows.splice(0, editRows.length, ...newRows.map(row => ({ ...row })))
})

function emitUpdate() {
  emit('update', editRows.map(row => ({ ...row })))
}
function emitDelete(idx) {
  emit('deleteRow', idx)
}
</script>

<style scoped>
.knowledge-sheet tr.active-row {
  background: #2d2d2d;
}
</style>
