<template>
  <div class="concert-page">
    <ClientOnly>
      <ConcertSimulator />
    </ClientOnly>
  </div>
</template>

<script setup>
import ConcertSimulator from '~/components/ConcertSimulator.vue'

definePageMeta({
  layout: false
})
</script>

<style scoped>
.concert-page {
  width: 100vw;
  height: 100vh;
  background: #000;
}
</style>
