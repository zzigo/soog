<template>
  <main class="callback-page">
    <p v-if="isLoading">Completing sign-in…</p>
    <p v-else-if="error">Unable to sign in: {{ error.message }}</p>
    <p v-else>Returning to SOOG…</p>
  </main>
</template>

<script setup lang="ts">
import { useHandleSignInCallback } from '@logto/vue'

const { isLoading, error } = useHandleSignInCallback()
</script>

<style scoped>
.callback-page {
  min-height: 100vh;
  display: grid;
  place-items: center;
  background: #050505;
  color: rgba(255, 255, 255, 0.72);
  font: 13px/1.5 'IBM Plex Mono', monospace;
}
</style>
