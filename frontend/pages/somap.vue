<template>
  <div class="somap-shell">
    <NavRail />
    <SomRegistry />
  </div>
</template>

<script setup>
// /somap is the registry graph. ?view=3d opens the 3D stage, ?node=<id>&tab=<node|details|filters>
// deep-links a note; both are handled inside SomRegistry.
useHead({ title: 'SOMAP' })
</script>

<style scoped>
.somap-shell {
  box-sizing: border-box;
  height: 100vh;
  padding-left: 44px;
  background: #000;
}
</style>
