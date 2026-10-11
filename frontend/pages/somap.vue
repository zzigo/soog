<template>
  <SomRegistry />
</template>

<script setup>
// /somap is the registry graph. ?view=3d opens the 3D stage, ?node=<id>&tab=<node|details|filters>
// deep-links a note; both are handled inside SomRegistry.
useHead({ title: 'SOMAP' })
</script>
