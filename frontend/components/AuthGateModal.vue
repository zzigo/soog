<template>
  <Teleport to="body">
    <div v-if="modelValue" class="auth-overlay" @click.self="emit('close')">
      <section class="auth-card" role="dialog" aria-modal="true" aria-labelledby="auth-title">
        <button class="auth-close" type="button" aria-label="Close" @click="emit('close')">×</button>
        <div class="auth-kicker">SOOG RENDER ACCESS</div>
        <h2 id="auth-title">Sign in to render</h2>
        <p>Browsing SOOG, reviewing the gallery, and generating prompts are free. Sign-in is only required when a render starts.</p>
        <p v-if="error" class="auth-error">{{ error }}</p>
        <div class="auth-actions">
          <button class="auth-primary" type="button" :disabled="loading || !configured" @click="emit('google')">
            {{ loading ? 'Connecting…' : 'Continue with Google' }}
          </button>
          <button class="auth-secondary" type="button" :disabled="loading || !configured" @click="emit('logto')">
            Use another Logto sign-in
          </button>
        </div>
        <small v-if="!configured">The SOOG Logto application is not configured yet.</small>
      </section>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
defineProps<{
  modelValue: boolean
  loading?: boolean
  configured?: boolean
  error?: string
}>()

const emit = defineEmits<{
  close: []
  google: []
  logto: []
}>()
</script>

<style scoped>
.auth-overlay {
  position: fixed;
  inset: 0;
  z-index: 5000;
  display: grid;
  place-items: center;
  padding: 20px;
  background: rgba(0, 0, 0, 0.82);
  backdrop-filter: blur(8px);
}

.auth-card {
  position: relative;
  width: min(420px, 100%);
  padding: 30px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: #080808;
  color: rgba(255, 255, 255, 0.8);
  font-family: 'IBM Plex Mono', monospace;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.55);
}

.auth-kicker {
  margin-bottom: 12px;
  color: #4caf50;
  font-size: 10px;
  letter-spacing: 0.16em;
}

h2 {
  margin: 0 0 14px;
  color: #fff;
  font-size: 20px;
  font-weight: 500;
}

p {
  margin: 0 0 18px;
  color: rgba(255, 255, 255, 0.55);
  font-size: 12px;
  line-height: 1.65;
}

.auth-error {
  color: #ef9a9a;
}

.auth-actions {
  display: grid;
  gap: 8px;
}

.auth-primary,
.auth-secondary {
  min-height: 42px;
  border-radius: 0;
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}

.auth-primary {
  border: 1px solid #4caf50;
  background: #4caf50;
  color: #071007;
}

.auth-secondary {
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: transparent;
  color: rgba(255, 255, 255, 0.68);
}

button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.auth-close {
  position: absolute;
  top: 12px;
  right: 14px;
  border: 0;
  background: transparent;
  color: rgba(255, 255, 255, 0.42);
  font-size: 22px;
  cursor: pointer;
}

small {
  display: block;
  margin-top: 12px;
  color: rgba(255, 255, 255, 0.35);
  font-size: 10px;
}
</style>
