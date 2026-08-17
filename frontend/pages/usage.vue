<template>
  <main class="usage-route">
    <div v-if="checkingProfile || (!profileChecked && isLoading)" class="usage-route-state">Loading admin access…</div>

    <section v-else-if="!isAuthenticated" class="access-card">
      <div class="access-kicker">SOOG ADMIN</div>
      <h1>Render usage</h1>
      <p>Sign in with an administrator account to review daily and weekly renders and manage user quotas.</p>
      <p v-if="error" class="access-error">{{ error }}</p>
      <div class="access-actions">
        <button class="primary" :disabled="signingIn || !configured" @click="beginSignIn(true)">
          {{ signingIn ? 'Connecting…' : 'Continue with Google' }}
        </button>
        <button :disabled="signingIn || !configured" @click="beginSignIn(false)">Use another Logto sign-in</button>
        <NuxtLink to="/">Back to SOOG</NuxtLink>
      </div>
    </section>

    <section v-else-if="!profile?.is_admin" class="access-card">
      <div class="access-kicker denied">ACCESS RESTRICTED</div>
      <h1>Administrator rights required</h1>
      <p>{{ profile?.email || profile?.name || profile?.subject || 'This account' }} is signed in without SOOG administrator rights.</p>
      <p v-if="error" class="access-error">{{ error }}</p>
      <NuxtLink to="/">Back to SOOG</NuxtLink>
    </section>

    <UsageDashboard
      v-else
      :model-value="true"
      :api-base="apiBase"
      @close="navigateTo('/')"
    />
  </main>
</template>

<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = computed(() => String(config.public.apiBase || '/api').replace(/\/$/, ''))
const signingIn = ref(false)
const checkingProfile = ref(false)
const profileChecked = ref(false)
const error = ref('')
const {
  isAuthenticated,
  isLoading,
  configured,
  profile,
  refreshProfile,
  signIn,
} = useSoogAuth()

useHead({ title: 'Render usage · SOOG' })

const beginSignIn = async (google: boolean) => {
  signingIn.value = true
  error.value = ''
  try {
    await signIn(google, '/usage')
  } catch (cause: any) {
    error.value = cause?.message || 'Unable to start sign-in.'
    signingIn.value = false
  }
}

watch(
  [isAuthenticated, isLoading],
  async ([authenticated, authBusy]) => {
    if (!authenticated) {
      profileChecked.value = false
      profile.value = null
      return
    }
    if (authBusy || profileChecked.value) return
    profileChecked.value = true
    checkingProfile.value = true
    error.value = ''
    try {
      await refreshProfile(apiBase.value)
    } catch (cause: any) {
      error.value = cause?.message || 'Unable to validate the SOOG session.'
      profile.value = null
    } finally {
      checkingProfile.value = false
    }
  },
  { immediate: true },
)
</script>

<style scoped>
.usage-route {
  min-height: 100vh;
  background: #050505;
  color: rgba(255, 255, 255, 0.72);
  font-family: 'IBM Plex Mono', monospace;
}

.usage-route-state,
.access-card {
  position: absolute;
  top: 50%;
  left: 50%;
  width: min(440px, calc(100% - 40px));
  transform: translate(-50%, -50%);
}

.usage-route-state {
  text-align: center;
  font-size: 12px;
}

.access-card {
  padding: 30px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: #080808;
}

.access-kicker {
  color: #4caf50;
  font-size: 10px;
  letter-spacing: 0.16em;
}

.access-kicker.denied,
.access-error {
  color: #ef9a9a;
}

h1 {
  margin: 10px 0 14px;
  color: white;
  font-size: 22px;
  font-weight: 400;
}

p {
  margin: 0 0 18px;
  font-size: 12px;
  line-height: 1.65;
}

.access-actions {
  display: grid;
  gap: 8px;
}

button,
a {
  min-height: 42px;
  padding: 0 14px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: transparent;
  color: rgba(255, 255, 255, 0.72);
  font: inherit;
  font-size: 12px;
  text-decoration: none;
  cursor: pointer;
}

a {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

button.primary {
  border-color: #4caf50;
  background: #4caf50;
  color: #071007;
}

button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
</style>
