import { computed, ref } from 'vue'
import { useLogto } from '@logto/vue'

export const useSoogAuth = () => {
  const profile = useState<Record<string, any> | null>('soog-auth-profile', () => null)
  const quota = useState<Record<string, any> | null>('soog-auth-quota', () => null)
  const localLoading = ref(false)
  const fallbackAuthenticated = ref(false)
  const fallbackLoading = ref(false)
  const config = useRuntimeConfig()

  const logto = import.meta.client ? useLogto() : null
  const isAuthenticated = logto?.isAuthenticated || fallbackAuthenticated
  const isLoading = logto?.isLoading || fallbackLoading
  const configured = computed(() => Boolean(String(config.public.logtoAppId || '').trim()))

  const authHeaders = async (): Promise<Record<string, string>> => {
    if (!logto || !isAuthenticated.value) throw new Error('Authentication required')
    const resource = String(config.public.logtoApiResource || 'https://soog.zztt.org/api')
    const [accessToken, idToken] = await Promise.all([
      logto.getAccessToken(resource),
      logto.getIdToken(),
    ])
    if (!accessToken) throw new Error('Access token unavailable')
    return {
      Authorization: `Bearer ${accessToken}`,
      ...(idToken ? { 'X-SOOG-ID-Token': idToken } : {}),
    }
  }

  const refreshProfile = async (apiBase: string) => {
    if (!logto || !isAuthenticated.value) {
      profile.value = null
      quota.value = null
      return null
    }
    localLoading.value = true
    try {
      const response = await fetch(`${apiBase}/auth/me`, {
        headers: { Accept: 'application/json', ...(await authHeaders()) },
      })
      if (!response.ok) throw new Error(`Session check failed (${response.status})`)
      const payload = await response.json()
      profile.value = payload.user || null
      quota.value = payload.quota || null
      return payload
    } finally {
      localLoading.value = false
    }
  }

  const signIn = async (google = false, postRedirectPath = '/?resume=render') => {
    if (!logto || !configured.value) throw new Error('Logto is not configured for SOOG')
    const origin = window.location.origin
    await logto.signIn({
      redirectUri: `${origin}/auth/callback`,
      postRedirectUri: `${origin}${postRedirectPath.startsWith('/') ? postRedirectPath : `/${postRedirectPath}`}`,
      ...(google
        ? { directSignIn: { method: 'social' as const, target: 'google' } }
        : {}),
    })
  }

  const signOut = async () => {
    if (!logto) return
    profile.value = null
    quota.value = null
    await logto.signOut(window.location.origin)
  }

  return {
    isAuthenticated,
    isLoading,
    configured,
    localLoading,
    profile,
    quota,
    authHeaders,
    refreshProfile,
    signIn,
    signOut,
  }
}
