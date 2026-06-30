import { ref, onMounted } from 'vue'
import { useRuntimeConfig } from '#app'

export const useApi = () => {
  const config = useRuntimeConfig()
  
  // Initialize with either the configured base or the default dev fallback
  const apiBase = ref(config.public.apiBase || 'http://127.0.0.1:10000/api')

  // Heuristic for development: if we are accessing the app from a remote machine (not localhost),
  // and the API is set to localhost/127.0.0.1, we assume the API is also on that same remote machine.
  const adjustApiBase = () => {
    if (typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      if (apiBase.value.includes('localhost') || apiBase.value.includes('127.0.0.1')) {
        const actualHost = window.location.hostname
        apiBase.value = apiBase.value.replace(/localhost|127\.0\.0\.1/, actualHost)
        console.log('🔄 Adjusted apiBase for remote access:', apiBase.value)
      }
    }
  }

  // Run adjustment immediately if window is available (client-side)
  if (typeof window !== 'undefined') {
    adjustApiBase()
  }

  // Also run on mount to be safe in SSR contexts
  onMounted(() => {
    adjustApiBase()
  })

  return {
    apiBase
  }
}
