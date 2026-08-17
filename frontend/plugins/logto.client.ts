import { createLogto, UserScope } from '@logto/vue'

export default defineNuxtPlugin((nuxtApp) => {
  const config = useRuntimeConfig()
  nuxtApp.vueApp.use(createLogto, {
    endpoint: String(config.public.logtoEndpoint || 'https://logto.zztt.org'),
    appId: String(config.public.logtoAppId || 'soog-not-configured'),
    resources: [String(config.public.logtoApiResource || 'https://soog.zztt.org/api')],
    scopes: [UserScope.Email],
  })
})
