export default defineNuxtConfig({
  css: ['~/assets/styles/global.css'], // Global CSS

  runtimeConfig: {
    public: {
      apiBase: 'https://soog.onrender.com/api', // Public backend URL for production
    },
  },

  nitro: {
    preset: 'node-server', // Suitable for Render deployment as a dynamic app
  },

  vite: {
    server: {
      fs: {
        allow: ['node_modules'], // Ensure Vite can access necessary modules
      },
    },
  },

  devServer: {
    port: 3000, // Default development server port (optional)
  },

  compatibilityDate: '2025-01-13', // Explicit compatibility date
  export default defineNuxtConfig({
    app: {
      head: {
        link: [
          { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' }, // Default favicon
          { rel: 'icon', type: 'image/png', href: '/favicon-32x32.png', sizes: '32x32' }, // 32x32 favicon
          { rel: 'icon', type: 'image/png', href: '/favicon-16x16.png', sizes: '16x16' }, // 16x16 favicon
          { rel: 'apple-touch-icon', href: '/apple-touch-icon.png', sizes: '180x180' }, // iOS devices
          { rel: 'manifest', href: '/site.webmanifest' }, // PWA manifest
        ],
      },
    },
});
});
