export default defineNuxtConfig({
  css: ['~/assets/styles/global.css'],
  runtimeConfig: {
    public: {
      apiBase: process.env.API_BASE || 'http://127.0.0.1:2604/api',
    },
  },

  nitro: {
    devProxy: {
      '/api': {
        target: 'http://127.0.0.1:2604', // Proxying to your Flask backend
        changeOrigin: true,
      },
      '/workers/worker-javascript.js': {
        target: 'http://127.0.0.1:3001', // Adjust this to where the worker is served from
        changeOrigin: true,
      },
    },
  },

  compatibilityDate: '2025-01-11', // Ensures compatibility with current API expectations

  vite: {
    server: {
      fs: {
        allow: ['node_modules/ace-builds'], // Explicitly allow Ace.js
      },
    },
  },

    server: {
      port: 3001, // Force Nuxt to use port 3001
    },
  
  devServer: {
    port: 3001, // Fallback for Nitro
  },

  compatibilityDate: '2025-01-11',
});