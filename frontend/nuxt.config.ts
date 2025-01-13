export default defineNuxtConfig({
  css: ['~/assets/styles/global.css'], // Include global CSS for styling

  runtimeConfig: {
    public: {
      apiBase: 'https://soog.onrender.com/api', // Backend API base URL for production
    },
  },

  nitro: {
    preset: 'node-server', // Use Nitro's node-server preset for a dynamic app
  },

  vite: {
    server: {
      fs: {
        allow: ['node_modules'], // Ensure Vite can access required modules
      },
    },
  },
});