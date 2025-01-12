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
});