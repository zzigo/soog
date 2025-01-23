import { defineNuxtConfig } from "nuxt/config";

export default defineNuxtConfig({
  experimental: {
    payloadExtraction: false,
    viewTransition: true,
    renderJsonPayloads: true,
  },

  ssr: true,

  css: ["~/assets/styles/global.css"], // Global CSS

  runtimeConfig: {
    public: {
      apiBase: "https://soog.onrender.com/api", // Public backend URL for production
    },
  },

  nitro: {
    preset: "node-server", // Suitable for Render deployment as a dynamic app
  },

  vite: {
    server: {
      fs: {
        allow: ["node_modules"], // Ensure Vite can access necessary modules
      },
    },
  },

  devServer: {
    port: 3000, // Default development server port (optional)
  },

  compatibilityDate: "2025-01-13", // Explicit compatibility date

  app: {
    head: {
      meta: [{ name: "theme-color", content: "#4CAF50" }],
      link: [
        { rel: "icon", type: "image/x-icon", href: "/favicon.ico" },
        {
          rel: "icon",
          type: "image/png",
          sizes: "32x32",
          href: "/favicon-32x32.png",
        },
        {
          rel: "icon",
          type: "image/png",
          sizes: "16x16",
          href: "/favicon-16x16.png",
        },
        {
          rel: "apple-touch-icon",
          sizes: "180x180",
          href: "/apple-touch-icon.png",
        },
        { rel: "manifest", href: "/manifest.json" },
      ],
    },
  },

  appConfig: {
    name: "SOOG",
    description: "Speculative Organology Organogram Generator",
    theme: {
      dark: true,
      colors: {
        primary: "#4CAF50",
        background: "#1a1a1a",
      },
    },
  },
});
