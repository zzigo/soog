import { defineNuxtConfig } from "nuxt/config";
import { networkInterfaces } from "node:os";

const getLocalIPv4 = () => {
  const nets = networkInterfaces();
  for (const entries of Object.values(nets)) {
    if (!entries) continue;
    for (const item of entries) {
      if (item && item.family === "IPv4" && !item.internal) {
        return item.address;
      }
    }
  }
  return null;
};

const localIPv4 = getLocalIPv4();
const devApiBase = localIPv4
  ? `http://${localIPv4}:10000/api`
  : "http://127.0.0.1:10000/api";

export default defineNuxtConfig({
  experimental: {
    appManifest: false,
    payloadExtraction: false,
    viewTransition: true,
    renderJsonPayloads: true,
  },

  ssr: true,

  css: ["~/assets/styles/global.css"], // Global CSS

  runtimeConfig: {
    public: {
      apiBase:
        process.env.NUXT_PUBLIC_API_BASE ||
        (process.env.NODE_ENV === "development"
          ? devApiBase
          : "https://soog.zztt.org/api"),
      // "0" means disabled (no client-side timeout cap).
      generateTimeoutMs: process.env.NUXT_PUBLIC_GENERATE_TIMEOUT_MS || "0",
    },
  },

  nitro: {
    preset: "node-server", // Suitable for Render deployment as a dynamic app
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
