import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  // configure-pages supplies the project/custom-domain base path in CI.
  base: (process.env.VITE_BASE_PATH || "/").replace(/\/+$/, "") + "/",
  // Serves physics/export.py's output (../data/*.json) at the app root, e.g.
  // /dzhanibekov_free.json -- no copying/duplicating the generated data.
  publicDir: "../data",
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  build: { rollupOptions: { input: { lab: resolve("index.html"), racket: resolve("racket.html") } } },
  test: {
    environment: "jsdom",
  },
});
