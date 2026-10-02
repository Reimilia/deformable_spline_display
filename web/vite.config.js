import { defineConfig } from "vite";
export default defineConfig({
  base: (process.env.VITE_BASE_PATH || "/").replace(/\/+$/, "") + "/",
  test: { environment: "jsdom" },
});
