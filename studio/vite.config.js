import { defineConfig } from "vite";

// The Python backend (server.py) does the model inference; Vite just proxies /api to it.
export default defineConfig({
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});
