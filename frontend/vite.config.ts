import { defineConfig, loadEnv } from 'vite'

// In dev the browser talks only to the Vite server; /api and /auth are proxied to the FastAPI backend, so the
// refresh-token cookie (path=/auth) is same-origin and no CORS is involved. Override the target with API_TARGET
// in frontend/.env.local if the backend is not on localhost:8000.
export default defineConfig(({ mode }) => {
  const target = loadEnv(mode, '.', '').API_TARGET || 'http://localhost:8000'
  const proxy = { '/api': { target }, '/auth': { target } }
  return {
    server: { port: 5173, strictPort: true, proxy },
    preview: { port: 5173, proxy },
  }
})
