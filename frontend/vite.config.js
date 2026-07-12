import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// API paths that should be proxied to the FastAPI backend during `npm run dev`.
// In production the same server hosts both the API and these static files, so
// the app just uses same-origin relative URLs.
const apiPaths = [
  '/chat',
  '/persona',
  '/personas',
  '/memories',
  '/history',
  '/health',
  '/stt',
  '/tts',
]

export default defineConfig({
  plugins: [react()],
  base: '/',
  server: {
    host: true, // expose dev server on the LAN too
    port: 5173,
    proxy: Object.fromEntries(
      apiPaths.map((p) => [p, { target: 'http://localhost:8000', changeOrigin: true }])
    ),
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
