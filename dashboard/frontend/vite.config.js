import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base:'./' makes every asset URL relative, which is what lets the built
// app work unchanged under SageMaker's /jupyterlab/default/proxy/8501/
// path prefix as well as at the server root locally.
export default defineConfig({
  plugins: [react()],
  base: './',
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    proxy: {
      '/api': 'http://localhost:8501',
      '/ws': { target: 'ws://localhost:8501', ws: true },
    },
  },
})
