import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        // Critical: disable proxy buffering for SSE streaming
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            // If the response is SSE, disable buffering
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              proxyRes.headers['Cache-Control'] = 'no-cache'
              proxyRes.headers['X-Accel-Buffering'] = 'no'
            }
          })
        },
      },
      '/static': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})
