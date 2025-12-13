import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5175,  // Use 5175 for Google OAuth compatibility
    strictPort: true,
    watch: {
      usePolling: true,
    },
    headers: {
      // Required for PMTiles range requests
      'Accept-Ranges': 'bytes',
      // Allow cross-origin requests (needed for PMTiles)
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
      'Access-Control-Allow-Headers': 'Range',
      // Disable caching in dev to prevent HMR reload issues
      'Cache-Control': 'no-store',
    },
    // Configure HMR for WebSocket connections
    hmr: {
      host: 'localhost',
      port: 5175,
    },
  },
  // Optimize asset handling
  build: {
    assetsInlineLimit: 0, // Don't inline any assets
    rollupOptions: {
      output: {
        // Ensure PMTiles files are handled correctly
        assetFileNames: (assetInfo) => {
          if (assetInfo.name && assetInfo.name.endsWith('.pmtiles')) {
            return 'assets/[name]-[hash][extname]';
          }
          return 'assets/[name]-[hash][extname]';
        },
      },
    },
  },
})
