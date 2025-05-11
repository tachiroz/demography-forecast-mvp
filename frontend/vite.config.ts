import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/upload-train': 'http://127.0.0.1:8000',
      '/forecast':     'http://127.0.0.1:8000',
    },
  },  
})
