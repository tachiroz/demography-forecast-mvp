import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/train':   'http://127.0.0.1:8000',
      '/metrics': 'http://127.0.0.1:8000',
      '/preds':   'http://127.0.0.1:8000',
    },
  },  
})
