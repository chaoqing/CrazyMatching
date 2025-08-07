import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  server: {
    allowedHosts: [
      '.loca.lt',
      'localhost',
      '127.0.0.1'
    ]
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'], // Exclude onnxruntime-web from Vite's dependency optimization
  },
});
