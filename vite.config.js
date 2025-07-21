import { defineConfig } from 'vite';

export default defineConfig({
  preview: {
    host: true,
    port: 4173,
    strictPort: true,
    allowedHosts: [
      '.loca.lt',
      'localhost',
      '127.0.0.1'
    ]
  }
});