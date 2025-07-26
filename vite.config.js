import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    assetsDir: 'assets'
  },
  server: {
    allowedHosts: [
      '.loca.lt',
      'localhost',
      '127.0.0.1'
    ]
  }
});