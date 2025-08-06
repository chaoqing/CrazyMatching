import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  base: './',
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: '.',
        },
      ],
    }),
  ],
  build: {
    assetsDir: 'assets',
    rollupOptions: {
      output: {
        assetFileNames: (assetInfo) => {
          // Check if the asset is one of the ONNX Runtime WASM files
          if (assetInfo.name && assetInfo.name.startsWith('ort-wasm')) {
            // Return the original filename without a hash
            return `[name][extname]`;
          }
          // For other assets, use the default hashing behavior
          return `assets/[name]-[hash][extname]`;
        },
      },
    },
  },
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
