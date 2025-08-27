import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig(({ mode }) => ({
  base: './',
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'tensorflow': ['@tensorflow/tfjs'],
          'onnxruntime': ['onnxruntime-web'],
        }
      }
    }
  },
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/opencv.js/opencv.js',
          dest: 'assets'
        }
      ]
    }),
    {
      name: "log-debug-data-api",
      // Add a custom middleware for logging in development mode
      configureServer(server) {
        if (mode === 'development') {
          server.middlewares.use('/log-debug-data', async (req, res) => {
            if (req.method === 'POST') {
              let body = '';
              req.on('data', (chunk) => {
                body += chunk.toString();
              });
              req.on('end', () => {
                try {
                  const data = JSON.parse(body);
                  const logDir = path.resolve(__dirname, 'debug-logs');
                  if (!fs.existsSync(logDir)) {
                    fs.mkdirSync(logDir);
                  }
                  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                  const logFilePath = path.join(logDir, `log-${timestamp}.json`);
                  fs.writeFileSync(logFilePath, JSON.stringify(data, null, 2));
                  res.statusCode = 200;
                  res.end('Log received');
                } catch (error) {
                  console.error('Failed to parse or write log data:', error);
                  res.statusCode = 500;
                  res.end('Error logging data');
                }
              });
            } else {
              res.statusCode = 405;
              res.end('Method Not Allowed');
            }
          });
        }
      }
    }
  ],
  server: {
    allowedHosts: [
      '.loca.lt',
      'localhost',
      '192.168.1.3',
      '127.0.0.1'
    ],
    watch: {
      ignored: [
        '**/.venv/**',
        '**/model/data/**',
        '**/.git/**',
        '**/node_modules/**'
      ]
    },
  },

  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
}));
