import { defineConfig } from 'vite';
import fs from 'node:fs';
import path from 'node:path';

export default defineConfig(({ mode }) => ({
  base: './',
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
  plugins: [
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
  optimizeDeps: {
    exclude: ['onnxruntime-web'], // Exclude onnxruntime-web from Vite's dependency optimization
  },
}));
