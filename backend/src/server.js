'use strict';

/**
 * Server entry point: boots the API, and shuts down gracefully on
 * SIGTERM/SIGINT so in-flight uploads finish before exit.
 */

const { getConfig } = require('./config/config');
const { createApp } = require('./app');

function startServer() {
  const config = getConfig();
  const app = createApp();

  const server = app.listen(config.port, () => {
    console.log(
      `[backend] listening on port ${config.port} (${config.nodeEnv}) — ML service: ${config.mlServiceUrl}`,
    );
  });

  const shutdown = (signal) => {
    console.log(`[backend] ${signal} received — closing server`);
    server.close(() => process.exit(0));
    // hard exit if connections do not drain in time
    setTimeout(() => process.exit(1), 10000).unref();
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  return server;
}

if (require.main === module) {
  startServer();
}

module.exports = { startServer };
