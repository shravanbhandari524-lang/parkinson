'use strict';

const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const morgan = require('morgan');
const { getConfig } = require('./config/config');
const { notFoundHandler, errorHandler } = require('./middleware/error.middleware');

const healthRoutes = require('./routes/health.routes');
const modelRoutes = require('./routes/model.routes');
const predictionRoutes = require('./routes/prediction.routes');

/** Build the fully wired Express app (exported for Supertest). */
function createApp() {
  const config = getConfig();
  const app = express();

  // -- security ----------------------------------------------------------
  app.disable('x-powered-by');
  app.use(
    helmet({
      // this is a pure JSON API — CSP for browsers adds no value here
      contentSecurityPolicy: false,
      crossOriginResourcePolicy: { policy: 'same-site' },
    }),
  );
  app.use(
    cors({
      origin: config.corsOrigin,
      methods: ['GET', 'POST'],
      credentials: true,
    }),
  );

  // -- parsing & logging ---------------------------------------------------
  app.use(express.json({ limit: '1mb' }));
  app.use(morgan('dev', { skip: () => config.nodeEnv === 'test' }));

  // -- routes --------------------------------------------------------------
  app.use('/api', healthRoutes);
  app.use('/api', modelRoutes);
  app.use('/api/predict', predictionRoutes);

  // -- errors --------------------------------------------------------------
  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}

module.exports = { createApp };
