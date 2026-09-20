'use strict';

const { Router } = require('express');
const mlService = require('../services/ml.service');
const { asyncHandler } = require('../middleware/error.middleware');

const router = Router();

/**
 * GET /api/health
 * 200 when the backend can reach a healthy ML service; 503 otherwise, so
 * load balancers can treat the whole stack as down.
 */
router.get(
  '/health',
  asyncHandler(async (req, res) => {
    const ml = await mlService.getHealth();
    const statusCode = ml.healthy ? 200 : 503;
    return res.status(statusCode).json({
      status: ml.healthy ? 'ok' : 'degraded',
      service: 'backend',
      ml_service: ml,
      timestamp: new Date().toISOString(),
    });
  }),
);

module.exports = router;
