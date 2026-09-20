'use strict';

/**
 * Model controller — surfaces training metadata and CV metrics of the
 * deployed models, proxied from the internal ML service.
 */

const mlService = require('../services/ml.service');
const { asyncHandler } = require('../middleware/error.middleware');

/** GET /api/model-info */
async function getModelInfo(req, res) {
  const info = await mlService.getModelInfo();
  return res.status(200).json(info);
}

module.exports = { getModelInfo: asyncHandler(getModelInfo) };
