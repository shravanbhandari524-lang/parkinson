'use strict';

/**
 * Prediction controller — forwards validated uploads to the internal ML
 * service and returns the normalized prediction contract.
 */

const mlService = require('../services/ml.service');
const { asyncHandler } = require('../middleware/error.middleware');

const SINGLE_MODALITIES = ['voice', 'handwriting', 'gait'];

/**
 * POST /api/predict/:modality  (voice | handwriting | gait)
 * Expects `req.file` (set by the upload middleware).
 */
const predictSingle = (modality) =>
  asyncHandler(async (req, res) => {
    const result = await mlService.predictModality(modality, req.file);
    return res.status(200).json(result);
  });

/**
 * POST /api/predict/multimodal
 * Expects `req.modalityFiles` — any subset of {voice, handwriting, gait}.
 * Missing modalities are simply omitted; the ML service renormalizes fusion.
 */
const predictMultimodal = asyncHandler(async (req, res) => {
  const files = req.modalityFiles || {};
  const result = await mlService.predictMultimodal(files);
  return res.status(200).json(result);
});

module.exports = {
  SINGLE_MODALITIES,
  predictVoice: predictSingle('voice'),
  predictHandwriting: predictSingle('handwriting'),
  predictGait: predictSingle('gait'),
  predictMultimodal,
};
