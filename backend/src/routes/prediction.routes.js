'use strict';

const { Router } = require('express');
const controller = require('../controllers/prediction.controller');
const { singleUploadChain, multimodalUploadChain } = require('../services/upload.service');

const router = Router();

// Bare /api/predict — the endpoint the React frontend calls (any subset of
// voice/handwriting/gait files); identical behaviour to /multimodal.
// /prediction is kept as an alias so either spelling works.
router.post('/', ...multimodalUploadChain(), controller.predictMultimodal);
router.post('/prediction', ...multimodalUploadChain(), controller.predictMultimodal);
router.post('/voice', ...singleUploadChain('voice'), controller.predictVoice);
router.post('/handwriting', ...singleUploadChain('handwriting'), controller.predictHandwriting);
router.post('/gait', ...singleUploadChain('gait'), controller.predictGait);
router.post('/multimodal', ...multimodalUploadChain(), controller.predictMultimodal);

module.exports = router;
