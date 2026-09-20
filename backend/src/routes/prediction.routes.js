'use strict';

const { Router } = require('express');
const controller = require('../controllers/prediction.controller');
const { singleUploadChain, multimodalUploadChain } = require('../services/upload.service');

const router = Router();

router.post('/voice', ...singleUploadChain('voice'), controller.predictVoice);
router.post('/handwriting', ...singleUploadChain('handwriting'), controller.predictHandwriting);
router.post('/gait', ...singleUploadChain('gait'), controller.predictGait);
router.post('/multimodal', ...multimodalUploadChain(), controller.predictMultimodal);

module.exports = router;
