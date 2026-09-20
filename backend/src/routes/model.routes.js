'use strict';

const { Router } = require('express');
const { getModelInfo } = require('../controllers/model.controller');

const router = Router();

router.get('/model-info', getModelInfo);

module.exports = router;
