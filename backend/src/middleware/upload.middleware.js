'use strict';

/**
 * Multer upload middleware.
 *
 * Files are kept in *memory only* (memoryStorage) and streamed to the
 * internal ML service per request — nothing is written to disk, so no
 * uploaded data is ever persisted.
 */

const multer = require('multer');
const { getConfig } = require('../config/config');
const { AppError } = require('./error.middleware');
const { ALL_MIME_TYPES } = require('../services/validation.service');

function buildMulter() {
  const config = getConfig();
  return multer({
    storage: multer.memoryStorage(),
    limits: {
      fileSize: config.maxFileSize,
      files: 3, // at most voice + handwriting + gait
      fields: 4,
    },
    fileFilter: (req, file, cb) => {
      if (!ALL_MIME_TYPES.includes(file.mimetype)) {
        return cb(new AppError(415, `Unsupported media type: ${file.mimetype}`, 'UNSUPPORTED_MEDIA_TYPE'));
      }
      return cb(null, true);
    },
  });
}

/** Upload middleware for a single-modality field named `file`. */
function uploadSingle() {
  return buildMulter().single('file');
}

/** Upload middleware for the multimodal endpoint (voice/handwriting/gait). */
function uploadMultimodal() {
  return buildMulter().fields([
    { name: 'voice', maxCount: 1 },
    { name: 'handwriting', maxCount: 1 },
    { name: 'gait', maxCount: 1 },
  ]);
}

module.exports = { uploadSingle, uploadMultimodal };
