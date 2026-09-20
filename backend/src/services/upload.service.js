'use strict';

/**
 * Upload service — owns the Multer wiring and per-modality file rules.
 * Kept separate from the middleware layer so controllers can be tested
 * without HTTP plumbing.
 */

const { uploadSingle, uploadMultimodal } = require('../middleware/upload.middleware');
const { validateFileForModality, multimodalRequestSchema } = require('./validation.service');
const { AppError } = require('../middleware/error.middleware');

const SINGLE_MODALITIES = ['voice', 'handwriting', 'gait'];

/**
 * Build middleware chain for a single-modality upload endpoint.
 * Multer parses the multipart body into memory; the modality rules then
 * verify extension/MIME before the file reaches the controller.
 */
function singleUploadChain(modality) {
  return [
    uploadSingle(),
    (req, res, next) => {
      try {
        validateFileForModality(modality, req.file);
        return next();
      } catch (err) {
        return next(err);
      }
    },
  ];
}

/**
 * Multimodal middleware: parse any subset of modality files, validate each
 * against its own rules, then enforce "at least one modality supplied".
 */
function multimodalUploadChain() {
  return [
    uploadMultimodal(),
    (req, res, next) => {
      try {
        const files = req.files || {};
        for (const modality of SINGLE_MODALITIES) {
          if (files[modality]) {
            validateFileForModality(modality, files[modality][0]);
          }
        }
        const flat = {};
        for (const modality of SINGLE_MODALITIES) {
          if (files[modality]) flat[modality] = files[modality][0];
        }
        const parsed = multimodalRequestSchema.safeParse(flat);
        if (!parsed.success) {
          throw new AppError(400, parsed.error.issues[0].message, 'VALIDATION_ERROR');
        }
        req.modalityFiles = flat;
        return next();
      } catch (err) {
        return next(err);
      }
    },
  ];
}

module.exports = { singleUploadChain, multimodalUploadChain };
