'use strict';

/**
 * Validation service — Zod schemas and input guards shared by middleware.
 *
 * Uploaded files are never persisted and never contain stored patient
 * identifiers; the only stored state per request is the in-memory buffer
 * forwarded to the internal ML service.
 */

const { z } = require('zod');
const { AppError } = require('../middleware/error.middleware');

/** Per-modality accepted file extensions. */
const MODALITY_EXTENSIONS = Object.freeze({
  voice: ['.csv'],
  handwriting: ['.csv'],
  gait: ['.txt', '.csv', '.tsv'],
});

/** Per-modality MIME types accepted by Multer. */
const MODALITY_MIME_TYPES = Object.freeze({
  voice: ['text/csv', 'application/vnd.ms-excel', 'text/plain', 'application/octet-stream'],
  handwriting: ['text/csv', 'application/vnd.ms-excel', 'text/plain', 'application/octet-stream'],
  gait: ['text/plain', 'text/csv', 'application/octet-stream'],
});

const ALL_MIME_TYPES = Object.freeze([
  ...new Set(Object.values(MODALITY_MIME_TYPES).flat()),
]);

/**
 * Validate that a file field conforms to the rules of its modality.
 * Throws AppError(415) on mismatch.
 */
function validateFileForModality(modality, file) {
  if (!file) {
    throw new AppError(400, `Missing required file field: ${modality}`, 'VALIDATION_ERROR');
  }
  const ext = path_extname(file.originalname).toLowerCase();
  if (!MODALITY_EXTENSIONS[modality].includes(ext)) {
    throw new AppError(
      415,
      `Invalid file type for ${modality}: expected one of ` +
        `${MODALITY_EXTENSIONS[modality].join(', ')}, got "${ext || 'none'}"`,
      'UNSUPPORTED_MEDIA_TYPE',
    );
  }
  if (file.mimetype && !MODALITY_MIME_TYPES[modality].includes(file.mimetype)) {
    throw new AppError(
      415,
      `Unsupported media type for ${modality}: ${file.mimetype}`,
      'UNSUPPORTED_MEDIA_TYPE',
    );
  }
  return file;
}

/** Small local helper so this module has no extra dependency on `path`. */
function path_extname(filename) {
  const index = filename.lastIndexOf('.');
  return index === -1 ? '' : filename.slice(index);
}

/**
 * Zod schema for the multimodal request (validated *after* Multer runs):
 * at least one modality file must be present.
 */
const multimodalRequestSchema = z
  .object({
    voice: z.any().optional(),
    handwriting: z.any().optional(),
    gait: z.any().optional(),
  })
  .refine(
    (value) =>
      ['voice', 'handwriting', 'gait'].some((m) => value[m] && value[m].buffer),
    { message: 'At least one modality file is required (voice, handwriting or gait)' },
  );

/**
 * Validate an already-parsed ML-service response body against the API
 * contract. Returns the normalized response or throws AppError(502).
 */
function normalizeMlResponse(body) {
  const shape = z.object({
    risk_band: z.enum(['low', 'borderline', 'high']),
    model_version: z.string().min(1),
    multimodal: modalityResultShape,
    voice: modalityResultShape.optional(),
    handwriting: modalityResultShape.optional(),
    gait: modalityResultShape.optional(),
  });

  const parsed = shape.safeParse(body);
  if (!parsed.success) {
    throw new AppError(502, 'ML service returned a malformed response');
  }
  return parsed.data;
}

const modalityResultShape = z.object({
  probability: z.number().min(0).max(1),
  top_features: z.array(
    z.object({
      modality: z.string().optional(),
      feature: z.string(),
      contribution: z.number(),
    }),
  ),
});

module.exports = {
  MODALITY_EXTENSIONS,
  MODALITY_MIME_TYPES,
  ALL_MIME_TYPES,
  validateFileForModality,
  multimodalRequestSchema,
  normalizeMlResponse,
};
