'use strict';

/**
 * ML service client — the ONLY place the Express backend talks to the
 * internal Python ML service. All probabilities come from here; nothing is
 * hard-coded or cached across requests.
 */

const axios = require('axios');
const { getConfig } = require('../config/config');
const { AppError } = require('../middleware/error.middleware');
const { normalizeMlResponse } = require('./validation.service');

function createClient() {
  const config = getConfig();
  return axios.create({
    baseURL: config.mlServiceUrl,
    timeout: config.mlTimeoutMs,
    headers: { Accept: 'application/json' },
    // uploads never contain more than the in-memory buffers we just received
    maxBodyLength: Infinity,
    maxContentLength: 10 * 1024 * 1024,
    validateStatus: null, // status mapping happens in toAppError
  });
}

/** Map transport / upstream failures to typed AppErrors. */
function toAppError(err) {
  if (err instanceof AppError) return err;

  if (err.code === 'ECONNREFUSED' || err.code === 'ENOTFOUND') {
    return new AppError(
      503,
      'ML service is unavailable',
      'ML_SERVICE_UNAVAILABLE',
    ).withCause(err);
  }
  if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
    return new AppError(504, 'ML service request timed out', 'ML_SERVICE_TIMEOUT').withCause(err);
  }
  if (err.response) {
    const status = err.response.status;
    const detail =
      err.response.data && (err.response.data.detail || err.response.data.message);
    return new AppError(
      status >= 500 ? 502 : status,
      detail || 'ML service returned an error',
      'ML_SERVICE_ERROR',
    ).withCause(err);
  }
  return new AppError(502, 'Failed to reach the ML service', 'ML_SERVICE_ERROR').withCause(err);
}

/** Forward an uploaded in-memory file to the ML service. */
function fileToFormEntry(file) {
  const blob = new Blob([file.buffer], { type: file.mimetype || 'application/octet-stream' });
  return { blob, filename: file.originalname || 'upload' };
}

const mlService = {
  /** GET /health on the ML service. Returns {status, model_version, models_loaded}. */
  async getHealth() {
    const client = createClient();
    try {
      const response = await client.get('/health');
      if (response.status >= 500) {
        // degraded but reachable — surface as operational data, not a crash
        return { reachable: true, healthy: false, ...response.data };
      }
      return { reachable: true, healthy: true, ...response.data };
    } catch (err) {
      throw toAppError(err);
    }
  },

  /** GET /model-info — training metadata + cross-validated metrics. */
  async getModelInfo() {
    const client = createClient();
    try {
      const response = await client.get('/model-info');
      if (response.status >= 400) {
        throw toAppError({ response });
      }
      return response.data;
    } catch (err) {
      throw toAppError(err);
    }
  },

  /**
   * POST /predict/<modality> with a single `file` upload.
   * Returns the normalized prediction response (see API contract).
   */
  async predictModality(modality, file) {
    const client = createClient();
    const { blob, filename } = fileToFormEntry(file);
    const form = new FormData();
    form.append('file', blob, filename);

    try {
      const response = await client.post(`/predict/${modality}`, form);
      if (response.status >= 400) {
        throw toAppError({ response });
      }
      return normalizeMlResponse(response.data);
    } catch (err) {
      throw toAppError(err);
    }
  },

  /**
   * POST /predict/multimodal with any subset of voice/handwriting/gait files.
   * The ML service renormalizes fusion for the supplied modalities only.
   */
  async predictMultimodal(files) {
    const client = createClient();
    const form = new FormData();
    for (const modality of ['voice', 'handwriting', 'gait']) {
      if (files[modality]) {
        const { blob, filename } = fileToFormEntry(files[modality]);
        form.append(modality, blob, filename);
      }
    }

    try {
      const response = await client.post('/predict/multimodal', form);
      if (response.status >= 400) {
        throw toAppError({ response });
      }
      return normalizeMlResponse(response.data);
    } catch (err) {
      throw toAppError(err);
    }
  },
};

module.exports = mlService;
