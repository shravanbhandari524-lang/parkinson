import axios from 'axios';
import { MODALITY_KEYS } from '../utils/modalityMeta.js';

/**
 * Axios client for the Express analysis API.
 *
 * The backend base URL comes exclusively from VITE_API_URL (see .env.example).
 * No host is hard-coded anywhere in the application.
 *
 * Contract (also documented in .env.example):
 *   POST /predict      multipart/form-data { voice?, handwriting?, gait? }
 *   GET  /model-info
 *   GET  /health
 */

export const DEFAULT_TIMEOUT_MS = 120000; // inference + SHAP can be slow

/** Parse and normalise VITE_API_URL. Returns null when not configured. */
export function getApiBaseUrl(env = import.meta.env) {
  const raw = env?.VITE_API_URL;
  if (typeof raw !== 'string' || !raw.trim()) return null;
  return raw.trim().replace(/\/+$/, '');
}

function createClient() {
  const baseURL = getApiBaseUrl();
  if (!baseURL) {
    // Dev-friendly fallback so the app still boots without .env; requests
    // will hit the same origin and surface a clear error in the UI.
    // eslint-disable-next-line no-console
    console.warn(
      '[api] VITE_API_URL is not set. Falling back to same-origin "/api". ' +
        'Copy .env.example to .env and set VITE_API_URL to your Express backend.',
    );
    return axios.create({ baseURL: '/api', timeout: DEFAULT_TIMEOUT_MS });
  }
  return axios.create({ baseURL, timeout: DEFAULT_TIMEOUT_MS });
}

export const client = createClient();

/** Error with a user-facing message extracted from an Axios failure. */
export class ApiError extends Error {
  constructor(message, { status = null, code = null, cause = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.cause = cause;
  }
}

function messageFromResponseData(data) {
  if (!data) return null;
  if (typeof data === 'string') return data;
  for (const key of ['error', 'message', 'detail', 'details']) {
    const value = data[key];
    if (typeof value === 'string' && value.trim()) return value;
    if (Array.isArray(value) && value.length) {
      const parts = value.map((item) =>
        typeof item === 'string' ? item : item?.msg ?? item?.message,
      );
      const joined = parts.filter(Boolean).join('; ');
      if (joined) return joined;
    }
  }
  return null;
}

/** Convert any thrown value into an ApiError with an actionable message. */
export function toApiError(error) {
  if (error instanceof ApiError) return error;
  if (axios.isAxiosError?.(error)) {
    const status = error.response?.status ?? null;
    const serverMessage = messageFromResponseData(error.response?.data);
    if (error.response) {
      return new ApiError(
        serverMessage ?? `The analysis API returned an error (HTTP ${status}).`,
        { status, code: error.code ?? null, cause: error },
      );
    }
    return new ApiError(
      'Cannot reach the analysis API. Verify that the Express backend is running and that VITE_API_URL is configured correctly.',
      { status: null, code: error.code ?? null, cause: error },
    );
  }
  return new ApiError(error?.message ?? 'Unexpected error during the request.', {
    cause: error,
  });
}

/* ------------------------------------------------------------------ */
/* Response normalisation                                              */
/* ------------------------------------------------------------------ */

function pick(obj, ...keys) {
  for (const key of keys) {
    const value = obj?.[key];
    if (value !== undefined && value !== null && value !== '') return value;
  }
  return undefined;
}

function normaliseExplanations(raw) {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((item) => {
      const feature = pick(item ?? {}, 'feature', 'name', 'feature_name');
      const shap = pick(item ?? {}, 'shap', 'shap_value', 'contribution', 'value');
      const featureValue = pick(item ?? {}, 'value', 'feature_value', 'featureValue');
      return {
        feature: typeof feature === 'string' ? feature : String(feature ?? ''),
        shap: Number(shap),
        value: featureValue === undefined ? null : Number(featureValue),
      };
    })
    .filter(
      (item) =>
        item.feature &&
        Number.isFinite(item.shap),
    );
}

/**
 * Map the API prediction payload onto the shape used by the UI.
 * Missing fields stay null/[] — the frontend never fabricates values.
 */
export function normalizePrediction(raw = {}) {
  const root = raw ?? {};
  const modalityBlock = pick(root, 'modalities', 'per_modality', 'probabilities') ?? {};

  const modalities = {};
  for (const key of MODALITY_KEYS) {
    const block = modalityBlock[key] ?? root[key] ?? {};
    const probability = pick(block, 'probability', 'p_pd', 'prob');
    modalities[key] = {
      available: pick(block, 'available') ?? (probability !== undefined),
      probability: probability === undefined ? null : Number(probability),
      contribution: (() => {
        const c = pick(block, 'contribution', 'weight', 'importance');
        return c === undefined ? null : Number(c);
      })(),
    };
  }

  const fusionBlock = pick(root, 'fusion', 'multimodal') ?? {};
  const overallProbability = pick(
    root,
    'overall_probability',
  ) ?? pick(pick(root, 'overall') ?? {}, 'probability');

  const explanationsRaw = pick(root, 'explanations', 'shap', 'explanation') ?? {};
  const explanations = {};
  for (const key of [...MODALITY_KEYS, 'multimodal']) {
    explanations[key] = normaliseExplanations(explanationsRaw[key]);
  }

  return {
    modelVersion: pick(root, 'model_version', 'modelVersion') ?? null,
    timestamp:
      pick(root, 'timestamp', 'prediction_timestamp', 'created_at', 'analyzed_at') ??
      null,
    overall: {
      probability:
        overallProbability === undefined
          ? fusionBlock.probability ?? null
          : Number(overallProbability),
      riskBand: pick(root, 'risk_band', 'riskBand') ??
        pick(pick(root, 'overall') ?? {}, 'risk_band') ??
        null,
    },
    modalities,
    fusion: {
      method: pick(fusionBlock, 'method', 'model') ?? null,
      probability:
        fusionBlock.probability === undefined
          ? null
          : Number(fusionBlock.probability),
      contribution:
        fusionBlock.contribution === undefined
          ? null
          : Number(fusionBlock.contribution),
    },
    explanations,
  };
}

/* ------------------------------------------------------------------ */
/* Endpoints                                                           */
/* ------------------------------------------------------------------ */

/**
 * POST /predict with the selected modality files.
 * @param {Object<string, File|null>} files keyed by modality
 * @param {{ onUploadProgress?: (event: import('axios').AxiosProgressEvent) => void,
 *           signal?: AbortSignal }} [options]
 * @returns {Promise<ReturnType<typeof normalizePrediction>>}
 */
export async function analyzeFiles(files, options = {}) {
  const formData = new FormData();
  let included = 0;
  for (const key of MODALITY_KEYS) {
    const file = files?.[key];
    if (file) {
      formData.append(key, file, file.name);
      included += 1;
    }
  }
  if (included === 0) {
    throw new ApiError('Select at least one modality file before running the analysis.');
  }

  try {
    const { data } = await client.post('/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      signal: options.signal,
      onUploadProgress: options.onUploadProgress,
    });
    return normalizePrediction(data);
  } catch (error) {
    throw toApiError(error);
  }
}

/** GET /model-info — cached model metadata from the backend. */
export async function getModelInfo() {
  try {
    const { data } = await client.get('/model-info');
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}

/** GET /health — lightweight connectivity probe. */
export async function checkHealth() {
  try {
    const { data } = await client.get('/health', { timeout: 8000 });
    return data;
  } catch (error) {
    throw toApiError(error);
  }
}
