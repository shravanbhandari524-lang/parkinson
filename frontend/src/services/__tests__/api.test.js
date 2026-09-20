import { beforeEach, describe, expect, it, vi } from 'vitest';

// Mock the axios module BEFORE importing the service under test, so that the
// module-level axios.create() call returns our controllable client.
vi.mock('axios', () => {
  const mockClient = { post: vi.fn(), get: vi.fn() };
  return {
    default: {
      create: vi.fn(() => mockClient),
      isAxiosError: (error) => Boolean(error && error.isAxiosError),
    },
  };
});

import axios from 'axios';
import {
  ApiError,
  analyzeFiles,
  checkHealth,
  getApiBaseUrl,
  getModelInfo,
  normalizePrediction,
  toApiError,
} from '../api.js';

// Calling axios.create() returns the same shared mock client the service uses.
const client = axios.create();

function axiosResponseError(status, data) {
  const error = new Error(`Request failed with status code ${status}`);
  error.isAxiosError = true;
  error.response = { status, data };
  error.config = {};
  return error;
}

function axiosNetworkError(code = 'ECONNREFUSED') {
  const error = new Error('Network error');
  error.isAxiosError = true;
  error.code = code;
  error.config = {};
  return error;
}

beforeEach(() => {
  client.post.mockReset();
  client.get.mockReset();
});

describe('getApiBaseUrl', () => {
  it('trims whitespace and trailing slashes from VITE_API_URL', () => {
    expect(
      getApiBaseUrl({ VITE_API_URL: '  http://api.example.com:3000/api/  ' }),
    ).toBe('http://api.example.com:3000/api');
  });

  it('returns null when the variable is missing or blank', () => {
    expect(getApiBaseUrl({})).toBeNull();
    expect(getApiBaseUrl({ VITE_API_URL: '   ' })).toBeNull();
    expect(getApiBaseUrl()).toBeNull();
  });
});

describe('toApiError', () => {
  it('surfaces the server error message and status', () => {
    const apiError = toApiError(
      axiosResponseError(422, { error: 'Invalid voice file: jitter column missing' }),
    );
    expect(apiError).toBeInstanceOf(ApiError);
    expect(apiError.status).toBe(422);
    expect(apiError.message).toContain('Invalid voice file');
  });

  it('maps network failures to an actionable message', () => {
    const apiError = toApiError(axiosNetworkError());
    expect(apiError.status).toBeNull();
    expect(apiError.message).toMatch(/cannot reach the analysis api/i);
  });

  it('passes through existing ApiError instances unchanged', () => {
    const original = new ApiError('Already mapped');
    expect(toApiError(original)).toBe(original);
  });
});

describe('normalizePrediction', () => {
  it('maps the documented API contract onto the UI shape', () => {
    const normalized = normalizePrediction({
      model_version: '1.4.2',
      timestamp: '2026-09-20T10:30:00Z',
      overall_probability: 0.72,
      risk_band: 'high',
      modalities: {
        voice: { available: true, probability: 0.81, contribution: 0.38 },
        handwriting: { available: true, probability: 0.7, contribution: 0.34 },
        gait: { available: false, probability: null, contribution: null },
      },
      fusion: { method: 'TabNet ensemble', probability: 0.72, contribution: 0.28 },
      explanations: {
        voice: [{ feature: 'MDVP:Jitter(%)', shap: 0.1423, value: 0.015 }],
        gait: [],
        multimodal: [{ feature: 'voice.probability', shap: 0.2134, value: 0.81 }],
      },
    });

    expect(normalized.modelVersion).toBe('1.4.2');
    expect(normalized.timestamp).toBe('2026-09-20T10:30:00Z');
    expect(normalized.overall.probability).toBeCloseTo(0.72);
    expect(normalized.overall.riskBand).toBe('high');
    expect(normalized.modalities.voice.probability).toBeCloseTo(0.81);
    expect(normalized.modalities.gait.available).toBe(false);
    expect(normalized.modalities.gait.probability).toBeNull();
    expect(normalized.fusion.method).toBe('TabNet ensemble');
    expect(normalized.explanations.voice[0]).toEqual({
      feature: 'MDVP:Jitter(%)',
      shap: 0.1423,
      value: 0.015,
    });
    expect(normalized.explanations.handwriting).toEqual([]);
    expect(normalized.explanations.multimodal[0].feature).toBe('voice.probability');
  });

  it('tolerates alternate key spellings without fabricating values', () => {
    const normalized = normalizePrediction({
      prediction_timestamp: '2026-09-20T11:00:00Z',
      overall_probability: 0.4,
      modalities: {
        voice: { p_pd: 0.55, weight: 0.5 },
      },
      explanations: {
        voice: [{ feature_name: 'HNR', shap_value: 0.2, feature_value: 20 }],
      },
    });

    expect(normalized.timestamp).toBe('2026-09-20T11:00:00Z');
    expect(normalized.modalities.voice.probability).toBeCloseTo(0.55);
    expect(normalized.modalities.voice.contribution).toBeCloseTo(0.5);
    expect(normalized.modalities.handwriting.probability).toBeNull();
    expect(normalized.explanations.voice[0]).toEqual({
      feature: 'HNR',
      shap: 0.2,
      value: 20,
    });
    // Nothing was fabricated for missing streams:
    expect(normalized.overall.riskBand).toBeNull();
    expect(normalized.fusion.method).toBeNull();
  });
});

describe('analyzeFiles', () => {
  it('rejects before any request when no file is selected', async () => {
    await expect(analyzeFiles({ voice: null, handwriting: null, gait: null })).rejects.toThrow(
      /select at least one modality file/i,
    );
    expect(client.post).not.toHaveBeenCalled();
  });

  it('posts the selected files as multipart form data to /predict', async () => {
    const voice = new File(['acoustic,features\n'], 'voice.csv', { type: 'text/csv' });
    const gait = new File(['gait,features\n'], 'gait.txt', { type: 'text/plain' });
    client.post.mockResolvedValueOnce({
      data: { model_version: '1.0.0', overall_probability: 0.5, risk_band: 'borderline' },
    });

    const result = await analyzeFiles({ voice, gait });

    expect(client.post).toHaveBeenCalledTimes(1);
    const [url, formData, config] = client.post.mock.calls[0];
    expect(url).toBe('/predict');
    expect(formData).toBeInstanceOf(FormData);
    // jsdom (and the spec) wrap blobs with a filename in a new File — compare metadata:
    expect(formData.get('voice').name).toBe('voice.csv');
    expect(formData.get('gait').name).toBe('gait.txt');
    expect(formData.get('handwriting')).toBeNull();
    expect(config.headers['Content-Type']).toBe('multipart/form-data');
    // Normalized output only:
    expect(result.modelVersion).toBe('1.0.0');
    expect(result.overall.probability).toBeCloseTo(0.5);
  });

  it('propagates API errors as ApiError for the UI to render', async () => {
    client.post.mockRejectedValueOnce(
      axiosResponseError(500, { message: 'Fusion model failed to load' }),
    );
    const voice = new File(['x'], 'voice.csv', { type: 'text/csv' });

    await expect(analyzeFiles({ voice })).rejects.toMatchObject({
      name: 'ApiError',
      status: 500,
      message: 'Fusion model failed to load',
    });
  });

  it('forwards the upload progress callback to axios', async () => {
    client.post.mockResolvedValueOnce({ data: {} });
    const onUploadProgress = vi.fn();
    const voice = new File(['x'], 'voice.csv', { type: 'text/csv' });

    await analyzeFiles({ voice }, { onUploadProgress });

    expect(client.post.mock.calls[0][2].onUploadProgress).toBe(onUploadProgress);
  });
});

describe('health and model info endpoints', () => {
  it('calls GET /health', async () => {
    client.get.mockResolvedValueOnce({ data: { status: 'ok' } });
    await expect(checkHealth()).resolves.toEqual({ status: 'ok' });
    expect(client.get.mock.calls[0][0]).toBe('/health');
  });

  it('calls GET /model-info', async () => {
    client.get.mockResolvedValueOnce({ data: { model_version: '1.0.0' } });
    await expect(getModelInfo()).resolves.toEqual({ model_version: '1.0.0' });
    expect(client.get.mock.calls[0][0]).toBe('/model-info');
  });

  it('maps a failing health check to an ApiError', async () => {
    client.get.mockRejectedValueOnce(axiosNetworkError());
    await expect(checkHealth()).rejects.toBeInstanceOf(ApiError);
  });
});
