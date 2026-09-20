'use strict';

process.env.NODE_ENV = 'test';
process.env.ML_SERVICE_URL = 'http://ml.test';

const nock = require('nock');
const request = require('supertest');
const { createApp } = require('../src/app');

describe('GET /api/model-info', () => {
  afterEach(() => nock.cleanAll());

  it('returns 200 with model metadata from the ML service', async () => {
    nock('http://ml.test').get('/model-info').reply(200, {
      model_version: '1.0.0',
      modalities: {
        voice: { primary_model: 'XGBoost', metrics: { roc_auc: 0.959 } },
        handwriting: { primary_model: 'RandomForest', metrics: { roc_auc: 0.754 } },
        gait: { primary_model: 'XGBoost', metrics: { roc_auc: 0.793 } },
      },
      fusion: { modalities: ['voice', 'handwriting', 'gait'] },
    });

    const res = await request(createApp()).get('/api/model-info');

    expect(res.status).toBe(200);
    expect(res.body.model_version).toBe('1.0.0');
    expect(res.body.modalities.voice.primary_model).toBe('XGBoost');
    expect(res.body.fusion.modalities).toHaveLength(3);
  });

  it('returns 502 when the ML service fails internally', async () => {
    nock('http://ml.test').get('/model-info').reply(500, { detail: 'artifacts missing' });

    const res = await request(createApp()).get('/api/model-info');

    expect(res.status).toBe(502);
    expect(res.body.error.code).toBe('ML_SERVICE_ERROR');
  });

  it('returns 503 when the ML service is unreachable', async () => {
    nock('http://ml.test')
      .get('/model-info')
      .replyWithError({ code: 'ECONNREFUSED', message: 'connect ECONNREFUSED' });

    const res = await request(createApp()).get('/api/model-info');

    expect(res.status).toBe(503);
    expect(res.body.error.code).toBe('ML_SERVICE_UNAVAILABLE');
  });
});
