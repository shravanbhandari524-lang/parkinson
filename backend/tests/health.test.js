'use strict';

process.env.NODE_ENV = 'test';
process.env.ML_SERVICE_URL = 'http://ml.test';

const nock = require('nock');
const request = require('supertest');
const { createApp } = require('../src/app');

describe('GET /api/health', () => {
  afterEach(() => nock.cleanAll());

  it('returns 200 with ml_service details when the ML service is healthy', async () => {
    nock('http://ml.test')
      .get('/health')
      .reply(200, { status: 'ok', model_version: '1.0.0', models_loaded: true });

    const res = await request(createApp()).get('/api/health');

    expect(res.status).toBe(200);
    expect(res.body.status).toBe('ok');
    expect(res.body.service).toBe('backend');
    expect(res.body.ml_service.healthy).toBe(true);
    expect(res.body.ml_service.model_version).toBe('1.0.0');
    expect(res.body.timestamp).toBeDefined();
  });

  it('returns 503 when the ML service reports degraded', async () => {
    nock('http://ml.test')
      .get('/health')
      .reply(503, { status: 'degraded', models_loaded: false, detail: 'artifacts missing' });

    const res = await request(createApp()).get('/api/health');

    expect(res.status).toBe(503);
    expect(res.body.status).toBe('degraded');
    expect(res.body.ml_service.healthy).toBe(false);
  });

  it('returns 503 when the ML service is unreachable', async () => {
    nock('http://ml.test')
      .get('/health')
      .replyWithError({ code: 'ECONNREFUSED', message: 'connect ECONNREFUSED' });

    const res = await request(createApp()).get('/api/health');

    expect(res.status).toBe(503);
    expect(res.body.error.code).toBe('ML_SERVICE_UNAVAILABLE');
  });
});
