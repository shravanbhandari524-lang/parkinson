'use strict';

process.env.NODE_ENV = 'test';
process.env.ML_SERVICE_URL = 'http://ml.test';

const nock = require('nock');
const request = require('supertest');
const { createApp } = require('../src/app');
const { mlPredictionResponse, bufferFor } = require('./helpers');

const ML = 'http://ml.test';

describe('POST /api/predict/voice', () => {
  afterEach(() => nock.cleanAll());

  it('returns 200 with the normalized prediction contract', async () => {
    nock(ML).post('/predict/voice').reply(200, mlPredictionResponse(['voice']));

    const res = await request(createApp())
      .post('/api/predict/voice')
      .attach('file', bufferFor('voice'), { filename: 'voice.csv', contentType: 'text/csv' });

    expect(res.status).toBe(200);
    expect(res.body.voice.probability).toBeCloseTo(0.68);
    expect(res.body.voice.top_features[0].feature).toBe('voice_feature_1');
    expect(res.body.multimodal.probability).toBeCloseTo(0.72);
    expect(res.body.risk_band).toBe('high');
    expect(res.body.model_version).toBe('1.0.0');
  });

  it('returns 400 when no file is uploaded', async () => {
    const res = await request(createApp()).post('/api/predict/voice');

    expect(res.status).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  it('returns 415 when the file type is not allowed', async () => {
    const res = await request(createApp())
      .post('/api/predict/voice')
      .attach('file', Buffer.from('not a csv'), { filename: 'voice.txt', contentType: 'text/plain' });

    expect(res.status).toBe(415);
    expect(res.body.error.code).toBe('UNSUPPORTED_MEDIA_TYPE');
  });

  it('returns 503 when the ML service is unreachable', async () => {
    nock(ML)
      .post('/predict/voice')
      .replyWithError({ code: 'ECONNREFUSED', message: 'connect ECONNREFUSED' });

    const res = await request(createApp())
      .post('/api/predict/voice')
      .attach('file', bufferFor('voice'), { filename: 'voice.csv', contentType: 'text/csv' });

    expect(res.status).toBe(503);
    expect(res.body.error.code).toBe('ML_SERVICE_UNAVAILABLE');
  });

  it('returns 504 when the ML service times out', async () => {
    nock(ML)
      .post('/predict/voice')
      .replyWithError({ code: 'ECONNABORTED', message: 'timeout of 30000ms exceeded' });

    const res = await request(createApp())
      .post('/api/predict/voice')
      .attach('file', bufferFor('voice'), { filename: 'voice.csv', contentType: 'text/csv' });

    expect(res.status).toBe(504);
    expect(res.body.error.code).toBe('ML_SERVICE_TIMEOUT');
  });

  it('propagates ML-side validation errors (422)', async () => {
    nock(ML)
      .post('/predict/voice')
      .reply(422, { detail: 'voice CSV is missing required columns' });

    const res = await request(createApp())
      .post('/api/predict/voice')
      .attach('file', bufferFor('voice'), { filename: 'voice.csv', contentType: 'text/csv' });

    expect(res.status).toBe(422);
    expect(res.body.error.message).toMatch(/missing required columns/);
  });
});

describe('POST /api/predict/handwriting and /gait', () => {
  afterEach(() => nock.cleanAll());

  it('predicts handwriting from a CSV upload', async () => {
    nock(ML).post('/predict/handwriting').reply(200, mlPredictionResponse(['handwriting']));

    const res = await request(createApp())
      .post('/api/predict/handwriting')
      .attach('file', bufferFor('handwriting'), { filename: 'spiral.csv', contentType: 'text/csv' });

    expect(res.status).toBe(200);
    expect(res.body.handwriting).toBeDefined();
    expect(res.body.voice).toBeUndefined();
    expect(res.body.gait).toBeUndefined();
  });

  it('predicts gait from a TXT upload', async () => {
    nock(ML).post('/predict/gait').reply(200, mlPredictionResponse(['gait']));

    const res = await request(createApp())
      .post('/api/predict/gait')
      .attach('file', bufferFor('gait'), { filename: 'GaPt03_01.txt', contentType: 'text/plain' });

    expect(res.status).toBe(200);
    expect(res.body.gait).toBeDefined();
    expect(res.body.voice).toBeUndefined();
  });

  it('returns 400 for gait when no file is uploaded', async () => {
    const res = await request(createApp()).post('/api/predict/gait');

    expect(res.status).toBe(400);
  });
});

describe('POST /api/predict/multimodal', () => {
  afterEach(() => nock.cleanAll());

  it('returns only the supplied modalities plus the fused result', async () => {
    nock(ML).post('/predict/multimodal').reply(200, mlPredictionResponse(['voice', 'gait']));

    const res = await request(createApp())
      .post('/api/predict/multimodal')
      .attach('voice', bufferFor('voice'), { filename: 'voice.csv', contentType: 'text/csv' })
      .attach('gait', bufferFor('gait'), { filename: 'gait.txt', contentType: 'text/plain' });

    expect(res.status).toBe(200);
    expect(res.body.voice).toBeDefined();
    expect(res.body.gait).toBeDefined();
    expect(res.body.handwriting).toBeUndefined();
    expect(res.body.multimodal.probability).toBeCloseTo(0.72);
    expect(res.body.risk_band).toBe('high');
    expect(res.body.model_version).toBe('1.0.0');
  });

  it('returns 400 when no modality file is supplied', async () => {
    const res = await request(createApp()).post('/api/predict/multimodal');

    expect(res.status).toBe(400);
    expect(res.body.error.message).toMatch(/at least one modality/i);
  });

  it('returns 415 when a modality file has the wrong type', async () => {
    const res = await request(createApp())
      .post('/api/predict/multimodal')
      .attach('handwriting', Buffer.from('binary-ish'), {
        filename: 'drawing.png',
        contentType: 'image/png',
      });

    expect(res.status).toBe(415);
  });

  it('returns 503 when the ML service is unavailable', async () => {
    nock(ML)
      .post('/predict/multimodal')
      .replyWithError({ code: 'ECONNREFUSED', message: 'connect ECONNREFUSED' });

    const res = await request(createApp())
      .post('/api/predict/multimodal')
      .attach('voice', bufferFor('voice'), { filename: 'voice.csv', contentType: 'text/csv' });

    expect(res.status).toBe(503);
    expect(res.body.error.code).toBe('ML_SERVICE_UNAVAILABLE');
  });
});

describe('upload limits', () => {
  afterEach(() => nock.cleanAll());

  it('returns 413 when the file exceeds MAX_FILE_SIZE', async () => {
    process.env.MAX_FILE_SIZE = '64';
    jest.resetModules();
    // re-require with the tiny limit so the test does not build a 5 MB payload
    const { createApp: freshApp } = require('../src/app');
    const big = Buffer.alloc(1024, 'a');

    const res = await request(freshApp())
      .post('/api/predict/voice')
      .attach('file', big, { filename: 'voice.csv', contentType: 'text/csv' });

    expect(res.status).toBe(413);
    expect(res.body.error.code).toBe('FILE_TOO_LARGE');
  });
});

describe('unknown routes', () => {
  it('returns 404 with the error envelope', async () => {
    const res = await request(createApp()).get('/api/does-not-exist');

    expect(res.status).toBe(404);
    expect(res.body.error.code).toBe('NOT_FOUND');
  });
});
