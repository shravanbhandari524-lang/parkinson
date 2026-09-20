'use strict';

/**
 * Centralized configuration with Zod validation.
 *
 * All values come from environment variables (.env in development) and are
 * validated once at boot — the app fails fast on misconfiguration instead of
 * misbehaving at request time.
 */

const fs = require('fs');
const path = require('path');
const { z } = require('zod');

const envSchema = z.object({
  NODE_ENV: z
    .enum(['development', 'test', 'production'])
    .default('development'),
  PORT: z.coerce.number().int().positive().default(4000),
  ML_SERVICE_URL: z
    .string()
    .url('ML_SERVICE_URL must be a valid URL')
    .default('http://127.0.0.1:8000'),
  ML_TIMEOUT_MS: z.coerce.number().int().positive().default(30000),
  MAX_FILE_SIZE: z.coerce
    .number()
    .int()
    .positive()
    .default(5 * 1024 * 1024),
  CORS_ORIGIN: z
    .string()
    .default('http://localhost:5173')
    .transform((value) => value.split(',').map((origin) => origin.trim()).filter(Boolean)),
});

let cached = null;

/**
 * Load and validate configuration. Reads `backend/.env` when present.
 * Cached after the first call (tests can clear it via `clearConfig`).
 */
function getConfig() {
  if (cached) return cached;

  const dotenv = require('dotenv');
  const envPath = path.resolve(__dirname, '..', '..', '.env');
  if (fs.existsSync(envPath)) {
    dotenv.config({ path: envPath });
  } else {
    dotenv.config();
  }

  const parsed = envSchema.safeParse(process.env);
  if (!parsed.success) {
    const details = parsed.error.issues
      .map((issue) => `  - ${issue.path.join('.')}: ${issue.message}`)
      .join('\n');
    throw new Error(`Invalid environment configuration:\n${details}`);
  }

  // expose camelCase accessors (config.port, config.mlServiceUrl, ...)
  const env = parsed.data;
  cached = Object.freeze({
    nodeEnv: env.NODE_ENV,
    port: env.PORT,
    mlServiceUrl: env.ML_SERVICE_URL,
    mlTimeoutMs: env.ML_TIMEOUT_MS,
    maxFileSize: env.MAX_FILE_SIZE,
    corsOrigin: env.CORS_ORIGIN,
  });
  return cached;
}

/** Reset the cached config (used by tests after mutating process.env). */
function clearConfig() {
  cached = null;
}

module.exports = { getConfig, clearConfig };
