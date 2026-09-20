'use strict';

/**
 * Centralized error handling: typed application errors, the async wrapper,
 * the 404 handler and the final Express error middleware.
 */

/**
 * Error with an HTTP status and a machine-readable code.
 * Anything thrown as AppError is safe to expose to clients; anything else
 * is logged and masked as a generic 500 in production.
 */
class AppError extends Error {
  constructor(statusCode, message, code = 'APP_ERROR') {
    super(message);
    this.name = 'AppError';
    this.statusCode = statusCode;
    this.code = code;
    this.isOperational = true;
  }

  /** Attach the underlying cause (shown only outside production). */
  withCause(cause) {
    this.cause = cause;
    return this;
  }
}

/** Wrap async route handlers so rejections reach the error middleware. */
const asyncHandler = (fn) => (req, res, next) =>
  Promise.resolve(fn(req, res, next)).catch(next);

/** Final catch-all for unmatched routes. */
function notFoundHandler(req, res, next) {
  next(new AppError(404, `Route not found: ${req.method} ${req.originalUrl}`, 'NOT_FOUND'));
}

// eslint-disable-next-line no-unused-vars
function errorHandler(err, req, res, next) {
  const isProduction = process.env.NODE_ENV === 'production';

  // Multer-specific upload errors -> meaningful client responses
  if (err && err.code === 'LIMIT_FILE_SIZE') {
    err = new AppError(413, 'Uploaded file exceeds the maximum allowed size', 'FILE_TOO_LARGE');
  } else if (err && err.code === 'LIMIT_UNEXPECTED_FILE') {
    err = new AppError(400, `Unexpected file field: ${err.field}`, 'UNEXPECTED_FILE_FIELD');
  }

  let statusCode = err.statusCode || 500;
  let message = err.message || 'Internal server error';
  let code = err.code || 'INTERNAL_ERROR';

  if (!err.isOperational) {
    // Unexpected bug: log the full stack, never leak internals to clients
    console.error('[error] unhandled error:', err.stack || err);
    if (isProduction) {
      statusCode = 500;
      message = 'Internal server error';
      code = 'INTERNAL_ERROR';
    }
  } else if (statusCode >= 500) {
    console.error('[error]', statusCode, code, err.stack || err.message);
  }

  const body = {
    error: {
      code,
      message,
      status: statusCode,
    },
  };
  if (!isProduction && err.cause) {
    body.error.cause = String(err.cause.message || err.cause);
  }

  res.status(statusCode).json(body);
}

module.exports = {
  AppError,
  asyncHandler,
  notFoundHandler,
  errorHandler,
};
