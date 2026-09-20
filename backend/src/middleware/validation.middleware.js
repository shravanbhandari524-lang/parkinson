'use strict';

/**
 * Generic Zod validation middleware.
 *
 * Usage: `router.post('/', validate(schema, 'body'), handler)` — validates
 * the chosen request property (body/query/params/files) and replaces it with
 * the parsed (defaults applied, unknown keys stripped) value on success.
 */

const { AppError } = require('./error.middleware');

function validate(schema, property = 'body') {
  return (req, res, next) => {
    const result = schema.safeParse(req[property]);
    if (!result.success) {
      const details = result.error.issues
        .map((issue) => `${issue.path.join('.') || property}: ${issue.message}`)
        .join('; ');
      return next(
        new AppError(400, `Validation failed: ${details}`, 'VALIDATION_ERROR'),
      );
    }
    req[property] = result.data;
    return next();
  };
}

module.exports = { validate };
