# Backend — Parkinson's Multimodal AI (Node.js / Express.js)

Production-quality REST API for the multimodal Parkinson's risk framework.
Express.js is the **public** API; the Python ML system runs as a separate
**internal** ML service that only Express talks to.

```
React/Vite (5173)  →  Express.js (4000)  →  Python ML service (8000)  →  Trained models
```

The Express layer never computes predictions itself — every probability comes
from the trained artifacts via the internal ML service.

## API

| Method | Route                     | Body                                                      |
|--------|---------------------------|-----------------------------------------------------------|
| GET    | `/api/health`             | — (200 only when the ML service is healthy)                |
| GET    | `/api/model-info`         | — (training metadata + CV metrics of the deployed models)  |
| POST   | `/api/predict/voice`      | multipart `file` — voice CSV (UCI Parkinson's columns)     |
| POST   | `/api/predict/handwriting`| multipart `file` — HandPD CSV (spiral or meander)          |
| POST   | `/api/predict/gait`       | multipart `file` — PhysioNet VGRF record (tab-separated)   |
| POST   | `/api/predict/multimodal` | multipart `voice` / `handwriting` / `gait` — any subset    |

### Response contract

```json
{
  "voice":       { "probability": 0.9878, "top_features": [ { "modality": "voice", "feature": "spread1", "contribution": 0.84 } ] },
  "handwriting": { "probability": 0.1709, "top_features": [] },
  "gait":        { "probability": 0.9289, "top_features": [] },
  "multimodal":  { "probability": 0.6842, "top_features": [] },
  "risk_band": "low | borderline | high",
  "model_version": "1.0.0"
}
```

Only the modalities that were actually supplied appear in the response; the
fusion model renormalizes for missing modalities (the ML meta-model was
trained with explicit missing-modality indicators). Risk bands:
`< 0.35 low`, `0.35–0.65 borderline`, `> 0.65 high`.

### Errors

All errors use one envelope:

```json
{ "error": { "code": "ML_SERVICE_UNAVAILABLE", "message": "...", "status": 503 } }
```

| Code | Meaning |
|---|---|
| `VALIDATION_ERROR` | 400 — missing file / no modality supplied |
| `UNSUPPORTED_MEDIA_TYPE` | 415 — wrong extension or MIME type |
| `ML_SERVICE_ERROR` | 422/502 — ML-side validation or upstream failure |
| `ML_SERVICE_UNAVAILABLE` | 503 — ML service unreachable |
| `ML_SERVICE_TIMEOUT` | 504 — ML service timed out |
| `FILE_TOO_LARGE` | 413 — upload exceeds `MAX_FILE_SIZE` |

## Setup

```bash
cd backend
npm install
cp .env.example .env          # adjust ML_SERVICE_URL if needed
npm start                     # or: npm run dev
```

The internal ML service (separate terminal):

```bash
# one-time: environment + training (artifacts are git-ignored)
python3 -m venv ml/.venv
ml/.venv/bin/pip install -r ml/requirements.txt
ml/.venv/bin/python ml/train_all.py

# serve the models
ml/.venv/bin/python -m ml.service       # http://127.0.0.1:8000
```

Environment variables (see `.env.example`): `PORT`, `NODE_ENV`,
`ML_SERVICE_URL`, `ML_TIMEOUT_MS`, `MAX_FILE_SIZE`, `CORS_ORIGIN`.

## Security

- **Helmet** security headers; `x-powered-by` disabled.
- **CORS** restricted to `CORS_ORIGIN` (default: the Vite dev server).
- **Multer** with `memoryStorage` — uploads are held in RAM, forwarded to the
  ML service, and **never written to disk**; nothing is persisted and no
  patient identifiers are stored anywhere.
- File-type validation (extension + MIME) per modality, size limit from
  `MAX_FILE_SIZE`, max 3 files per request.
- **Zod** validation of env config at boot and of the ML response contract.
- Centralized error handling — unexpected errors are logged with stack traces
  but masked as generic 500s in production.

## Tests

```bash
cd backend && npm test
```

21 Supertest + nock tests cover: health (healthy / degraded / unreachable),
model-info, prediction endpoints, invalid and missing uploads, oversized
uploads, multimodal partial-modality behaviour, ML timeout and ML-side 422
propagation, and the 404 envelope.

## Project layout

```
backend/src
├── config/config.js               Zod-validated env config
├── controllers/                   prediction + model handlers
├── routes/                        health, model, prediction routers
├── services/
│   ├── ml.service.js              the only place that talks to Python
│   ├── upload.service.js          multer wiring + per-modality rules
│   └── validation.service.js      Zod schemas + file-type rules
├── middleware/                    error, upload, validation
├── app.js                         security/CORS/logging wiring
└── server.js                      graceful shutdown
```

The Python side of the internal service lives in `ml/service/`
(FastAPI + uvicorn, internal only — never exposed publicly).
