# ParkinsonAI — Multimodal Risk Assessment (Frontend)

Professional research dashboard for the project **"Explainable Multimodal AI
Framework for Early Parkinson's Disease Risk Prediction Using Voice,
Handwriting, and Gait Biomarkers"**.

Built with React 18, Vite, Tailwind CSS, React Router, Axios, Recharts, and
Lucide icons. Tested with Vitest + React Testing Library.

## Pages

| Route          | Purpose                                                                    |
| -------------- | -------------------------------------------------------------------------- |
| `/`            | Landing page — framework overview, biomarker explanations, disclaimer      |
| `/assessment`  | Upload voice / handwriting / gait feature files, run the analysis          |
| `/results`     | Risk gauge, per-modality probabilities, contribution charts, SHAP reports  |
| `/methodology` | Full pipeline: Dataset → … → TabNet fusion → SHAP → Risk Prediction        |
| `/about`       | Objective, datasets, algorithms, research gap, limitations, tech stack     |

## Getting started

```bash
cd frontend
npm install

# configure the backend URL (no host is hard-coded in the app)
cp .env.example .env
# then edit .env and set VITE_API_URL, e.g. http://localhost:3000/api

npm run dev       # start dev server on http://localhost:5173
npm test          # run the Vitest suites once
npm run build     # production build
```

## Environment

`VITE_API_URL` is the **only** source for the backend base URL
(see `src/services/api.js`). Supported endpoints:

- `POST /predict` — multipart/form-data with any of the fields
  `voice`, `handwriting`, `gait` (feature files). Returns the fused
  prediction, per-modality probabilities/contributions, and SHAP explanations.
- `GET /model-info` — model metadata.
- `GET /health` — connectivity probe.

The exact JSON contract (including alternate key spellings the normaliser
tolerates) is documented in `.env.example`.

> All probabilities, risk bands, and SHAP values shown in the UI come from the
> API. The frontend never fabricates or estimates values locally.

## Project structure

```
frontend/
├── public/                 # static assets (favicon)
├── src/
│   ├── components/         # reusable UI: layout, upload cards, charts, primitives
│   ├── pages/              # Home, Assessment, Results, Methodology, About
│   ├── services/           # Axios API client (api.js)
│   ├── hooks/              # useAssessment context (shared analysis state)
│   ├── utils/              # formatting, validation, modality metadata
│   ├── assets/
│   ├── test/               # Vitest setup (jsdom polyfills)
│   ├── App.jsx             # router
│   ├── main.jsx            # entry point
│   └── index.css           # Tailwind layers + global styles
├── package.json
├── vite.config.js          # Vite + Vitest config
└── .env.example
```

## Tests

- `ModalityUploadCard.test.jsx` — file validation (type, size, empty), selection, removal
- `AssessmentPage.test.jsx` — upload cards, analysis modes, progress, success navigation, API error state
- `api.test.js` — base URL parsing, error mapping, response normalisation, FormData posting
- `ResultsPage.test.jsx` — empty state, gauge/risk band, metadata, SHAP rendering

```bash
npm test            # run once
npm run test:watch  # watch mode
```

## Disclaimer

This application is a research and educational prototype and is not intended
to provide a medical diagnosis.
