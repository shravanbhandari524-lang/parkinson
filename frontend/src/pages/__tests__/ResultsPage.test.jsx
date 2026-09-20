import { useEffect, useRef } from 'react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import Results from '../Results.jsx';
import { AssessmentProvider, useAssessment } from '../../hooks/useAssessment.jsx';
import { analyzeFiles } from '../../services/api.js';

vi.mock('../../services/api.js', () => ({
  analyzeFiles: vi.fn(),
  ApiError: class ApiError extends Error {
    constructor(message, options = {}) {
      super(message);
      this.name = 'ApiError';
      this.status = options.status ?? null;
      this.code = options.code ?? null;
    }
  },
  getApiBaseUrl: vi.fn(() => 'http://test.local/api'),
  client: { post: vi.fn(), get: vi.fn() },
  getModelInfo: vi.fn(),
  checkHealth: vi.fn(),
  normalizePrediction: vi.fn((x) => x),
  toApiError: vi.fn(),
  DEFAULT_TIMEOUT_MS: 120000,
}));

// Payload exactly as normalizePrediction would return it from the API.
const SAMPLE_RESULT = {
  modelVersion: '1.4.2',
  timestamp: '2026-09-20T10:30:00Z',
  overall: { probability: 0.72, riskBand: 'high' },
  modalities: {
    voice: { available: true, probability: 0.81, contribution: 0.38 },
    handwriting: { available: true, probability: 0.7, contribution: 0.34 },
    gait: { available: false, probability: null, contribution: null },
  },
  fusion: { method: 'TabNet ensemble', probability: 0.72, contribution: 0.28 },
  explanations: {
    voice: [
      { feature: 'MDVP:Jitter(%)', shap: 0.1423, value: 0.015 },
      { feature: 'HNR', shap: -0.0512, value: 24.3 },
    ],
    handwriting: [{ feature: 'RMS', shap: 0.0912, value: 21.4 }],
    gait: [],
    multimodal: [{ feature: 'voice.probability', shap: 0.2134, value: 0.81 }],
  },
};

// restoreMocks resets implementations between tests, so seed per-test.
beforeEach(() => {
  analyzeFiles.mockResolvedValue(SAMPLE_RESULT);
});

/** Calls analyze() once on mount so the provider holds a completed result. */
function Preloaded() {
  const { analyze } = useAssessment();
  const started = useRef(false);
  useEffect(() => {
    if (started.current) return;
    started.current = true;
    analyze({ voice: new File(['x'], 'voice.csv', { type: 'text/csv' }) });
  }, [analyze]);
  return null;
}

function renderResults({ preloaded = true } = {}) {
  return render(
    <MemoryRouter initialEntries={['/results']}>
      <AssessmentProvider>
        {preloaded && <Preloaded />}
        <Routes>
          <Route path="/results" element={<Results />} />
          <Route path="/assessment" element={<div>assessment-page</div>} />
        </Routes>
      </AssessmentProvider>
    </MemoryRouter>,
  );
}

describe('Results page — empty state', () => {
  it('prompts the user to run an assessment when no result exists', async () => {
    renderResults({ preloaded: false });
    expect(await screen.findByText(/no analysis results yet/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /go to assessment/i })).toHaveAttribute(
      'href',
      '/assessment',
    );
  });
});

describe('Results page — rendered prediction', () => {
  it('renders the overall probability gauge with the risk band', async () => {
    renderResults();
    expect(await screen.findByText('Assessment Results')).toBeInTheDocument();

    expect(
      screen.getByRole('img', { name: /overall pd probability: 72 percent/i }),
    ).toBeInTheDocument();
    expect(screen.getByText('High risk')).toBeInTheDocument();
  });

  it('shows model version, timestamp, and fusion method from the API', async () => {
    renderResults();
    await screen.findByText('Assessment Results');

    expect(screen.getByText('1.4.2')).toBeInTheDocument();
    // Locale-independent: the ISO timestamp's year must render
    expect(screen.getByText(/2026/)).toBeInTheDocument();
    expect(screen.getByText('TabNet ensemble')).toBeInTheDocument();
  });

  it('renders all modality probability cards, marking missing ones', async () => {
    renderResults();
    await screen.findByText('Assessment Results');

    expect(screen.getByText('81.0%')).toBeInTheDocument(); // voice
    expect(screen.getByText('70.0%')).toBeInTheDocument(); // handwriting
    expect(screen.getByText('38.0% of fused score')).toBeInTheDocument();
    expect(screen.getByText(/not provided in this run/i)).toBeInTheDocument(); // gait
  });

  it('renders SHAP sections for every explanation stream', async () => {
    renderResults();
    await screen.findByText('Assessment Results');

    expect(screen.getByText('Top voice features')).toBeInTheDocument();
    expect(screen.getByText('Top handwriting features')).toBeInTheDocument();
    expect(screen.getByText('Top gait features')).toBeInTheDocument();
    expect(screen.getByText('Top multimodal features')).toBeInTheDocument();

    // Feature attributions arrive verbatim from the API:
    expect(screen.getByText(/MDVP:Jitter\(%\): SHAP 0\.1423/i)).toBeInTheDocument();
    expect(screen.getByText(/HNR: SHAP -0\.0512/i)).toBeInTheDocument();
    expect(screen.getByText(/RMS: SHAP 0\.0912/i)).toBeInTheDocument();
    expect(screen.getByText(/voice\.probability: SHAP 0\.2134/i)).toBeInTheDocument();

    // The gait stream was not uploaded — no values are invented:
    expect(screen.getAllByText('No SHAP values available').length).toBeGreaterThan(0);
  });

  it('charts carry an accessible summary of API values', async () => {
    renderResults();
    await screen.findByText('Assessment Results');

    expect(
      screen.getByRole('img', {
        name: /modality probability comparison: voice 81\.0%, handwriting 70\.0%, fusion 72\.0%/i,
      }),
    ).toBeInTheDocument();
  });

  it('links back to a fresh assessment', async () => {
    renderResults();
    await screen.findByText('Assessment Results');
    expect(screen.getByRole('button', { name: /new analysis/i })).toBeInTheDocument();
  });
});
