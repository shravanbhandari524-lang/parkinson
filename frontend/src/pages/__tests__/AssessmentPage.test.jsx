import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import Assessment from '../Assessment.jsx';
import { AssessmentProvider } from '../../hooks/useAssessment.jsx';
import { ApiError, analyzeFiles } from '../../services/api.js';

// The hook under the page calls these; the axios layer is mocked at the
// module boundary so no network is touched.
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
    voice: [{ feature: 'MDVP:Jitter(%)', shap: 0.1423, value: 0.015 }],
    handwriting: [],
    gait: [],
    multimodal: [],
  },
};

function renderAssessment() {
  return render(
    <MemoryRouter initialEntries={['/assessment']}>
      <AssessmentProvider>
        <Routes>
          <Route path="/assessment" element={<Assessment />} />
          <Route path="/results" element={<div>results-page-reached</div>} />
        </Routes>
      </AssessmentProvider>
    </MemoryRouter>,
  );
}

function uploadVoiceFile() {
  const input = document.querySelector('input[type="file"][accept=".csv,.txt"]');
  fireEvent.change(input, {
    target: {
      files: [new File(['acoustic,features\n'], 'voice.csv', { type: 'text/csv' })],
    },
  });
}

describe('Assessment page', () => {
  it('renders the three modality upload cards', () => {
    renderAssessment();
    expect(screen.getByRole('heading', { name: 'Voice' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Handwriting' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Gait' })).toBeInTheDocument();
  });

  it('disables Run Analysis until at least one file is selected', () => {
    renderAssessment();
    expect(screen.getByTestId('analyze-button')).toBeDisabled();
    expect(screen.getByText(/no files selected yet/i)).toBeInTheDocument();
  });

  it('shows the single-modality mode and enables analysis after selecting a voice file', () => {
    renderAssessment();
    uploadVoiceFile();

    expect(screen.getByTestId('voice-file-selected')).toBeInTheDocument();
    expect(screen.getByText(/voice-only analysis/i)).toBeInTheDocument();
    expect(screen.getByTestId('analyze-button')).toBeEnabled();
  });

  it('shows the multimodal mode when all three files are provided', () => {
    renderAssessment();
    const inputs = document.querySelectorAll('input[type="file"][accept=".csv,.txt"]');
    expect(inputs).toHaveLength(3);
    const names = ['voice.csv', 'handwriting.csv', 'gait.csv'];
    inputs.forEach((input, index) => {
      fireEvent.change(input, {
        target: { files: [new File(['data'], names[index], { type: 'text/csv' })] },
      });
    });

    expect(screen.getByText(/multimodal fusion/i)).toBeInTheDocument();
    expect(screen.getByText(/3 modality files selected/i)).toBeInTheDocument();
  });

  it('runs the analysis and navigates to results on success', async () => {
    analyzeFiles.mockResolvedValueOnce(SAMPLE_RESULT);
    const user = userEvent.setup();
    renderAssessment();

    uploadVoiceFile();
    await user.click(screen.getByTestId('analyze-button'));

    await waitFor(() => {
      expect(screen.getByText('results-page-reached')).toBeInTheDocument();
    });
    expect(analyzeFiles).toHaveBeenCalledTimes(1);
    const passedFiles = analyzeFiles.mock.calls[0][0];
    expect(passedFiles.voice).toBeInstanceOf(File);
    expect(passedFiles.voice.name).toBe('voice.csv');
    expect(passedFiles.handwriting).toBeNull();
  });

  it('renders the API error state and stays on the page on failure', async () => {
    analyzeFiles.mockRejectedValueOnce(
      new ApiError('Fusion model failed to load', { status: 500 }),
    );
    const user = userEvent.setup();
    renderAssessment();

    uploadVoiceFile();
    await user.click(screen.getByTestId('analyze-button'));

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/analysis failed/i);
    expect(alert).toHaveTextContent(/fusion model failed to load/i);
    expect(screen.queryByText('results-page-reached')).not.toBeInTheDocument();
    expect(screen.getByTestId('analyze-button')).toBeEnabled();
  });

  it('keeps cards disabled while the analysis is running', async () => {
    let resolveAnalysis;
    analyzeFiles.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveAnalysis = resolve;
      }),
    );
    const user = userEvent.setup();
    renderAssessment();

    uploadVoiceFile();
    await user.click(screen.getByTestId('analyze-button'));

    expect(screen.getByTestId('analyze-button')).toBeDisabled();
    expect(screen.getByText(/analyzing/i)).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toBeInTheDocument();

    resolveAnalysis(SAMPLE_RESULT);
    await waitFor(() => {
      expect(screen.getByText('results-page-reached')).toBeInTheDocument();
    });
  });
});
