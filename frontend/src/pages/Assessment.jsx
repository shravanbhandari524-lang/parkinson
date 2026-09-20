import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CirclePlay, LoaderCircle, Trash2 } from 'lucide-react';
import ModalityUploadCard from '../components/ModalityUploadCard.jsx';
import Disclaimer from '../components/Disclaimer.jsx';
import Button from '../components/ui/Button.jsx';
import { Alert, EmptyState, ProgressBar } from '../components/ui/index.js';
import { useAssessment } from '../hooks/useAssessment.jsx';
import { MODALITY_KEYS, MODALITY_META } from '../utils/modalityMeta.js';
import { getAnalysisMode } from '../utils/validation.js';

export default function Assessment() {
  const navigate = useNavigate();
  const { status, error, progress, analyze, reset } = useAssessment();
  const [files, setFiles] = useState({ voice: null, handwriting: null, gait: null });

  const { mode, count } = getAnalysisMode(files);
  const analyzing = status === 'analyzing';
  const hasFiles = count > 0;

  const handleFileChange = (modality, file) => {
    setFiles((prev) => ({ ...prev, [modality]: file }));
  };

  const handleAnalyze = async () => {
    const ok = await analyze(files);
    if (ok) navigate('/results');
    // On error the Alert below renders; the user stays on the page.
  };

  const handleClearAll = () => {
    setFiles({ voice: null, handwriting: null, gait: null });
    reset();
  };

  return (
    <div className="space-y-8 animate-fade-in">
      <header>
        <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
          Multimodal Assessment
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
          Upload one feature file per biomarker stream. Any single modality works —
          the fusion renormalises around what you provide.
        </p>
      </header>

      {/* Upload cards */}
      <div className="grid gap-4 lg:grid-cols-3">
        {MODALITY_KEYS.map((key) => (
          <ModalityUploadCard
            key={key}
            modality={key}
            file={files[key]}
            onFileChange={handleFileChange}
            disabled={analyzing}
          />
        ))}
      </div>

      {/* Mode summary + actions */}
      <section
        aria-label="Analysis controls"
        className="rounded-2xl border border-slate-800 bg-surface-raised p-5 sm:p-6"
      >
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
              Analysis mode
            </h2>
            {hasFiles ? (
              <p className="mt-1.5 flex flex-wrap items-center gap-2">
                <span className="text-sm font-medium text-teal-300">{mode}</span>
                <span className="rounded-md bg-slate-800 px-2 py-0.5 text-xs text-slate-400">
                  {count} modality file{count > 1 ? 's' : ''} selected
                </span>
              </p>
            ) : (
              <p className="mt-1.5 text-sm text-slate-500">
                No files selected yet — add at least one modality to enable analysis.
              </p>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="ghost"
              onClick={handleClearAll}
              disabled={!hasFiles || analyzing}
            >
              <Trash2 aria-hidden="true" className="h-4 w-4" />
              Clear all
            </Button>
            <Button
              size="lg"
              onClick={handleAnalyze}
              disabled={!hasFiles}
              loading={analyzing}
              data-testid="analyze-button"
            >
              {analyzing ? (
                <>
                  <LoaderCircle aria-hidden="true" className="h-4 w-4 animate-spin" />
                  Analyzing…
                </>
              ) : (
                <>
                  <CirclePlay aria-hidden="true" className="h-5 w-5" />
                  Run Analysis
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Upload progress */}
        {analyzing && (
          <div className="mt-5 animate-fade-in">
            <ProgressBar
              value={progress}
              label="Uploading feature files to the analysis API"
            />
            <p className="mt-2 flex items-center gap-2 text-xs text-slate-500">
              <LoaderCircle aria-hidden="true" className="h-3.5 w-3.5 animate-spin" />
              Inference and SHAP explanation may take a few seconds after upload
              completes.
            </p>
          </div>
        )}
      </section>

      {/* Error state */}
      {status === 'error' && error && (
        <Alert variant="error" title="Analysis failed">
          <p>{error.message}</p>
          <p className="mt-1 text-xs opacity-80">
            Check that the Express backend is reachable, then try again.
          </p>
        </Alert>
      )}

      {/* Guidance */}
      {!hasFiles && status !== 'analyzing' && (
        <EmptyState
          icon={CirclePlay}
          title="Nothing to analyze yet"
          description="Select a voice, handwriting or gait feature file above — or combine all three for the full multimodal assessment."
        />
      )}

      <Disclaimer />
    </div>
  );
}
