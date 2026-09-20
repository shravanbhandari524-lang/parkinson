import { Link } from 'react-router-dom';
import {
  ArrowRight,
  CalendarClock,
  GitBranch,
  Info,
  Microscope,
  RefreshCw,
} from 'lucide-react';
import {
  Card,
  CardBody,
  CardHeader,
} from '../components/ui/Card.jsx';
import { Alert, Button, EmptyState } from '../components/ui/index.js';
import {
  ContributionPie,
  ModalityBarChart,
  RiskGauge,
  ShapChart,
} from '../components/charts/index.js';
import { useAssessment } from '../hooks/useAssessment.jsx';
import { FUSION_COLOR, MODALITY_KEYS, MODALITY_META } from '../utils/modalityMeta.js';
import {
  formatPercent,
  formatTimestamp,
  riskBandInfo,
} from '../utils/format.js';

const EXPLANATION_LABELS = {
  voice: 'Top voice features',
  handwriting: 'Top handwriting features',
  gait: 'Top gait features',
  multimodal: 'Top multimodal features',
};

function ProbabilityCard({ modality, probability, contribution, color, available = true }) {
  const meta = MODALITY_META[modality];
  const Icon = meta.icon;
  return (
    <Card
      className={!available ? 'opacity-70' : ''}
      aria-label={`${meta.label} probability ${available ? formatPercent(probability) : 'not provided'}`}
    >
      <div className="p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <span className={`flex h-9 w-9 items-center justify-center rounded-lg border ${meta.tint}`}>
              <Icon aria-hidden="true" className="h-4 w-4" />
            </span>
            <div>
              <p className="text-sm font-semibold text-slate-200">{meta.label}</p>
              <p className="text-xs text-slate-500">
                {available
                  ? contribution != null
                    ? `${(contribution * 100).toFixed(1)}% of fused score`
                    : 'No contribution reported'
                  : 'Not provided in this run'}
              </p>
            </div>
          </div>
          <p
            className={`font-mono text-xl font-bold tabular-nums ${
              available ? 'text-white' : 'text-slate-600'
            }`}
          >
            {available ? formatPercent(probability) : '—'}
          </p>
        </div>
        {/* Probability meter */}
        <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full rounded-full transition-[width] duration-500 ease-out"
            style={{
              width: `${available ? Math.round((probability ?? 0) * 100) : 0}%`,
              backgroundColor: available ? color : '#334155',
            }}
          />
        </div>
      </div>
    </Card>
  );
}

export default function Results() {
  const { status, result, error, analyzedFileNames, reset } = useAssessment();

  /* ----------------------------- Empty state ----------------------------- */
  if (!result && status !== 'analyzing') {
    return (
      <div className="animate-fade-in">
        {error && status === 'error' ? (
          <Alert variant="error" title="Analysis failed">
            <p>{error.message}</p>
          </Alert>
        ) : (
          <EmptyState
            icon={Microscope}
            title="No analysis results yet"
            description="Run a multimodal assessment first — the results, SHAP explanations and risk band will appear here."
            action={
              <Link to="/assessment">
                <Button>
                  Go to Assessment
                  <ArrowRight aria-hidden="true" className="h-4 w-4" />
                </Button>
              </Link>
            }
          />
        )}
      </div>
    );
  }

  /* --------------------------- Loading state ----------------------------- */
  if (status === 'analyzing') {
    return (
      <div className="flex min-h-[50vh] flex-col items-center justify-center gap-4 animate-fade-in">
        <span
          aria-hidden="true"
          className="h-12 w-12 animate-spin rounded-full border-2 border-slate-700 border-t-teal-400"
        />
        <p className="text-sm font-medium text-slate-300">
          Running multimodal inference…
        </p>
        <p className="max-w-sm text-center text-xs leading-relaxed text-slate-500">
          Scoring each biomarker stream, fusing probabilities, and computing SHAP
          explanations.
        </p>
      </div>
    );
  }

  const { overall, modalities, fusion, explanations } = result;
  const band = riskBandInfo(overall.riskBand, overall.probability);
  const availableModalities = MODALITY_KEYS.filter(
    (key) => modalities[key]?.available && modalities[key]?.probability != null,
  );

  const probabilityEntries = [
    ...availableModalities.map((key) => ({
      key,
      label: MODALITY_META[key].label,
      value: modalities[key].probability,
      color: MODALITY_META[key].color,
    })),
    ...(fusion.probability != null
      ? [{ key: 'fusion', label: 'Fusion', value: fusion.probability, color: FUSION_COLOR }]
      : []),
  ];

  const contributionEntries = availableModalities
    .filter((key) => modalities[key].contribution != null)
    .map((key) => ({
      label: MODALITY_META[key].label,
      value: modalities[key].contribution,
      color: MODALITY_META[key].color,
    }));

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header + risk band */}
      <header className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
            Assessment Results
          </h1>
          <p className="mt-2 text-sm text-slate-400">
            {analyzedFileNames && Object.keys(analyzedFileNames).length > 0
              ? `Based on: ${Object.values(analyzedFileNames).join(' · ')}`
              : 'Multimodal inference output from the analysis API.'}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-3">
          <span
            className={`rounded-lg border px-3 py-1.5 text-sm font-semibold ${band.bg} ${band.border} ${band.text}`}
          >
            {band.label}
          </span>
          <Button variant="secondary" size="sm" onClick={reset}>
            <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />
            New analysis
          </Button>
        </div>
      </header>

      {/* Error banner when a stale result is shown */}
      {status === 'error' && error && (
        <Alert variant="error" title="Previous analysis failed">
          <p>{error.message}</p>
        </Alert>
      )}

      {/* Gauge + metadata */}
      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="p-6 lg:col-span-1">
          <RiskGauge
            probability={overall.probability}
            color={band.hex}
            label="Overall PD probability"
          />
          <p className={`mt-3 text-center text-sm ${band.text}`}>{band.summary}</p>
        </Card>

        <Card className="p-6 lg:col-span-2">
          <CardHeader
            icon={Info}
            title="Prediction metadata"
            subtitle="Supplied by the analysis API for this run."
          />
          <CardBody>
            <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3">
                <dt className="text-xs uppercase tracking-wide text-slate-500">
                  Model version
                </dt>
                <dd className="mt-1 font-mono text-sm text-slate-200">
                  {result.modelVersion ?? '—'}
                </dd>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3">
                <dt className="text-xs uppercase tracking-wide text-slate-500">
                  Prediction timestamp
                </dt>
                <dd className="mt-1 font-mono text-sm text-slate-200">
                  {formatTimestamp(result.timestamp)}
                </dd>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3">
                <dt className="text-xs uppercase tracking-wide text-slate-500">
                  Fusion method
                </dt>
                <dd className="mt-1 font-mono text-sm text-slate-200">
                  {fusion.method ?? '—'}
                </dd>
                <dd className="mt-1 flex items-center gap-1.5 text-xs text-slate-500">
                  <GitBranch aria-hidden="true" className="h-3 w-3" />
                  Multimodal model
                </dd>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3">
                <dt className="text-xs uppercase tracking-wide text-slate-500">
                  Modalities used
                </dt>
                <dd className="mt-1 flex flex-wrap gap-1.5">
                  {availableModalities.length > 0 ? (
                    availableModalities.map((key) => (
                      <span
                        key={key}
                        className="rounded-md bg-slate-800 px-2 py-0.5 text-xs text-slate-300"
                      >
                        {MODALITY_META[key].label}
                      </span>
                    ))
                  ) : (
                    <span className="text-sm text-slate-500">—</span>
                  )}
                </dd>
              </div>
            </dl>
          </CardBody>
        </Card>
      </div>

      {/* Probability cards — one per modality, always visible */}
      <section aria-labelledby="probabilities-heading">
        <h2 id="probabilities-heading" className="text-lg font-semibold tracking-tight text-white">
          Per-modality probabilities
        </h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {MODALITY_KEYS.map((key) => (
            <ProbabilityCard
              key={key}
              modality={key}
              probability={modalities[key]?.probability ?? null}
              contribution={modalities[key]?.contribution ?? null}
              color={MODALITY_META[key].color}
              available={Boolean(modalities[key]?.available && modalities[key]?.probability != null)}
            />
          ))}
        </div>
      </section>

      {/* Charts */}
      <section aria-labelledby="charts-heading" className="space-y-4">
        <h2 id="charts-heading" className="text-lg font-semibold tracking-tight text-white">
          Visual analysis
        </h2>

        <div className="grid gap-4 lg:grid-cols-2">
          <Card className="p-5">
            <CardHeader
              title="Probability comparison"
              subtitle="PD probability per available modality and the fused estimate."
            />
            <CardBody>
              {probabilityEntries.length > 0 ? (
                <ModalityBarChart entries={probabilityEntries} />
              ) : (
                <EmptyState
                  title="No probabilities to chart"
                  description="The API returned no numeric probabilities for this run."
                />
              )}
            </CardBody>
          </Card>

          <Card className="p-5">
            <CardHeader
              title="Modality contribution"
              subtitle="Relative weight of each modality inside the fused decision."
            />
            <CardBody>
              {contributionEntries.length > 0 ? (
                <ContributionPie entries={contributionEntries} />
              ) : (
                <EmptyState
                  title="No contribution data"
                  description="The API did not report per-modality fusion contributions."
                />
              )}
            </CardBody>
          </Card>
        </div>
      </section>

      {/* Explainability */}
      <section aria-labelledby="explainability-heading">
        <div className="flex items-center justify-between">
          <div>
            <h2 id="explainability-heading" className="text-lg font-semibold tracking-tight text-white">
              Explainability — SHAP feature contributions
            </h2>
            <p className="mt-1 text-sm text-slate-400">
              All attribution values are returned by the analysis API; the UI never
              fabricates explanations.
            </p>
          </div>
          <CalendarClock aria-hidden="true" className="hidden h-5 w-5 text-slate-600 sm:block" />
        </div>

        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          {['voice', 'handwriting', 'gait', 'multimodal'].map((key) => {
            const features = explanations?.[key] ?? [];
            const color =
              key === 'multimodal'
                ? FUSION_COLOR
                : MODALITY_META[key]?.color ?? FUSION_COLOR;
            return (
              <Card key={key} className="p-5">
                <CardHeader
                  title={EXPLANATION_LABELS[key]}
                  subtitle={
                    features.length > 0
                      ? `${features.length} features attributed by the model`
                      : 'No attributions returned for this stream.'
                  }
                />
                <CardBody>
                  {features.length > 0 ? (
                    <ShapChart features={features} color={color} />
                  ) : (
                    <EmptyState
                      title="No SHAP values available"
                      description="This modality was not part of the run, or the API returned no attributions."
                    />
                  )}
                </CardBody>
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
}
