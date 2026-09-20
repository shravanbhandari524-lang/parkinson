import { ArrowDown, BrainCircuit, Database, Filter, Layers, Network, ScanSearch, Target } from 'lucide-react';
import { Card } from '../components/ui/Card.jsx';

const PIPELINE = [
  {
    icon: Database,
    title: 'Dataset',
    body: 'Three curated biomarker sources: the UCI Parkinson\u2019s voice dataset (195 recordings, 22 acoustic features), the HandPD handwriting dataset (spiral + meander exams with 10 signal features), and PhysioNet \u201CGait in Parkinson\u2019s Disease\u201D force-plate records (166 subjects, 100 Hz vertical ground reaction).',
  },
  {
    icon: Filter,
    title: 'Preprocessing',
    body: 'Label harmonisation to a shared PD/healthy target, per-modality scaling inside scikit-learn pipelines, correlation pruning of near-duplicate acoustic features, and patient-level integrity checks.',
  },
  {
    icon: ScanSearch,
    title: 'Feature Engineering',
    body: 'Voice and handwriting use their published feature sets. Gait records are reduced to engineered descriptors: cadence via peak detection, stride-time statistics, stance/swing timing, left/right symmetry, and spectral measures in the 0–5 Hz band.',
  },
  {
    icon: Layers,
    title: 'Modality Models',
    body: 'Voice → XGBoost · Handwriting → Random Forest (isotonic calibration) · Gait → XGBoost. Group-aware cross-validation prevents patient leakage; sensitivity and specificity are reported alongside AUC.',
  },
  {
    icon: Network,
    title: 'Multimodal Fusion',
    body: 'Calibrated per-modality probabilities are combined by a fusion model trained on out-of-fold predictions, so no information leaks from validation folds into the meta-model.',
  },
  {
    icon: BrainCircuit,
    title: 'TabNet',
    body: 'A TabNet-style tabular network acts as the learnable fusion head, using sequential attention to select which modality features drive each decision — an attentive, end-to-end alternative to fixed weight averaging.',
  },
  {
    icon: Target,
    title: 'SHAP / Explainability',
    body: 'TreeSHAP explains each modality model; the fusion head is explained on its inputs (per-modality probabilities and selected features), producing per-prediction attributions returned by the API.',
  },
  {
    icon: Target,
    title: 'Risk Prediction',
    body: 'The fused probability is mapped to a risk band (low < 0.35, borderline 0.35–0.65, high > 0.65) and delivered with per-modality probabilities, contributions, and SHAP explanations.',
  },
];

const MODEL_MATRIX = [
  { modality: 'Voice', model: 'XGBoost', note: 'Gradient-boosted trees on 22 acoustic features; handles correlated jitter/shimmer features.' },
  { modality: 'Handwriting', model: 'Random Forest', note: 'Robust on the small HandPD table with isotonic probability calibration.' },
  { modality: 'Gait', model: 'XGBoost', note: 'Gradient-boosted trees on engineered force-plate descriptors.' },
  { modality: 'Fusion', model: 'TabNet', note: 'Attentive tabular fusion of the three calibrated probabilities.' },
];

export default function Methodology() {
  return (
    <div className="space-y-10 animate-fade-in">
      <header>
        <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
          Methodology
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
          The complete modelling pipeline, from raw biomarker datasets to the
          explainable risk prediction served by the API.
        </p>
      </header>

      {/* Pipeline diagram */}
      <section aria-label="Pipeline overview" className="mx-auto max-w-3xl">
        {PIPELINE.map((step, index) => {
          const Icon = step.icon;
          const last = index === PIPELINE.length - 1;
          return (
            <div key={step.title}>
              <div className="flex items-start gap-4">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-teal-500/30 bg-teal-500/10 text-teal-300">
                  <Icon aria-hidden="true" className="h-5 w-5" />
                </span>
                <div className="min-w-0 pb-2">
                  <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-200">
                    <span className="mr-2 font-mono text-xs text-teal-400/80">
                      0{index + 1}
                    </span>
                    {step.title}
                  </h2>
                  <p className="mt-1.5 text-sm leading-relaxed text-slate-400">
                    {step.body}
                  </p>
                </div>
              </div>
              {!last && (
                <div className="ml-5 flex h-8 items-center" aria-hidden="true">
                  <ArrowDown className="h-4 w-4 text-slate-700" />
                </div>
              )}
            </div>
          );
        })}
      </section>

      {/* Model matrix */}
      <section aria-labelledby="models-heading">
        <h2 id="models-heading" className="text-lg font-semibold tracking-tight text-white">
          Model assignment per modality
        </h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          {MODEL_MATRIX.map((row) => (
            <Card key={row.modality} className="p-5">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold text-slate-200">{row.modality}</h3>
                <span className="rounded-md border border-teal-500/30 bg-teal-500/10 px-2 py-0.5 font-mono text-xs text-teal-300">
                  {row.model}
                </span>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{row.note}</p>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
