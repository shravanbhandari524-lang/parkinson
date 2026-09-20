import { Link } from 'react-router-dom';
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  FileText,
  Footprints,
  Mic,
  PenLine,
} from 'lucide-react';
import Disclaimer from '../components/Disclaimer.jsx';
import { Card } from '../components/ui/Card.jsx';
import Button from '../components/ui/Button.jsx';

const BIOMARKERS = [
  {
    icon: Mic,
    tint: 'text-sky-300 bg-sky-500/10 border-sky-500/20',
    title: 'Voice Biomarkers',
    body: 'Sustained phonation summarised into acoustic features — jitter, shimmer, harmonic-to-noise ratio and nonlinear measures — that reflect vocal-fold control affected by Parkinson\u2019s.',
  },
  {
    icon: PenLine,
    tint: 'text-violet-300 bg-violet-500/10 border-violet-500/20',
    title: 'Handwriting Biomarkers',
    body: 'Spiral and meander drawing exams quantified into kinematic signal features such as stroke dynamics, speed variability and pressure changes associated with micrographia.',
  },
  {
    icon: Footprints,
    tint: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/20',
    title: 'Gait Biomarkers',
    body: 'Force-plate walking records reduced to cadence, stride regularity, stance/swing timing and left–right ground-reaction-force asymmetry.',
  },
  {
    icon: BrainCircuit,
    tint: 'text-teal-300 bg-teal-500/10 border-teal-500/20',
    title: 'Explainable AI',
    body: 'SHAP feature attributions per modality and per fusion decision, so every prediction can be traced to the biomarkers that drove it.',
  },
];

export default function Home() {
  return (
    <div className="space-y-16 animate-fade-in">
      {/* Hero */}
      <section className="relative overflow-hidden rounded-3xl border border-slate-800 bg-surface-raised px-6 py-14 text-center sm:px-12 sm:py-20">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 opacity-[0.35]"
          style={{
            backgroundImage:
              'radial-gradient(circle at 25% 20%, rgba(45, 212, 191, 0.08) 0, transparent 45%), radial-gradient(circle at 80% 70%, rgba(56, 189, 248, 0.06) 0, transparent 45%)',
          }}
        />
        <div className="relative mx-auto max-w-3xl">
          <span className="inline-flex items-center gap-2 rounded-full border border-teal-500/30 bg-teal-500/10 px-3 py-1 text-xs font-medium text-teal-300">
            <Activity aria-hidden="true" className="h-3.5 w-3.5" />
            Multimodal AI · Research Framework
          </span>
          <h1 className="mt-6 text-4xl font-bold leading-tight tracking-tight text-white text-balance sm:text-5xl">
            Parkinson&apos;s Disease
            <span className="block text-teal-300">Multimodal AI Risk Assessment</span>
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-base leading-relaxed text-slate-400 sm:text-lg">
            An explainable machine-learning framework that combines voice, handwriting
            and gait biomarkers to estimate early Parkinson&apos;s disease risk — with
            every prediction traced back to the features that produced it.
          </p>
          <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Link to="/assessment">
              <Button size="lg">
                Start Assessment
                <ArrowRight aria-hidden="true" className="h-4 w-4" />
              </Button>
            </Link>
            <Link to="/methodology">
              <Button size="lg" variant="secondary">
                <FileText aria-hidden="true" className="h-4 w-4" />
                Read the Methodology
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Biomarker cards */}
      <section aria-labelledby="biomarkers-heading">
        <h2
          id="biomarkers-heading"
          className="text-xl font-semibold tracking-tight text-white"
        >
          Three modalities, one explainable decision
        </h2>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
          Each biomarker stream is modelled independently, then fused so that partial
          assessments remain possible — and every modality stays individually
          interpretable.
        </p>
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {BIOMARKERS.map((item) => {
            const Icon = item.icon;
            return (
              <Card key={item.title} className="p-5 transition-colors hover:border-slate-700">
                <span
                  className={`inline-flex h-10 w-10 items-center justify-center rounded-lg border ${item.tint}`}
                >
                  <Icon aria-hidden="true" className="h-5 w-5" />
                </span>
                <h3 className="mt-4 text-sm font-semibold text-slate-100">
                  {item.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-400">{item.body}</p>
              </Card>
            );
          })}
        </div>
      </section>

      {/* Pipeline teaser */}
      <section
        aria-labelledby="pipeline-heading"
        className="rounded-2xl border border-slate-800 bg-surface-raised p-6 sm:p-8"
      >
        <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
          <div className="max-w-xl">
            <h2 id="pipeline-heading" className="text-xl font-semibold tracking-tight text-white">
              From raw signals to an explainable risk band
            </h2>
            <p className="mt-2 text-sm leading-relaxed text-slate-400">
              XGBoost and Random Forest models score each modality, a TabNet-style
              fusion network combines their calibrated probabilities, and SHAP
              attributions explain the outcome.
            </p>
          </div>
          <ol className="grid grid-cols-2 gap-2 text-xs text-slate-400 sm:grid-cols-4">
            {['Voice · XGBoost', 'Handwriting · Random Forest', 'Gait · XGBoost', 'Fusion · TabNet + SHAP'].map(
              (step, index) => (
                <li
                  key={step}
                  className="rounded-lg border border-slate-800 bg-slate-900/50 px-3 py-2.5 text-center"
                >
                  <span className="mb-1 block font-mono text-[10px] text-slate-600">
                    0{index + 1}
                  </span>
                  {step}
                </li>
              ),
            )}
          </ol>
        </div>
      </section>

      {/* Disclaimer */}
      <Disclaimer className="mx-auto max-w-3xl" />
    </div>
  );
}
