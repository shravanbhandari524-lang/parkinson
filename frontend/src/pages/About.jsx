import { BookOpen, CodeXml, Compass, FlaskConical, Microscope, Scale, TriangleAlert } from 'lucide-react';
import { Card, CardBody, CardHeader } from '../components/ui/Card.jsx';
import Disclaimer from '../components/Disclaimer.jsx';

const SECTIONS = [
  {
    icon: Compass,
    title: 'Project Objective',
    body: 'Build an explainable multimodal AI framework that estimates early Parkinson\u2019s disease risk from three non-invasive biomarker streams — voice, handwriting, and gait — and makes every prediction auditable through feature-level attributions.',
  },
  {
    icon: Microscope,
    title: 'Datasets',
    body: 'UCI Parkinson\u2019s voice dataset (195 recordings, 22 acoustic features), the HandPD handwriting dataset (spiral and meander exams with ten signal features each), and PhysioNet \u201CGait in Parkinson\u2019s Disease\u201D (166 subjects with 100 Hz force-plate walking records and demographics). All labels are harmonised to a single PD/healthy target.',
  },
  {
    icon: CodeXml,
    title: 'Algorithms',
    body: 'XGBoost for voice and gait, Random Forest with isotonic calibration for handwriting, and a TabNet-style attentive network for multimodal fusion. SHAP (TreeSHAP for tree ensembles) supplies per-prediction feature attributions across every modality and the fused decision.',
  },
  {
    icon: FlaskConical,
    title: 'Research Gap',
    body: 'Most published Parkinson\u2019s screening models rely on a single biomarker stream, report accuracy from leakage-prone random splits, and offer little explanation of individual predictions. Few works combine calibrated per-modality probabilities in a patient-safe, group-aware pipeline with per-decision explainability.',
  },
  {
    icon: Scale,
    title: 'Proposed Contribution',
    body: 'A group-aware, leakage-safe training protocol across three modalities; a calibrated fusion stage that degrades gracefully when a modality is missing; and an end-to-end explainability surface — SHAP per modality and for the fused decision — exposed through a research dashboard rather than a black-box score.',
  },
  {
    icon: TriangleAlert,
    title: 'Limitations',
    body: 'The voice dataset is recording-level rather than patient-level, so its metrics should be read as proof-of-concept. Handwriting exams repeat patients across rows, and gait records repeat subjects — group splits mitigate but do not eliminate sampling bias. Datasets are small; the framework is a research prototype and not a clinical device.',
  },
  {
    icon: BookOpen,
    title: 'Technology Stack',
    body: 'Frontend: React 18, Vite, Tailwind CSS, React Router, Recharts, Lucide icons, Axios, Vitest. Backend: Express (Node.js) REST API with multipart upload handling. ML: Python, scikit-learn, XGBoost, SHAP, joblib artefacts.',
  },
];

export default function About() {
  return (
    <div className="space-y-8 animate-fade-in">
      <header>
        <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
          About this research
        </h1>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
          Context, motivation, and honest limitations of the multimodal Parkinson&apos;s
          risk framework behind this dashboard.
        </p>
      </header>

      <div className="grid gap-4 lg:grid-cols-2">
        {SECTIONS.map((section) => {
          const Icon = section.icon;
          return (
            <Card key={section.title} className="p-6">
              <CardHeader icon={Icon} title={section.title} />
              <CardBody>
                <p className="text-sm leading-relaxed text-slate-400">{section.body}</p>
              </CardBody>
            </Card>
          );
        })}
      </div>

      <Disclaimer />
    </div>
  );
}
