import { Footprints, Mic, PenLine } from 'lucide-react';

/**
 * Single source of truth for modality presentation: labels, icons,
 * accepted uploads, and the chart colour each modality keeps everywhere.
 */
export const MODALITY_META = {
  voice: {
    key: 'voice',
    label: 'Voice',
    plural: 'Voice Biomarkers',
    accept: '.csv,.txt',
    icon: Mic,
    color: '#38bdf8', // sky-400
    tint: 'text-sky-300 bg-sky-500/10 border-sky-500/20',
    fieldName: 'voice',
    description:
      'Sustained-phonation recordings are summarised into acoustic features that capture vocal tremor and dysphonia.',
    featureHint: 'MDVP jitter & shimmer, NHR, HNR, RPDE, DFA, PPE…',
    uploadTitle: 'Upload voice feature file',
    uploadHint: 'CSV/TXT export of acoustic features (one sample row).',
  },
  handwriting: {
    key: 'handwriting',
    label: 'Handwriting',
    plural: 'Handwriting Biomarkers',
    accept: '.csv,.txt',
    icon: PenLine,
    color: '#a78bfa', // violet-400
    tint: 'text-violet-300 bg-violet-500/10 border-violet-500/20',
    fieldName: 'handwriting',
    description:
      'Spiral and meander drawing exams capture micrographia-related kinematics such as stroke dynamics and speed variability.',
    featureHint: 'RMS, max/min/std between-stroke metrics, MRT, sign changes…',
    uploadTitle: 'Upload handwriting feature file',
    uploadHint: 'CSV/TXT export of HandPD signal features (one exam row).',
  },
  gait: {
    key: 'gait',
    label: 'Gait',
    plural: 'Gait Biomarkers',
    accept: '.csv,.txt',
    icon: Footprints,
    color: '#34d399', // emerald-400
    tint: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/20',
    fieldName: 'gait',
    description:
      'Force-plate walking records are reduced to cadence, stride regularity, and left/right ground-reaction-force asymmetry.',
    featureHint: 'Cadence, stride CV, stance/swing timing, L/R symmetry…',
    uploadTitle: 'Upload gait feature file',
    uploadHint: 'CSV/TXT export of engineered gait features (one record row).',
  },
};

export const MODALITY_KEYS = ['voice', 'handwriting', 'gait'];

/** Teal reserved for the fused / overall outputs. */
export const FUSION_COLOR = '#2dd4bf';
