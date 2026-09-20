/**
 * Formatting helpers shared across pages and components.
 * All display values are formatted here so tests and UI stay consistent.
 */

export const RISK_BANDS = {
  low: {
    label: 'Low risk',
    short: 'Low',
    text: 'text-emerald-300',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    hex: '#34d399',
    summary:
      'The multimodal pattern is closer to the healthy reference group in this model.',
  },
  borderline: {
    label: 'Borderline risk',
    short: 'Borderline',
    text: 'text-amber-300',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
    hex: '#fbbf24',
    summary:
      'The fused probability falls between the low and high decision thresholds.',
  },
  high: {
    label: 'High risk',
    short: 'High',
    text: 'text-red-300',
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    hex: '#f87171',
    summary:
      'The fused probability exceeds the high-risk decision threshold of the model.',
  },
};

/** Clamp a number into [0, 1]; returns null for non-finite input. */
export function clamp01(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return Math.min(1, Math.max(0, n));
}

/** 0.724 -> "72.4%" ; null/undefined -> em dash placeholder. */
export function formatPercent(value, { digits = 1 } = {}) {
  const clamped = clamp01(value);
  if (clamped === null) return '—';
  return `${(clamped * 100).toFixed(digits)}%`;
}

/** ISO string -> "Sep 20, 2026, 10:30 AM" style local timestamp. */
export function formatTimestamp(iso) {
  if (!iso) return '—';
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** Bytes -> "1.4 MB" / "820 KB". */
export function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Resolve the risk band for display. The API's own band string is preferred;
 * thresholds (<0.35 low, 0.35–0.65 borderline, >0.65 high) are only used to
 * pick a colour when the API did not send a band.
 */
export function riskBandInfo(band, probability) {
  const key = String(band ?? '').toLowerCase();
  if (key.includes('high')) return RISK_BANDS.high;
  if (key.includes('border') || key.includes('moder')) return RISK_BANDS.borderline;
  if (key.includes('low')) return RISK_BANDS.low;
  const p = clamp01(probability);
  if (p === null) return RISK_BANDS.low;
  if (p > 0.65) return RISK_BANDS.high;
  if (p >= 0.35) return RISK_BANDS.borderline;
  return RISK_BANDS.low;
}
