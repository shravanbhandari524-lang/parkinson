/**
 * Upload validation shared by the assessment page and upload cards.
 * Feature files are expected as CSV/TXT exports from the dataset pipelines.
 */

export const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

export const MODALITY_EXTENSIONS = {
  voice: ['.csv', '.txt'],
  handwriting: ['.csv', '.txt'],
  gait: ['.csv', '.txt'],
};

export function getFileExtension(name = '') {
  const idx = name.lastIndexOf('.');
  return idx === -1 ? '' : name.slice(idx).toLowerCase();
}

/**
 * Validate an uploaded file for a modality.
 * @returns {string|null} a human-readable error message, or null when valid.
 */
export function validateFile(modality, file) {
  if (!file) return null;

  const allowed = MODALITY_EXTENSIONS[modality];
  if (!allowed) return `Unknown modality "${modality}".`;

  const ext = getFileExtension(file.name);
  if (!allowed.includes(ext)) {
    return `Unsupported file type "${ext || 'unknown'}". Expected ${allowed.join(' or ')}.`;
  }

  if (file.size === 0) {
    return 'The selected file is empty.';
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    const mb = (MAX_FILE_SIZE_BYTES / (1024 * 1024)).toFixed(0);
    return `File exceeds the ${mb} MB limit (${(file.size / (1024 * 1024)).toFixed(1)} MB).`;
  }

  return null;
}

/**
 * Describe the current analysis mode from the selected files.
 * @returns {{ mode: string, count: number, keys: string[] }}
 */
export function getAnalysisMode(files) {
  const keys = ['voice', 'handwriting', 'gait'].filter((key) => files?.[key]);
  const labels = { voice: 'Voice', handwriting: 'Handwriting', gait: 'Gait' };
  if (keys.length === 0) return { mode: 'none', count: 0, keys };
  if (keys.length === 3) return { mode: 'Multimodal fusion (voice + handwriting + gait)', count: 3, keys };
  if (keys.length === 1) return { mode: `${labels[keys[0]]}-only analysis`, count: 1, keys };
  return { mode: `Bimodal analysis (${keys.map((k) => labels[k]).join(' + ')})`, count: 2, keys };
}
