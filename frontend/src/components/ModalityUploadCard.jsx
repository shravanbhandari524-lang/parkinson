import { useRef, useState } from 'react';
import { CloudUpload, FileCheck2, RotateCcw, X } from 'lucide-react';
import { MODALITY_META } from '../utils/modalityMeta.js';
import { validateFile } from '../utils/validation.js';
import { formatFileSize } from '../utils/format.js';
import { cn } from '../utils/cn.js';

/**
 * Upload card for one biomarker modality (voice / handwriting / gait).
 * Validation runs inside the card; the parent only receives valid files.
 *
 * @param {object} props
 * @param {'voice'|'handwriting'|'gait'} props.modality
 * @param {File|null} props.file
 * @param {(modality: string, file: File|null) => void} props.onFileChange
 * @param {boolean} [props.disabled]
 */
export default function ModalityUploadCard({
  modality,
  file,
  onFileChange,
  disabled = false,
}) {
  const meta = MODALITY_META[modality];
  const Icon = meta.icon;
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [validationError, setValidationError] = useState(null);

  const acceptFile = (candidate) => {
    if (!candidate) return;
    const error = validateFile(modality, candidate);
    setValidationError(error);
    onFileChange(modality, error ? null : candidate);
  };

  const handleSelect = (event) => {
    const selected = event.target.files?.[0];
    event.target.value = ''; // allow re-selecting the same file
    acceptFile(selected);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    if (disabled) return;
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  };

  const handleRemove = () => {
    setValidationError(null);
    onFileChange(modality, null);
  };

  return (
    <section
      aria-label={`${meta.label} biomarker upload`}
      className="flex flex-col rounded-2xl border border-slate-800 bg-surface-raised p-5"
    >
      {/* Header */}
      <div className="flex items-start gap-3">
        <span
          className={cn(
            'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border',
            meta.tint,
          )}
        >
          <Icon aria-hidden="true" className="h-5 w-5" />
        </span>
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-200">
            {meta.label}
          </h3>
          <p className="mt-1 text-xs leading-relaxed text-slate-500">
            {meta.description}
          </p>
        </div>
      </div>

      {/* Dropzone / selected file */}
      <div className="mt-4 flex-1">
        {file ? (
          <div
            className="flex h-full flex-col justify-between rounded-xl border border-teal-500/30 bg-teal-500/5 px-4 py-3.5 animate-fade-in"
            data-testid={`${modality}-file-selected`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-start gap-2.5">
                <FileCheck2 aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-teal-400" />
                <div className="min-w-0">
                  <p
                    className="truncate text-sm font-medium text-slate-100"
                    title={file.name}
                  >
                    {file.name}
                  </p>
                  <p className="mt-0.5 text-xs text-slate-500">
                    {formatFileSize(file.size)} · ready for analysis
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={handleRemove}
                disabled={disabled}
                aria-label={`Remove ${meta.label.toLowerCase()} file`}
                className="rounded-md p-1 text-slate-500 transition-colors hover:bg-slate-800 hover:text-slate-200 disabled:opacity-40"
              >
                <X aria-hidden="true" className="h-4 w-4" />
              </button>
            </div>
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={disabled}
              className="mt-3 inline-flex items-center gap-1.5 self-start text-xs font-medium text-teal-300 hover:text-teal-200 disabled:opacity-40"
            >
              <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
              Replace file
            </button>
          </div>
        ) : (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            onDragOver={(event) => {
              event.preventDefault();
              if (!disabled) setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            disabled={disabled}
            aria-label={meta.uploadTitle}
            data-testid={`${modality}-dropzone`}
            className={cn(
              'flex h-full min-h-[150px] w-full flex-col items-center justify-center gap-2 rounded-xl border border-dashed px-4 py-6 text-center transition-colors',
              dragging
                ? 'border-teal-400/60 bg-teal-500/10'
                : 'border-slate-700 bg-slate-900/40 hover:border-slate-600 hover:bg-slate-900/70',
              disabled && 'cursor-not-allowed opacity-50',
            )}
          >
            <CloudUpload
              aria-hidden="true"
              className={cn('h-7 w-7', dragging ? 'text-teal-300' : 'text-slate-500')}
            />
            <span className="text-sm font-medium text-slate-300">
              {meta.uploadTitle}
            </span>
            <span className="text-xs text-slate-500">
              Drag &amp; drop or click to browse
            </span>
            <span className="mt-1 rounded-md bg-slate-800/80 px-2 py-0.5 font-mono text-[11px] text-slate-400">
              {meta.accept} · max 10 MB
            </span>
          </button>
        )}

        {/* Validation error */}
        {validationError && (
          <p
            role="alert"
            className="mt-2 flex items-start gap-1.5 text-xs text-red-300 animate-fade-in"
          >
            <X aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            {validationError}
          </p>
        )}

        <p className="mt-3 text-xs leading-relaxed text-slate-600">
          {meta.featureHint}
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={meta.accept}
        className="sr-only"
        aria-label={meta.uploadTitle}
        onChange={handleSelect}
        disabled={disabled}
      />
    </section>
  );
}
