import { ShieldAlert } from 'lucide-react';
import { cn } from '../utils/cn.js';

/**
 * The mandatory research/education disclaimer shown on Home and Assessment.
 * `compact` renders a single-line footer strip; default renders a callout.
 */
export default function Disclaimer({ compact = false, className = '' }) {
  if (compact) {
    return (
      <p
        className={cn(
          'flex items-start gap-2 rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-200/90',
          className,
        )}
      >
        <ShieldAlert aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" />
        This application is a research and educational prototype and is not intended
        to provide a medical diagnosis.
      </p>
    );
  }

  return (
    <aside
      aria-label="Medical disclaimer"
      className={cn(
        'flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3.5',
        className,
      )}
    >
      <ShieldAlert aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
      <div>
        <p className="text-sm font-semibold text-amber-200">Research use only</p>
        <p className="mt-0.5 text-sm leading-relaxed text-amber-100/80">
          This application is a research and educational prototype and is not
          intended to provide a medical diagnosis.
        </p>
      </div>
    </aside>
  );
}
