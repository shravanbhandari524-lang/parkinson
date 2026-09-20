import { LoaderCircle } from 'lucide-react';

/**
 * Button with variants used across the app.
 * `variant`: primary | secondary | ghost | danger
 * `size`: sm | md | lg
 */
const VARIANTS = {
  primary:
    'bg-teal-500 text-slate-950 font-semibold hover:bg-teal-400 active:bg-teal-500 disabled:bg-teal-500/40 disabled:text-slate-950/60',
  secondary:
    'bg-slate-800 text-slate-100 border border-slate-700 hover:bg-slate-700 hover:border-slate-600 disabled:opacity-50',
  ghost:
    'text-slate-300 hover:text-white hover:bg-slate-800/70 disabled:opacity-50',
  danger:
    'bg-red-500/90 text-white hover:bg-red-500 disabled:opacity-50',
};

const SIZES = {
  sm: 'px-3 py-1.5 text-sm rounded-lg gap-1.5',
  md: 'px-4 py-2 text-sm rounded-lg gap-2',
  lg: 'px-5 py-2.5 text-base rounded-xl gap-2',
};

export default function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  children,
  className = '',
  type = 'button',
  ...props
}) {
  return (
    <button
      type={type}
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center transition-colors duration-150 focus-visible:ring-2 focus-visible:ring-teal-400/70 focus-visible:ring-offset-2 focus-visible:ring-offset-surface disabled:cursor-not-allowed ${VARIANTS[variant]} ${SIZES[size]} ${className}`}
      {...props}
    >
      {loading && <LoaderCircle aria-hidden="true" className="h-4 w-4 animate-spin" />}
      {children}
    </button>
  );
}
