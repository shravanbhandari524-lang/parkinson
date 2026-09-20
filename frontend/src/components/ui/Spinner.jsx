export function Spinner({ size = 'md', label, className = '' }) {
  const dims = { sm: 'h-4 w-4', md: 'h-6 w-6', lg: 'h-10 w-10' }[size] ?? 'h-6 w-6';
  return (
    <span role="status" className={`inline-flex items-center gap-2 ${className}`}>
      <span
        aria-hidden="true"
        className={`${dims} animate-spin rounded-full border-2 border-slate-600 border-t-teal-400`}
      />
      {label && <span className="text-sm text-slate-400">{label}</span>}
      {!label && <span className="sr-only">Loading…</span>}
    </span>
  );
}

export function FullPageLoader({ label = 'Loading…' }) {
  return (
    <div className="flex min-h-[50vh] flex-col items-center justify-center gap-3 text-slate-400">
      <Spinner size="lg" />
      <p className="text-sm">{label}</p>
    </div>
  );
}
