import { cn } from '../../utils/cn.js';

export default function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className = '',
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-slate-700 bg-surface-raised/50 px-6 py-12 text-center',
        className,
      )}
    >
      {Icon && (
        <span className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-800/80 text-slate-400">
          <Icon aria-hidden="true" className="h-6 w-6" />
        </span>
      )}
      <div>
        <h3 className="text-base font-semibold text-slate-200">{title}</h3>
        {description && (
          <p className="mx-auto mt-1 max-w-md text-sm leading-relaxed text-slate-500">
            {description}
          </p>
        )}
      </div>
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
}
