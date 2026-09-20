import { CircleCheckBig, Info, TriangleAlert } from 'lucide-react';

const STYLES = {
  error: {
    icon: TriangleAlert,
    cls: 'border-red-500/30 bg-red-500/10 text-red-200',
    iconCls: 'text-red-400',
  },
  warning: {
    icon: TriangleAlert,
    cls: 'border-amber-500/30 bg-amber-500/10 text-amber-200',
    iconCls: 'text-amber-400',
  },
  info: {
    icon: Info,
    cls: 'border-sky-500/30 bg-sky-500/10 text-sky-200',
    iconCls: 'text-sky-400',
  },
  success: {
    icon: CircleCheckBig,
    cls: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200',
    iconCls: 'text-emerald-400',
  },
};

export default function Alert({ variant = 'info', title, children, className = '' }) {
  const style = STYLES[variant] ?? STYLES.info;
  const Icon = style.icon;
  return (
    <div
      role={variant === 'error' ? 'alert' : 'status'}
      className={`flex items-start gap-3 rounded-xl border px-4 py-3 text-sm ${style.cls} ${className}`}
    >
      <Icon aria-hidden="true" className={`mt-0.5 h-4 w-4 shrink-0 ${style.iconCls}`} />
      <div className="min-w-0">
        {title && <p className="font-semibold">{title}</p>}
        {children && <div className={title ? 'mt-0.5' : ''}>{children}</div>}
      </div>
    </div>
  );
}
