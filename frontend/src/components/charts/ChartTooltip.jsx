/**
 * Shared tooltip content for all Recharts charts — keeps the dark theme
 * consistent instead of Recharts' default white tooltip.
 */
export default function ChartTooltip({ active, payload, label, labelFormatter, valueFormatter }) {
  if (!active || !payload || payload.length === 0) return null;
  const shownLabel =
    typeof labelFormatter === 'function' ? labelFormatter(label, payload) : label;

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/95 px-3 py-2 text-xs shadow-xl shadow-black/40">
      {shownLabel !== undefined && shownLabel !== null && (
        <p className="mb-1 max-w-[220px] truncate font-semibold text-slate-200">
          {shownLabel}
        </p>
      )}
      <ul className="space-y-0.5">
        {payload.map((entry) => (
          <li key={entry.dataKey ?? entry.name} className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className="h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: entry.color ?? entry.payload?.fill }}
            />
            <span className="text-slate-400">{entry.name}:</span>
            <span className="font-mono tabular-nums text-slate-100">
              {valueFormatter ? valueFormatter(entry.value, entry) : entry.value}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
