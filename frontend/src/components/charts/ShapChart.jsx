import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import ChartTooltip from './ChartTooltip.jsx';

const PD_PUSH = '#f87171'; // positive SHAP -> pushes toward PD
const PROTECTIVE = '#34d399'; // negative SHAP -> pushes toward healthy

function truncate(name, max = 22) {
  return name.length > max ? `${name.slice(0, max - 1)}…` : name;
}

/**
 * Diverging horizontal bar chart of per-feature SHAP contributions.
 * Values arrive verbatim from the API's SHAP explainer — never fabricated.
 *
 * @param {Array<{ feature: string, shap: number, value: number|null }>} features
 * @param {number} [topN]
 */
export default function ShapChart({ features, topN = 8, color = '#2dd4bf' }) {
  const data = [...features]
    .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
    .slice(0, topN)
    .map((f) => ({ name: f.feature, shap: f.shap, value: f.value }));

  const height = Math.max(120, data.length * 34 + 48);

  return (
    <div>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 4, right: 24, bottom: 4, left: 8 }}
          >
            <XAxis
              type="number"
              tick={{ fill: '#64748b', fontSize: 11 }}
              axisLine={{ stroke: '#334155' }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={130}
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              tickFormatter={(v) => truncate(v)}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              cursor={{ fill: 'rgba(148, 163, 184, 0.08)' }}
              content={
                <ChartTooltip
                  valueFormatter={(v, entry) =>
                    `SHAP ${Number(v).toFixed(4)}${
                      entry?.payload?.value != null
                        ? ` · feature ${entry.payload.value}`
                        : ''
                    }`
                  }
                />
              }
            />
            <ReferenceLine x={0} stroke="#475569" />
            <Bar dataKey="shap" name="SHAP" barSize={16} radius={[3, 3, 3, 3]}>
              {data.map((entry) => (
                <Cell
                  key={entry.name}
                  fill={entry.shap >= 0 ? PD_PUSH : PROTECTIVE}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Accessible text alternative + stable hook for tests */}
      <ul className="sr-only" data-testid="shap-list">
        {data.map((entry) => (
          <li key={entry.name}>
            {entry.name}: SHAP {entry.shap.toFixed(4)}
            {entry.value != null ? `, feature value ${entry.value}` : ''}
          </li>
        ))}
      </ul>
      <p className="mt-2 flex flex-wrap items-center gap-4 text-[11px] text-slate-500">
        <span className="flex items-center gap-1.5">
          <span aria-hidden="true" className="h-2 w-2 rounded-sm" style={{ backgroundColor: PD_PUSH }} />
          pushes toward PD
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden="true" className="h-2 w-2 rounded-sm" style={{ backgroundColor: PROTECTIVE }} />
          pushes toward healthy
        </span>
        <span className="ml-auto" style={{ color }}>API-supplied SHAP values</span>
      </p>
    </div>
  );
}
