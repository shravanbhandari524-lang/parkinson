import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import ChartTooltip from './ChartTooltip.jsx';

/**
 * Column chart comparing PD probability per available modality + fusion.
 * @param {Array<{ label: string, value: number, color: string }>} entries
 *        values are probabilities in [0, 1]
 */
export default function ModalityBarChart({ entries, height = 280 }) {
  const data = entries.map((e) => ({
    label: e.label,
    value: Math.round((e.value ?? 0) * 1000) / 10,
    fill: e.color,
  }));

  return (
    <div
      style={{ height }}
      role="img"
      aria-label={`Modality probability comparison: ${entries
        .map((e) => `${e.label} ${(e.value * 100).toFixed(1)}%`)
        .join(', ')}`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
          <CartesianGrid stroke="#1e293b" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: '#94a3b8', fontSize: 12 }}
            axisLine={{ stroke: '#334155' }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v) => `${v}%`}
          />
          <Tooltip
            cursor={{ fill: 'rgba(148, 163, 184, 0.08)' }}
            content={<ChartTooltip valueFormatter={(v) => `${v}%`} />}
          />
          <Bar dataKey="value" name="PD probability" radius={[6, 6, 0, 0]} maxBarSize={56}>
            {data.map((entry) => (
              <Cell key={entry.label} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
