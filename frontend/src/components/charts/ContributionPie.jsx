import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';
import ChartTooltip from './ChartTooltip.jsx';

/**
 * Donut chart of each modality's relative contribution to the fused score.
 * @param {Array<{ label: string, value: number, color: string }>} entries
 *        contributions from the API (weights), in [0, 1]
 */
export default function ContributionPie({ entries, height = 260 }) {
  const total = entries.reduce((sum, e) => sum + (e.value ?? 0), 0);
  const data = entries
    .filter((e) => (e.value ?? 0) > 0)
    .map((e) => ({
      name: e.label,
      value: Math.round((e.value ?? 0) * 1000) / 1000,
      fill: e.color,
    }));

  if (data.length === 0) {
    return (
      <div
        style={{ height }}
        className="flex items-center justify-center text-sm text-slate-500"
      >
        No contribution data returned by the API.
      </div>
    );
  }

  return (
    <div style={{ height }} role="img" aria-label="Modality contributions to the fused prediction">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            innerRadius="58%"
            outerRadius="85%"
            paddingAngle={3}
            stroke="#0b1220"
            strokeWidth={2}
            isAnimationActive={false}
          >
            {data.map((entry) => (
              <Cell key={entry.name} fill={entry.fill} />
            ))}
          </Pie>
          <Tooltip
            content={
              <ChartTooltip
                valueFormatter={(v) =>
                  `${((v / total) * 100).toFixed(1)}% of fused score`
                }
              />
            }
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
