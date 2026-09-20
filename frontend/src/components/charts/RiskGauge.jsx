import { RadialBar, RadialBarChart, ResponsiveContainer } from 'recharts';
import { clamp01 } from '../../utils/format.js';

/**
 * 270° circular gauge for the overall Parkinson's probability.
 * @param {number|null} probability 0–1
 * @param {string} color  hex accent for the filled arc
 */
export default function RiskGauge({
  probability,
  color = '#2dd4bf',
  label = 'Overall PD probability',
  size = 'h-52',
}) {
  const clamped = clamp01(probability);
  const pct = clamped === null ? 0 : Math.round(clamped * 100);
  const data = [{ name: 'pd-risk', value: pct, fill: color }];

  return (
    <div
      className={`relative mx-auto w-full max-w-[260px] ${size}`}
      role="img"
      aria-label={`${label}: ${pct} percent`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <RadialBarChart
          innerRadius="74%"
          outerRadius="98%"
          data={data}
          startAngle={210}
          endAngle={-30}
          barSize={16}
        >
          <RadialBar
            background={{ fill: '#1e293b' }}
            dataKey="value"
            cornerRadius={8}
            isAnimationActive={false}
          />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center pt-2">
        <span className="text-4xl font-bold tracking-tight text-white">
          {clamped === null ? '—' : `${pct}%`}
        </span>
        <span className="mt-1 max-w-[160px] text-center text-[11px] uppercase tracking-wider text-slate-500">
          {label}
        </span>
      </div>
    </div>
  );
}
