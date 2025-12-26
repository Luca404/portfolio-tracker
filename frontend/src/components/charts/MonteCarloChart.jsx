import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function MonteCarloChart({ data }) {
  if (!data || !data.percentiles) return null;

  const currentValue = data.current_value;

  // Convert to percentages relative to current value
  const chartData = data.dates.map((date, i) => ({
    date,
    p5: ((data.percentiles.p5[i] - currentValue) / currentValue) * 100,
    p25: ((data.percentiles.p25[i] - currentValue) / currentValue) * 100,
    p50: ((data.percentiles.p50[i] - currentValue) / currentValue) * 100,
    p75: ((data.percentiles.p75[i] - currentValue) / currentValue) * 100,
    p95: ((data.percentiles.p95[i] - currentValue) / currentValue) * 100
  }));

  const finalP95 = data.percentiles.p95[data.percentiles.p95.length - 1];
  const finalP50 = data.percentiles.p50[data.percentiles.p50.length - 1];
  const finalP5 = data.percentiles.p5[data.percentiles.p5.length - 1];

  const pctP95 = ((finalP95 - currentValue) / currentValue) * 100;
  const pctP50 = ((finalP50 - currentValue) / currentValue) * 100;
  const pctP5 = ((finalP5 - currentValue) / currentValue) * 100;

  return (
    <div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-green-50 rounded-lg p-4 text-center">
          <p className="text-sm text-green-700 mb-1">Best Case (95th)</p>
          <p className="text-2xl font-bold text-green-900">
            {pctP95 >= 0 ? '+' : ''}{pctP95.toFixed(1)}%
          </p>
          <p className="text-xs text-green-600 mt-1">
            ({finalP95?.toLocaleString('en-US', { style: 'currency', currency: 'EUR' })})
          </p>
        </div>
        <div className="bg-blue-50 rounded-lg p-4 text-center">
          <p className="text-sm text-blue-700 mb-1">Median (50th)</p>
          <p className="text-2xl font-bold text-blue-900">
            {pctP50 >= 0 ? '+' : ''}{pctP50.toFixed(1)}%
          </p>
          <p className="text-xs text-blue-600 mt-1">
            ({finalP50?.toLocaleString('en-US', { style: 'currency', currency: 'EUR' })})
          </p>
        </div>
        <div className="bg-red-50 rounded-lg p-4 text-center">
          <p className="text-sm text-red-700 mb-1">Worst Case (5th)</p>
          <p className="text-2xl font-bold text-red-900">
            {pctP5 >= 0 ? '+' : ''}{pctP5.toFixed(1)}%
          </p>
          <p className="text-xs text-red-600 mt-1">
            ({finalP5?.toLocaleString('en-US', { style: 'currency', currency: 'EUR' })})
          </p>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="date" tick={{ fontSize: 12 }} />
          <YAxis tickFormatter={(v) => `${v.toFixed(0)}%`} />
          <Tooltip formatter={(value) => [`${value.toFixed(2)}%`, '']} />
          <Line type="monotone" dataKey="p95" stroke="#10b981" strokeWidth={1.5} dot={false} name="95th percentile" />
          <Line type="monotone" dataKey="p75" stroke="#60a5fa" strokeWidth={1} dot={false} name="75th percentile" strokeDasharray="5 5" />
          <Line type="monotone" dataKey="p50" stroke="#2563eb" strokeWidth={2} dot={false} name="Median" />
          <Line type="monotone" dataKey="p25" stroke="#f59e0b" strokeWidth={1} dot={false} name="25th percentile" strokeDasharray="5 5" />
          <Line type="monotone" dataKey="p5" stroke="#ef4444" strokeWidth={1.5} dot={false} name="5th percentile" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
