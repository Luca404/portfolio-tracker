import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function DrawdownChart({ data }) {
  if (!data || !data.dates || !data.drawdown) return null;

  const chartData = data.dates.map((date, i) => ({
    date,
    drawdown: data.drawdown[i]
  }));

  return (
    <ResponsiveContainer width="100%" height={350}>
      <LineChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="date" tick={{ fontSize: 12 }} />
        <YAxis tickFormatter={(v) => `${v.toFixed(1)}%`} />
        <Tooltip formatter={(value) => [`${value.toFixed(2)}%`, 'Drawdown']} />
        <Line type="monotone" dataKey="drawdown" stroke="#ef4444" strokeWidth={2} dot={false} fill="#fecaca" />
      </LineChart>
    </ResponsiveContainer>
  );
}
