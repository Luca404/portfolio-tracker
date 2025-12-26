import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function AssetPerformanceChart({ data, symbols }) {
  const [visibleAssets, setVisibleAssets] = useState(() => {
    // Initially show all assets
    return symbols.reduce((acc, symbol) => ({ ...acc, [symbol]: true }), {});
  });

  const toggleAsset = (symbol) => {
    setVisibleAssets(prev => ({ ...prev, [symbol]: !prev[symbol] }));
  };

  // Color palette for assets
  const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

  return (
    <div className="bg-slate-50 rounded-lg p-4">
      <h3 className="text-lg font-semibold text-slate-900 mb-4">Asset Performance Over Time (Normalized to 100)</h3>

      {/* Legend */}
      <div className="flex flex-wrap gap-2 mb-4">
        {symbols.map((symbol, idx) => (
          <button
            key={symbol}
            onClick={() => toggleAsset(symbol)}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition ${
              visibleAssets[symbol]
                ? 'bg-blue-100 text-blue-700 border border-blue-300'
                : 'bg-slate-100 text-slate-400 border border-slate-200'
            }`}
          >
            <span className="inline-block w-3 h-3 rounded-full mr-2" style={{ backgroundColor: COLORS[idx % COLORS.length] }}></span>
            {symbol}
          </button>
        ))}
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="date" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} label={{ value: 'Normalized Value', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          {symbols.map((symbol, idx) => (
            visibleAssets[symbol] && (
              <Line
                key={symbol}
                type="monotone"
                dataKey={symbol}
                stroke={COLORS[idx % COLORS.length]}
                strokeWidth={2}
                dot={false}
                name={symbol}
                connectNulls
              />
            )
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
