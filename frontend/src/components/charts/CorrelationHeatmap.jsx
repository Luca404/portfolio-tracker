export default function CorrelationHeatmap({ data }) {
  if (!data || !data.symbols || !data.matrix) return null;

  const symbols = data.symbols;
  const matrix = data.matrix;

  const getColor = (value) => {
    if (value >= 0.8) return 'bg-green-700 text-white';
    if (value >= 0.5) return 'bg-green-500 text-white';
    if (value >= 0.2) return 'bg-green-300 text-slate-900';
    if (value >= -0.2) return 'bg-slate-100 text-slate-900';
    if (value >= -0.5) return 'bg-red-300 text-slate-900';
    if (value >= -0.8) return 'bg-red-500 text-white';
    return 'bg-red-700 text-white';
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className="p-2 text-sm font-semibold text-slate-700 border border-slate-200 bg-slate-50"></th>
            {symbols.map((sym) => (
              <th key={sym} className="p-2 text-sm font-semibold text-slate-700 border border-slate-200 bg-slate-50 min-w-[80px]">
                {sym}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {symbols.map((symRow, i) => (
            <tr key={symRow}>
              <td className="p-2 text-sm font-semibold text-slate-700 border border-slate-200 bg-slate-50">
                {symRow}
              </td>
              {symbols.map((symCol, j) => {
                const value = matrix[i][j];
                return (
                  <td
                    key={`${symRow}-${symCol}`}
                    className={`p-2 text-center text-sm font-semibold border border-slate-200 ${getColor(value)}`}
                  >
                    {value.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-4 flex items-center justify-center gap-6 text-xs text-slate-600">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-700 rounded"></div>
          <span>Strong positive (0.8+)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-slate-100 border border-slate-300 rounded"></div>
          <span>Neutral (-0.2 to 0.2)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-700 rounded"></div>
          <span>Strong negative (-0.8-)</span>
        </div>
      </div>
    </div>
  );
}
