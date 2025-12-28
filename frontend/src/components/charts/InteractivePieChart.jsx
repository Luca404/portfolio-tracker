import { useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Sector } from 'recharts';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];

const renderActiveShape = (props) => {
  const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill } = props;

  return (
    <g>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius + 8}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
    </g>
  );
};

function InteractivePieChart({ data, totalValue, formatValue, currency }) {
  const [activeIndex, setActiveIndex] = useState(null);

  const onPieEnter = (_, index) => {
    setActiveIndex(index);
  };

  const onPieLeave = () => {
    setActiveIndex(null);
  };

  const onLegendEnter = (index) => {
    setActiveIndex(index);
  };

  const onLegendLeave = () => {
    setActiveIndex(null);
  };

  if (!data || data.length === 0) {
    return <p className="text-center text-slate-500 py-16">No positions</p>;
  }

  return (
    <div className="flex flex-col lg:flex-row gap-6">
      <div className="flex-1 select-none [&_*]:outline-none">
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
              onMouseEnter={onPieEnter}
              onMouseLeave={onPieLeave}
              activeIndex={activeIndex}
              activeShape={renderActiveShape}
              style={{ outline: 'none' }}
            >
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                  opacity={activeIndex !== null && activeIndex !== index ? 0.5 : 1}
                  style={{ cursor: 'pointer', outline: 'none' }}
                />
              ))}
            </Pie>
            <Tooltip
              formatter={(value) => formatValue(value, currency)}
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e2e8f0',
                borderRadius: '0.5rem',
                padding: '0.5rem'
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <div className="flex-shrink-0 lg:w-64">
        <div className="space-y-2">
          {data.map((entry, index) => {
            const percentage = totalValue > 0 ? (entry.value / totalValue) * 100 : 0;
            const isActive = activeIndex === index;

            return (
              <div
                key={`legend-${index}`}
                className={`flex items-center justify-between p-2 rounded transition-all cursor-pointer ${
                  isActive
                    ? 'bg-slate-100 shadow-sm'
                    : activeIndex !== null
                      ? 'opacity-50'
                      : 'hover:bg-slate-50'
                }`}
                onMouseEnter={() => onLegendEnter(index)}
                onMouseLeave={onLegendLeave}
              >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <div
                    className="w-3 h-3 rounded-full flex-shrink-0"
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="text-sm font-medium text-slate-700 truncate">
                    {entry.name}
                  </span>
                </div>
                <div className="flex flex-col items-end ml-2">
                  <span className="text-sm font-bold text-slate-900">
                    {percentage.toFixed(1)}%
                  </span>
                  <span className="text-xs text-slate-500">
                    {formatValue(entry.value, currency)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {data.length > 8 && (
          <p className="text-xs text-slate-400 mt-3 text-center">
            Hover over items to highlight
          </p>
        )}
      </div>
    </div>
  );
}

export default InteractivePieChart;
