import React, { useState, useEffect, useRef } from 'react';
import { Info } from 'lucide-react';

// Risk Metric Card with Info Tooltip
function MetricCard({ title, value, subtitle, color, description, interpretation, showAbove = false }) {
  const [showTooltip, setShowTooltip] = useState(false);
  const cardRef = useRef(null);

  // Close tooltip when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (cardRef.current && !cardRef.current.contains(event.target)) {
        setShowTooltip(false);
      }
    };

    if (showTooltip) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showTooltip]);

  // Define color classes statically so Tailwind can detect them
  const colorClasses = {
    blue: {
      bg: 'bg-gradient-to-br from-blue-50 to-blue-100',
      title: 'text-blue-700',
      value: 'text-blue-900',
      subtitle: 'text-blue-600',
      icon: 'text-blue-600',
      hover: 'hover:bg-blue-200'
    },
    purple: {
      bg: 'bg-gradient-to-br from-purple-50 to-purple-100',
      title: 'text-purple-700',
      value: 'text-purple-900',
      subtitle: 'text-purple-600',
      icon: 'text-purple-600',
      hover: 'hover:bg-purple-200'
    },
    orange: {
      bg: 'bg-gradient-to-br from-orange-50 to-orange-100',
      title: 'text-orange-700',
      value: 'text-orange-900',
      subtitle: 'text-orange-600',
      icon: 'text-orange-600',
      hover: 'hover:bg-orange-200'
    },
    red: {
      bg: 'bg-gradient-to-br from-red-50 to-red-100',
      title: 'text-red-700',
      value: 'text-red-900',
      subtitle: 'text-red-600',
      icon: 'text-red-600',
      hover: 'hover:bg-red-200'
    },
    green: {
      bg: 'bg-gradient-to-br from-green-50 to-green-100',
      title: 'text-green-700',
      value: 'text-green-900',
      subtitle: 'text-green-600',
      icon: 'text-green-600',
      hover: 'hover:bg-green-200'
    },
    yellow: {
      bg: 'bg-gradient-to-br from-yellow-50 to-yellow-100',
      title: 'text-yellow-700',
      value: 'text-yellow-900',
      subtitle: 'text-yellow-600',
      icon: 'text-yellow-600',
      hover: 'hover:bg-yellow-200'
    },
    indigo: {
      bg: 'bg-gradient-to-br from-indigo-50 to-indigo-100',
      title: 'text-indigo-700',
      value: 'text-indigo-900',
      subtitle: 'text-indigo-600',
      icon: 'text-indigo-600',
      hover: 'hover:bg-indigo-200'
    },
    pink: {
      bg: 'bg-gradient-to-br from-pink-50 to-pink-100',
      title: 'text-pink-700',
      value: 'text-pink-900',
      subtitle: 'text-pink-600',
      icon: 'text-pink-600',
      hover: 'hover:bg-pink-200'
    }
  };

  const classes = colorClasses[color] || colorClasses.blue;

  return (
    <div ref={cardRef} className={`${classes.bg} rounded-lg p-6 relative`}>
      <div className="flex items-center justify-between mb-2">
        <p className={`text-sm ${classes.title}`}>{title}</p>
        <button
          onClick={() => setShowTooltip(!showTooltip)}
          className={`p-1 rounded-full ${classes.hover} transition`}
          title="More info"
        >
          <Info size={14} className={classes.icon} />
        </button>
      </div>
      <p className={`text-3xl font-bold ${classes.value}`}>{value}</p>
      <p className={`text-xs ${classes.subtitle} mt-2`}>{subtitle}</p>

      {showTooltip && (
        <div className={`absolute z-10 ${showAbove ? 'bottom-full mb-2' : 'top-full mt-2'} left-0 right-0 bg-white border border-slate-200 rounded-lg shadow-lg p-4 text-sm`}>
          <p className="font-semibold text-slate-900 mb-2">Calculation:</p>
          <p className="text-slate-700 mb-3">{description}</p>
          <p className="font-semibold text-slate-900 mb-2">Interpretation:</p>
          <p className="text-slate-700">{interpretation}</p>
        </div>
      )}
    </div>
  );
}

export default MetricCard;
