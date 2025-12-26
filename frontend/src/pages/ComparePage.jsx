import React from 'react';
import { BarChart3 } from 'lucide-react';

function ComparePage({ token, portfolio }) {
  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Compare Portfolios</h1>
        <p className="text-slate-600">Compare your portfolio against benchmarks and model portfolios</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6 text-center py-16">
        <BarChart3 className="w-16 h-16 text-slate-300 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-slate-900 mb-2">Coming Soon</h3>
        <p className="text-slate-600">
          Portfolio comparison features will be available here soon.
        </p>
      </div>
    </div>
  );
}

export default ComparePage;
