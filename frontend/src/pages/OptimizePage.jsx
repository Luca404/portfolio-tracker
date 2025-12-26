import React, { useState } from 'react';
import { API_URL } from '../config';

function OptimizePage({ token, portfolio }) {
  const [formData, setFormData] = useState({
    symbols: '',
    optimization_type: 'max_sharpe'
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleOptimize = async () => {
    setLoading(true);
    try {
      const symbols = formData.symbols.split(',').map(s => s.trim());
      const res = await fetch(`${API_URL}/orders/optimize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          portfolio_id: portfolio.id,
          symbols,
          optimization_type: formData.optimization_type
        })
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Portfolio Optimization</h1>
        <p className="text-slate-600">Optimize your portfolio allocation using Modern Portfolio Theory</p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h2 className="text-xl font-bold mb-4">Optimization Parameters</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Symbols (comma-separated)
            </label>
            <input
              type="text"
              value={formData.symbols}
              onChange={(e) => setFormData({...formData, symbols: e.target.value})}
              className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              placeholder="AAPL, MSFT, GOOGL, AMZN"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Optimization Strategy
            </label>
            <select
              value={formData.optimization_type}
              onChange={(e) => setFormData({...formData, optimization_type: e.target.value})}
              className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="max_sharpe">Maximum Sharpe Ratio</option>
              <option value="min_volatility">Minimum Volatility</option>
              <option value="efficient_risk">Efficient Risk</option>
            </select>
          </div>

          <button
            onClick={handleOptimize}
            disabled={loading}
            className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition disabled:opacity-50"
          >
            {loading ? 'Optimizing...' : 'Run Optimization'}
          </button>
        </div>
      </div>

      {result && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold mb-6">Optimization Results</h2>

          <div className="grid grid-cols-3 gap-6 mb-8">
            <div className="text-center p-6 bg-green-50 rounded-lg">
              <p className="text-sm text-green-700 mb-2">Expected Return</p>
              <p className="text-3xl font-bold text-green-600">{result.expected_return}%</p>
            </div>
            <div className="text-center p-6 bg-orange-50 rounded-lg">
              <p className="text-sm text-orange-700 mb-2">Volatility</p>
              <p className="text-3xl font-bold text-orange-600">{result.volatility}%</p>
            </div>
            <div className="text-center p-6 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-700 mb-2">Sharpe Ratio</p>
              <p className="text-3xl font-bold text-blue-600">{result.sharpe_ratio}</p>
            </div>
          </div>

          <div className="mb-8">
            <h3 className="font-bold text-lg mb-4">Recommended Weights</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(result.weights).map(([symbol, weight]) => (
                <div key={symbol} className="flex justify-between bg-slate-50 p-4 rounded-lg">
                  <span className="font-semibold text-slate-900">{symbol}</span>
                  <span className="text-blue-600 font-bold">{(weight * 100).toFixed(2)}%</span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="font-bold text-lg mb-4">Shares to Buy</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(result.allocation).map(([symbol, qty]) => (
                <div key={symbol} className="flex justify-between bg-slate-50 p-4 rounded-lg">
                  <span className="font-semibold text-slate-900">{symbol}</span>
                  <span className="text-slate-700 font-bold">{qty} shares</span>
                </div>
              ))}
            </div>
            <p className="text-sm text-slate-600 mt-4">
              Remaining cash: <span className="font-semibold">${result.leftover.toFixed(2)}</span>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default OptimizePage;
