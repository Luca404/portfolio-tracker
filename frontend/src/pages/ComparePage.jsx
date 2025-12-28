import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, BarChart3 } from 'lucide-react';
import { API_URL } from '../config';
import MetricCard from '../components/MetricCard';
import { getCurrencySymbol } from '../utils/helpers';

function ComparePage({ token, portfolio, portfolios, onSelectPortfolio }) {
  const [activeTab, setActiveTab] = useState('dcavslumpsum');
  const [compareData, setCompareData] = useState(null);
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [selectedBenchmark, setSelectedBenchmark] = useState('SPY');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const handlePortfolioChange = (e) => {
    const newPortfolioId = parseInt(e.target.value);
    const newPortfolio = portfolios.find(p => p.id === newPortfolioId);
    if (newPortfolio) {
      onSelectPortfolio(newPortfolio);
    }
  };

  useEffect(() => {
    if (!portfolio) {
      setLoading(false);
      return;
    }

    if (activeTab === 'dcavslumpsum') {
      fetchDCAvsLumpSum();
    } else if (activeTab === 'benchmarks') {
      fetchBenchmarkComparison();
    } else {
      setLoading(false);
    }
  }, [portfolio, token, activeTab, selectedBenchmark]);

  const fetchDCAvsLumpSum = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/portfolios/compare/${portfolio.id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Failed to fetch comparison data');
      }

      const data = await res.json();
      setCompareData(data);
    } catch (err) {
      console.error('[COMPARE] Error fetching data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchBenchmarkComparison = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/portfolios/compare-benchmark/${portfolio.id}?benchmark=${selectedBenchmark}`, {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Failed to fetch benchmark comparison data');
      }

      const data = await res.json();
      setBenchmarkData(data);
    } catch (err) {
      console.error('[BENCHMARK] Error fetching data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!portfolio) {
    return (
      <div className="bg-white rounded-lg shadow p-6 text-center py-16">
        <BarChart3 className="w-16 h-16 text-slate-300 mx-auto mb-4" />
        <p className="text-slate-600">Please select a portfolio to compare strategies.</p>
      </div>
    );
  }

  const currencySymbol = getCurrencySymbol(portfolio.reference_currency);
  const formatValue = (value) => `${currencySymbol}${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  const prepareBenchmarkChartData = () => {
    if (!benchmarkData) return [];

    // Create a map of dates to values for both portfolio and benchmark
    const dateMap = new Map();

    // Add portfolio timeline data
    benchmarkData.portfolio.timeline.forEach(entry => {
      if (!dateMap.has(entry.date)) {
        dateMap.set(entry.date, { date: entry.date });
      }
      dateMap.get(entry.date).portfolio = entry.value;
    });

    // Add benchmark timeline data
    benchmarkData.benchmark.timeline.forEach(entry => {
      if (!dateMap.has(entry.date)) {
        dateMap.set(entry.date, { date: entry.date });
      }
      dateMap.get(entry.date).benchmark = entry.value;
    });

    // Convert to array and sort by date
    const chartData = Array.from(dateMap.values()).sort((a, b) => {
      const [dayA, monthA, yearA] = a.date.split('-');
      const [dayB, monthB, yearB] = b.date.split('-');
      const dateA = new Date(yearA, monthA - 1, dayA);
      const dateB = new Date(yearB, monthB - 1, dayB);
      return dateA - dateB;
    });

    return chartData;
  };

  return (
    <div>
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
        <div className="flex-1">
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Compare Strategies</h1>
          <p className="text-slate-600">Compare your portfolio performance against different investment strategies and benchmarks</p>
        </div>
        {portfolios && portfolios.length > 1 && (
          <select
            value={portfolio.id}
            onChange={handlePortfolioChange}
            className="px-3 py-2 border border-slate-200 rounded-lg text-sm font-medium text-slate-700 bg-white hover:border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent cursor-pointer"
          >
            {portfolios.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <div className="flex flex-wrap gap-3">
          <button
            onClick={() => setActiveTab('dcavslumpsum')}
            className={`px-4 py-2 rounded-lg font-medium transition ${
              activeTab === 'dcavslumpsum' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            DCA vs Lump Sum
          </button>
          <button
            onClick={() => setActiveTab('benchmarks')}
            className={`px-4 py-2 rounded-lg font-medium transition ${
              activeTab === 'benchmarks' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            Portfolio vs Benchmark
          </button>
          <button
            onClick={() => setActiveTab('efficient')}
            className={`px-4 py-2 rounded-lg font-medium transition ${
              activeTab === 'efficient' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            Efficient Frontier
          </button>
          <button
            onClick={() => setActiveTab('allocation')}
            className={`px-4 py-2 rounded-lg font-medium transition ${
              activeTab === 'allocation' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            Model Portfolios
          </button>
        </div>
      </div>

      {/* DCA vs Lump Sum Tab */}
      {activeTab === 'dcavslumpsum' && (
        loading ? (
          <div className="bg-white rounded-lg shadow p-6 text-center py-16">
            <div className="animate-pulse">
              <div className="h-8 bg-slate-200 rounded w-1/3 mx-auto mb-4"></div>
              <div className="h-4 bg-slate-200 rounded w-1/2 mx-auto"></div>
            </div>
          </div>
        ) : error ? (
          <div className="bg-white rounded-lg shadow p-6 text-center py-16">
            <p className="text-red-600">{error}</p>
          </div>
        ) : compareData && (
          <div>
            <div className="bg-white rounded-lg shadow p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="w-6 h-6 text-blue-600" />
                <h2 className="text-2xl font-bold text-slate-900">DCA vs Lump Sum Analysis</h2>
              </div>
              <p className="text-slate-600 mb-6">
                Comparing your Dollar Cost Averaging strategy against a Lump Sum investment made on {compareData.first_order_date}
              </p>

              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <MetricCard
                  title="DCA Strategy"
                  value={formatValue(compareData.dca.final_value)}
                  subtitle={`Return: ${compareData.dca.return_pct >= 0 ? '+' : ''}${compareData.dca.return_pct.toFixed(2)}%`}
                  trend={compareData.dca.return_pct >= 0 ? 'up' : 'down'}
                  color="blue"
                  hideInfo={true}
                />
                <MetricCard
                  title="Lump Sum Strategy"
                  value={formatValue(compareData.lumpsum.final_value)}
                  subtitle={`Return: ${compareData.lumpsum.return_pct >= 0 ? '+' : ''}${compareData.lumpsum.return_pct.toFixed(2)}%`}
                  trend={compareData.lumpsum.return_pct >= 0 ? 'up' : 'down'}
                  color="purple"
                  hideInfo={true}
                />
                <MetricCard
                  title="Winner"
                  value={compareData.comparison.winner}
                  subtitle={`${compareData.comparison.difference_pct.toFixed(2)}% better`}
                  trend={compareData.comparison.difference_value >= 0 ? 'up' : 'down'}
                  color={compareData.comparison.winner === 'DCA' ? 'blue' : 'purple'}
                  hideInfo={true}
                />
              </div>

              {/* Investment Details */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-slate-50 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-slate-900 mb-4">DCA Strategy (Actual)</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-600">Total Invested</span>
                      <span className="font-semibold text-slate-900">{formatValue(compareData.dca.total_invested)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600">Current Value</span>
                      <span className="font-semibold text-slate-900">{formatValue(compareData.dca.final_value)}</span>
                    </div>
                    <div className="flex justify-between border-t border-slate-200 pt-3">
                      <span className="text-slate-600">Gain/Loss</span>
                      <span className={`font-semibold ${compareData.dca.gain_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {compareData.dca.gain_loss >= 0 ? '+' : ''}{formatValue(compareData.dca.gain_loss)}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-slate-50 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-slate-900 mb-4">Lump Sum Strategy (Hypothetical)</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-600">Total Invested</span>
                      <span className="font-semibold text-slate-900">{formatValue(compareData.lumpsum.total_invested)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600">Current Value</span>
                      <span className="font-semibold text-slate-900">{formatValue(compareData.lumpsum.final_value)}</span>
                    </div>
                    <div className="flex justify-between border-t border-slate-200 pt-3">
                      <span className="text-slate-600">Gain/Loss</span>
                      <span className={`font-semibold ${compareData.lumpsum.gain_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {compareData.lumpsum.gain_loss >= 0 ? '+' : ''}{formatValue(compareData.lumpsum.gain_loss)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Chart */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">Portfolio Value Over Time</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={compareData.dca.timeline.map((dcaEntry, index) => {
                    const lsEntry = compareData.lumpsum.timeline[index];
                    return {
                      date: dcaEntry.date,
                      dca: dcaEntry.value,
                      lumpsum: lsEntry.value
                    };
                  })}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="date"
                      stroke="#64748b"
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => {
                        // Parse DD-MM-YYYY format
                        const [day, month, year] = value.split('-');
                        const date = new Date(year, month - 1, day);
                        return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
                      }}
                    />
                    <YAxis
                      stroke="#64748b"
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => `${currencySymbol}${(value / 1000).toFixed(0)}k`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'white',
                        border: '1px solid #e2e8f0',
                        borderRadius: '0.5rem',
                        padding: '0.75rem'
                      }}
                      formatter={(value) => formatValue(value)}
                      labelFormatter={(label) => {
                        const [day, month, year] = label.split('-');
                        const date = new Date(year, month - 1, day);
                        return date.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="dca"
                      name="DCA Strategy"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="lumpsum"
                      name="Lump Sum"
                      stroke="#a855f7"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Explanation */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-blue-900 mb-2">What does this comparison show?</h3>
                <div className="text-blue-800 space-y-2">
                  <p>
                    <strong>DCA (Dollar Cost Averaging):</strong> Your actual investment strategy, where you bought assets over time as orders were placed.
                  </p>
                  <p>
                    <strong>Lump Sum:</strong> A hypothetical scenario where you invested all your money on {compareData.first_order_date} and bought the same quantities of assets you currently hold.
                  </p>
                  <p className="mt-4">
                    The comparison shows which strategy would have performed better given your specific portfolio composition and the market conditions during your investment period.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )
      )}

      {/* vs Benchmarks Tab */}
      {activeTab === 'benchmarks' && (
        loading ? (
          <div className="bg-white rounded-lg shadow p-6 text-center py-16">
            <div className="animate-pulse">
              <div className="h-8 bg-slate-200 rounded w-1/3 mx-auto mb-4"></div>
              <div className="h-4 bg-slate-200 rounded w-1/2 mx-auto"></div>
            </div>
          </div>
        ) : error ? (
          <div className="bg-white rounded-lg shadow p-6 text-center py-16">
            <p className="text-red-600">{error}</p>
          </div>
        ) : benchmarkData && (
          <div>
            <div className="bg-white rounded-lg shadow p-6 mb-6">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="w-6 h-6 text-green-600" />
                <h2 className="text-2xl font-bold text-slate-900">Portfolio vs Benchmark</h2>
              </div>
              <p className="text-slate-600 mb-6">
                Comparing your portfolio against {benchmarkData.benchmark_symbol} using the same DCA investment strategy
              </p>

              {/* Benchmark Selector */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-slate-700 mb-2">Select Benchmark</label>
                <select
                  value={selectedBenchmark}
                  onChange={(e) => setSelectedBenchmark(e.target.value)}
                  className="px-3 py-2 border border-slate-200 rounded-lg text-sm font-medium text-slate-700 bg-white hover:border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent cursor-pointer"
                >
                  <option value="SPY">S&P 500</option>
                  <option value="VWCE">FTSE All-World (VWCE)</option>
                </select>
              </div>

              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <MetricCard
                  title="Your Portfolio"
                  value={formatValue(benchmarkData.portfolio.final_value)}
                  subtitle={`Return: ${benchmarkData.portfolio.return_pct >= 0 ? '+' : ''}${benchmarkData.portfolio.return_pct.toFixed(2)}%`}
                  trend={benchmarkData.portfolio.return_pct >= 0 ? 'up' : 'down'}
                  color="blue"
                  hideInfo={true}
                />
                <MetricCard
                  title={`Benchmark (${benchmarkData.benchmark_symbol})`}
                  value={formatValue(benchmarkData.benchmark.final_value)}
                  subtitle={`Return: ${benchmarkData.benchmark.return_pct >= 0 ? '+' : ''}${benchmarkData.benchmark.return_pct.toFixed(2)}%`}
                  trend={benchmarkData.benchmark.return_pct >= 0 ? 'up' : 'down'}
                  color="green"
                  hideInfo={true}
                />
                <MetricCard
                  title="Winner"
                  value={benchmarkData.comparison.winner}
                  subtitle={`${benchmarkData.comparison.difference_pct.toFixed(2)}% better`}
                  trend={benchmarkData.comparison.winner === 'Portfolio' ? 'up' : 'down'}
                  color={benchmarkData.comparison.winner === 'Portfolio' ? 'blue' : 'green'}
                  hideInfo={true}
                />
              </div>

              {/* Investment Details */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div className="bg-slate-50 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-slate-900 mb-4">Your Portfolio</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-600">Total Invested</span>
                      <span className="font-semibold text-slate-900">{formatValue(benchmarkData.portfolio.total_invested)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600">Current Value</span>
                      <span className="font-semibold text-slate-900">{formatValue(benchmarkData.portfolio.final_value)}</span>
                    </div>
                    <div className="flex justify-between border-t border-slate-200 pt-3">
                      <span className="text-slate-600">Gain/Loss</span>
                      <span className={`font-semibold ${benchmarkData.portfolio.gain_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {benchmarkData.portfolio.gain_loss >= 0 ? '+' : ''}{formatValue(benchmarkData.portfolio.gain_loss)}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-slate-50 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-slate-900 mb-4">Benchmark ({benchmarkData.benchmark_symbol})</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-slate-600">Total Invested</span>
                      <span className="font-semibold text-slate-900">{formatValue(benchmarkData.benchmark.total_invested)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600">Current Value</span>
                      <span className="font-semibold text-slate-900">{formatValue(benchmarkData.benchmark.final_value)}</span>
                    </div>
                    <div className="flex justify-between border-t border-slate-200 pt-3">
                      <span className="text-slate-600">Gain/Loss</span>
                      <span className={`font-semibold ${benchmarkData.benchmark.gain_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {benchmarkData.benchmark.gain_loss >= 0 ? '+' : ''}{formatValue(benchmarkData.benchmark.gain_loss)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Performance Chart */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">Performance Over Time</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={prepareBenchmarkChartData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="date"
                      tickFormatter={(value) => {
                        const [day, month, year] = value.split('-');
                        const date = new Date(year, month - 1, day);
                        return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
                      }}
                    />
                    <YAxis />
                    <Tooltip
                      labelFormatter={(label) => {
                        const [day, month, year] = label.split('-');
                        const date = new Date(year, month - 1, day);
                        return date.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="portfolio"
                      name="Your Portfolio"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      connectNulls={true}
                    />
                    <Line
                      type="monotone"
                      dataKey="benchmark"
                      name={`Benchmark (${benchmarkData.benchmark_symbol})`}
                      stroke="#10b981"
                      strokeWidth={2}
                      strokeDasharray="0"
                      dot={false}
                      connectNulls={true}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Explanation */}
              <div className="bg-green-50 border border-green-200 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-green-900 mb-2">How does this work?</h3>
                <div className="text-green-800 space-y-2">
                  <p>
                    This comparison applies your exact DCA investment strategy to the benchmark. On each date you made a purchase,
                    the same amount is invested in the benchmark index.
                  </p>
                  <p className="mt-4">
                    This shows whether your portfolio selection outperformed the market benchmark, given the same investment timing and amounts.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )
      )}

      {/* Efficient Frontier Tab */}
      {activeTab === 'efficient' && (
        <div className="bg-white rounded-lg shadow p-6 text-center py-16">
          <BarChart3 className="w-16 h-16 text-slate-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-slate-900 mb-2">Coming Soon</h3>
          <p className="text-slate-600 max-w-md mx-auto">
            See how your portfolio compares to the efficient frontier - the optimal risk/return combinations for different asset allocations.
          </p>
        </div>
      )}

      {/* Model Portfolios Tab */}
      {activeTab === 'allocation' && (
        <div className="bg-white rounded-lg shadow p-6 text-center py-16">
          <BarChart3 className="w-16 h-16 text-slate-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-slate-900 mb-2">Coming Soon</h3>
          <p className="text-slate-600 max-w-md mx-auto">
            Compare your asset allocation against proven model portfolios like 60/40, All Weather, and Three-Fund Portfolio.
          </p>
        </div>
      )}
    </div>
  );
}

export default ComparePage;
