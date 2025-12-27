import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Activity, TrendingUp, AlertTriangle } from 'lucide-react';
import { API_URL } from '../config';
import MetricCard from '../components/MetricCard';
import AnalysisTabSkeleton from '../components/skeletons/AnalysisTabSkeleton';
import { getCurrencySymbol } from '../utils/helpers';

function AnalyzePage({ token, portfolio, portfolios, onSelectPortfolio }) {
  const [analysisData, setAnalysisData] = useState({});
  const [loadingTab, setLoadingTab] = useState({});
  const [activeTab, setActiveTab] = useState('correlation');
  const [monteCarloYears, setMonteCarloYears] = useState(1);

  // Load data for a specific tab with localStorage cache
  const fetchTabData = async (tab, mcYears = monteCarloYears, forceRefresh = false) => {
    // Map frontend tab names to backend keys
    const tabKeyMap = {
      'attribution': 'performance_attribution',
      'risk': 'risk_metrics'
    };
    const backendKey = tabKeyMap[tab] || tab;

    // Include Monte Carlo years in cache key for montecarlo tab
    const cacheKey = tab === 'montecarlo'
      ? `analysis_${portfolio.id}_${backendKey}_${mcYears}y`
      : `analysis_${portfolio.id}_${backendKey}`;

    const cached = localStorage.getItem(cacheKey);

    if (cached && !forceRefresh) {
      try {
        const cachedData = JSON.parse(cached);
        const cacheTime = cachedData._cacheTime || 0;
        const now = Date.now();
        // Cache valid for 1 day
        if (now - cacheTime < 24 * 60 * 60 * 1000) {
          console.log(`[CACHE] Analysis ${tab}: using cache`);
          setAnalysisData(prev => ({ ...prev, [tab]: cachedData.data }));

          // If loading attribution tab, also load historical_prices if cached
          if (tab === 'attribution') {
            const historicalPricesCacheKey = `analysis_${portfolio.id}_historical_prices`;
            const historicalPricesCached = localStorage.getItem(historicalPricesCacheKey);
            if (historicalPricesCached) {
              try {
                const historicalPricesData = JSON.parse(historicalPricesCached);
                if (now - (historicalPricesData._cacheTime || 0) < 24 * 60 * 60 * 1000) {
                  setAnalysisData(prev => ({ ...prev, historical_prices: historicalPricesData.data }));
                }
              } catch (e) {
                console.warn('[CACHE] Error parsing historical_prices cache:', e);
              }
            }
          }

          return;
        }
      } catch (e) {
        console.warn('[CACHE] Error parsing cache:', e);
      }
    }

    // Fetch from server
    setLoadingTab(prev => ({ ...prev, [tab]: true }));
    try {
      const res = await fetch(`${API_URL}/portfolios/analysis/${portfolio.id}?monte_carlo_years=${mcYears}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();

      // Save individual tab data to localStorage
      const tabData = {
        correlation: data.correlation,
        montecarlo: data.montecarlo,
        risk_metrics: data.risk_metrics,
        drawdown: data.drawdown,
        performance_attribution: data.performance_attribution,
        historical_prices: data.historical_prices
      };

      for (const [key, value] of Object.entries(tabData)) {
        // Include years in cache key for montecarlo
        const tabCacheKey = key === 'montecarlo'
          ? `analysis_${portfolio.id}_${key}_${mcYears}y`
          : `analysis_${portfolio.id}_${key}`;
        localStorage.setItem(tabCacheKey, JSON.stringify({
          data: value,
          _cacheTime: Date.now()
        }));
      }

      // Map to frontend tab names
      const frontendTabData = {
        correlation: data.correlation,
        montecarlo: data.montecarlo,
        risk: data.risk_metrics,
        drawdown: data.drawdown,
        attribution: data.performance_attribution,
        historical_prices: data.historical_prices
      };

      setAnalysisData(frontendTabData);
    } catch (err) {
      console.error('Error fetching analysis:', err);
    } finally {
      setLoadingTab(prev => ({ ...prev, [tab]: false }));
    }
  };

  // Load cached data on mount to avoid unnecessary fetches
  useEffect(() => {
    const loadCachedData = () => {
      // Map backend keys to frontend tab names
      const backendToFrontend = {
        'correlation': 'correlation',
        'montecarlo': 'montecarlo',
        'risk_metrics': 'risk',
        'drawdown': 'drawdown',
        'performance_attribution': 'attribution'
      };

      const cachedData = {};
      const now = Date.now();

      for (const [backendKey, frontendKey] of Object.entries(backendToFrontend)) {
        const cacheKey = `analysis_${portfolio.id}_${backendKey}`;
        const cached = localStorage.getItem(cacheKey);

        if (cached) {
          try {
            const cachedObj = JSON.parse(cached);
            const cacheTime = cachedObj._cacheTime || 0;
            // Cache valid for 1 day
            if (now - cacheTime < 24 * 60 * 60 * 1000) {
              cachedData[frontendKey] = cachedObj.data;
            }
          } catch (e) {
            console.warn(`[CACHE] Error parsing ${backendKey} cache:`, e);
          }
        }
      }

      if (Object.keys(cachedData).length > 0) {
        console.log(`[CACHE] Loaded ${Object.keys(cachedData).length} tabs from cache for portfolio ${portfolio.id}`);
        setAnalysisData(cachedData);
      }
    };

    loadCachedData();
  }, [portfolio.id]);

  // Fetch data when tab changes (only if not already loaded)
  useEffect(() => {
    if (!analysisData[activeTab]) {
      fetchTabData(activeTab);
    }
  }, [activeTab, portfolio.id]);

  const handlePortfolioChange = (e) => {
    const nextPortfolio = portfolios.find(p => p.id === Number(e.target.value));
    if (nextPortfolio) {
      setAnalysisData({});  // Clear data when switching portfolio
      onSelectPortfolio(nextPortfolio);
    }
  };

  const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

  return (
    <div>
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
        <div className="flex-1">
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Portfolio Analysis</h1>
          <p className="text-slate-600">Advanced analytics and risk metrics for {portfolio.name}</p>
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
      <div className="bg-white rounded-lg shadow mb-6 p-4">
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => setActiveTab('correlation')}
            className={`px-4 py-2 rounded-lg font-medium transition ${
              activeTab === 'correlation' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            Correlation Heatmap
          </button>
          <button
            onClick={() => setActiveTab('montecarlo')}
            className={`px-4 py-2 rounded-lg font-medium transition ${
              activeTab === 'montecarlo' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            Monte Carlo
          </button>
          <button
            onClick={() => setActiveTab('risk')}
            className={`px-4 py-2 rounded-lg font-medium transition ${
              activeTab === 'risk' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            Risk Metrics
          </button>
          <button
            onClick={() => setActiveTab('drawdown')}
            className={`px-4 py-2 rounded-lg font-medium transition ${
              activeTab === 'drawdown' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            Drawdown
          </button>
          <button
            onClick={() => setActiveTab('attribution')}
            className={`px-4 py-2 rounded-lg font-medium transition ${
              activeTab === 'attribution' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            Performance Attribution
          </button>
        </div>
      </div>

      {/* Correlation Heatmap */}
      {activeTab === 'correlation' && (
        loadingTab.correlation ? (
          <AnalysisTabSkeleton />
        ) : (
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center gap-2 mb-6">
              <Activity className="w-6 h-6 text-blue-600" />
              <h2 className="text-2xl font-bold text-slate-900">Asset Correlation Matrix</h2>
            </div>
            <p className="text-slate-600 mb-6">
              Correlation coefficients between assets. Values close to 1 indicate strong positive correlation,
              close to -1 indicate strong negative correlation, and near 0 indicate no correlation.
            </p>

            {analysisData.correlation ? (
              <CorrelationHeatmap data={analysisData.correlation} />
            ) : (
              <div className="text-center py-16 text-slate-500">
                Not enough data for correlation analysis. Need at least 2 assets with historical data.
              </div>
            )}
          </div>
        )
      )}

      {/* Monte Carlo Simulation */}
      {activeTab === 'montecarlo' && (
        loadingTab.montecarlo ? (
          <AnalysisTabSkeleton />
        ) : (
          <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-green-600" />
              <h2 className="text-2xl font-bold text-slate-900">Monte Carlo Simulation</h2>
            </div>
            {/* Years Selector */}
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium text-slate-700">Projection Period:</label>
              <select
                value={monteCarloYears}
                onChange={(e) => {
                  const newYears = parseInt(e.target.value);
                  setMonteCarloYears(newYears);
                  fetchTabData('montecarlo', newYears);
                }}
                className="px-3 py-1.5 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value={1}>1 Year</option>
                <option value={3}>3 Years</option>
                <option value={5}>5 Years</option>
                <option value={10}>10 Years</option>
              </select>
            </div>
          </div>
          <p className="text-slate-600 mb-6">
            10,000 simulated portfolio trajectories based on historical returns and volatility.
          </p>

          {analysisData.montecarlo ? (
            <MonteCarloChart
              data={analysisData.montecarlo}
            />
          ) : (
            <div className="text-center py-16 text-slate-500">
              Simulation data not available. Portfolio needs sufficient historical data.
            </div>
          )}
        </div>
        )
      )}

      {/* Risk Metrics */}
      {activeTab === 'risk' && (
        (loadingTab.risk || !analysisData?.risk) ? (
          <AnalysisTabSkeleton />
        ) : (
          <div className="space-y-6">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center gap-2 mb-6">
              <AlertTriangle className="w-6 h-6 text-orange-600" />
              <h2 className="text-2xl font-bold text-slate-900">Risk Metrics</h2>
            </div>

            {analysisData?.risk ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard
                  title="Sharpe Ratio"
                  value={analysisData.risk.sharpe_ratio?.toFixed(2) || 'N/A'}
                  subtitle="Risk-adjusted return"
                  color="blue"
                  description="(Return - Risk Free Rate) / Volatility. Assumes 2% risk-free rate."
                  interpretation="Higher is better. <0 = underperforming risk-free rate, 0-0.5 = subpar, 0.5-1 = acceptable, 1-2 = good, >2 = very good (rare). S&P 500 historically ~0.7-1."
                />

                <MetricCard
                  title="Sortino Ratio"
                  value={analysisData.risk.sortino_ratio?.toFixed(2) || 'N/A'}
                  subtitle="Downside risk-adjusted"
                  color="purple"
                  description="(Return - Risk Free Rate) / Downside Deviation. Only considers downside volatility."
                  interpretation="Higher is better. Usually higher than Sharpe. 1-1.5 = acceptable, 1.5-2 = good, >2 = very good. Only penalizes downside risk, not upside volatility."
                />

                <MetricCard
                  title="VaR (95%)"
                  value={analysisData.risk.var_95 ? `${analysisData.risk.var_95.toFixed(2)}%` : 'N/A'}
                  subtitle="Value at Risk"
                  color="orange"
                  description="5th percentile of daily returns. The worst daily loss you can expect 95% of the time."
                  interpretation="Negative value. -1% = conservative, -2% = moderate risk, -3% = high risk. Example: -2.5% means 95% of days you won't lose more than 2.5%."
                />

                <MetricCard
                  title="CVaR (95%)"
                  value={analysisData.risk.cvar_95 ? `${analysisData.risk.cvar_95.toFixed(2)}%` : 'N/A'}
                  subtitle="Conditional VaR"
                  color="red"
                  description="Average of returns below VaR threshold. Expected loss when things go bad (worst 5% of cases)."
                  interpretation="Negative value, worse than VaR. Shows average loss during bad days (worst 5%). More informative for extreme events. -3% to -5% typical for diversified portfolios."
                />

                <MetricCard
                  title="Calmar Ratio"
                  value={analysisData.risk.calmar_ratio ? analysisData.risk.calmar_ratio.toFixed(2) : 'N/A'}
                  subtitle="Return/Max Drawdown"
                  color="green"
                  description="Annual Return (XIRR) / Absolute Max Drawdown. Measures return relative to worst historical loss."
                  interpretation="Higher is better. 0.5-1 = acceptable, 1-2 = good, >2 = excellent (rare). Shows if returns justify maximum pain. Hedge funds target >0.5."
                  showAbove={true}
                />

                <MetricCard
                  title="Volatility"
                  value={analysisData.risk.volatility ? `${analysisData.risk.volatility.toFixed(2)}%` : 'N/A'}
                  subtitle="Annual volatility"
                  color="yellow"
                  description="Annualized standard deviation of daily returns (× √252)."
                  interpretation="Lower is less risky. 10-15% is low, 15-25% is moderate, >25% is high. Measures price fluctuation magnitude."
                  showAbove={true}
                />

                <MetricCard
                  title="Max Drawdown"
                  value={analysisData.risk.max_drawdown ? `${analysisData.risk.max_drawdown.toFixed(2)}%` : 'N/A'}
                  subtitle="Largest peak-to-trough"
                  color="indigo"
                  description="Largest percentage decline from peak to trough in portfolio value history."
                  interpretation="Negative value. E.g., -20% means the worst historical drop was 20%. Important for understanding pain tolerance needed."
                  showAbove={true}
                />

                <MetricCard
                  title={`Beta (vs ${analysisData.risk.beta_vs || 'Market'})`}
                  value={analysisData.risk.beta?.toFixed(2) || 'N/A'}
                  subtitle="Market sensitivity"
                  color="pink"
                  description={`Covariance with ${analysisData.risk.beta_vs || 'Market'} / Variance of ${analysisData.risk.beta_vs || 'Market'}. Measures correlation and sensitivity to market movements.`}
                  interpretation="1 = moves with market, >1 = more volatile than market, <1 = less volatile, <0 = inverse correlation. Helps assess systematic risk."
                  showAbove={true}
                />
              </div>
            ) : (
              <div className="text-center py-16 text-slate-500">
                Risk metrics not available. Portfolio needs sufficient historical data.
              </div>
            )}
          </div>
        </div>
        )
      )}

      {/* Drawdown Analysis */}
      {activeTab === 'drawdown' && (
        loadingTab.drawdown ? (
          <AnalysisTabSkeleton />
        ) : (
          <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center gap-2 mb-6">
            <TrendingUp className="w-6 h-6 text-red-600" />
            <h2 className="text-2xl font-bold text-slate-900">Drawdown Analysis</h2>
          </div>
          <p className="text-slate-600 mb-6">
            Drawdown represents the decline from a historical peak. Periods below 0% indicate losses from previous highs.
          </p>

          {analysisData.drawdown ? (
            <DrawdownChart data={analysisData.drawdown} />
          ) : (
            <div className="text-center py-16 text-slate-500">
              Drawdown data not available. Portfolio needs sufficient historical data.
            </div>
          )}
        </div>
        )
      )}

      {/* Performance Attribution */}
      {activeTab === 'attribution' && (
        (loadingTab.attribution || !analysisData?.attribution) ? (
          <AnalysisTabSkeleton />
        ) : (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center gap-2 mb-6">
                <TrendingUp className="w-6 h-6 text-green-600" />
                <h2 className="text-2xl font-bold text-slate-900">Performance Attribution</h2>
              </div>
              <p className="text-slate-600 mb-6">
                Contribution of each asset to portfolio returns. Understand which investments are driving performance.
              </p>

              {analysisData.attribution ? (
                <div className="space-y-6">
                  {/* Summary Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4">
                      <p className="text-sm text-blue-700 mb-1">Portfolio Total Return</p>
                      <p className={`text-2xl font-bold ${analysisData.attribution.portfolio_total_return >= 0 ? 'text-green-700' : 'text-red-700'}`}>
                        {analysisData.attribution.portfolio_total_return >= 0 ? '+' : ''}{analysisData.attribution.portfolio_total_return?.toFixed(2)}%
                      </p>
                    </div>
                    <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-4">
                      <p className="text-sm text-green-700 mb-1">Top Contributor</p>
                      <p className="text-lg font-bold text-green-900">
                        {analysisData.attribution.top_contributors?.[0]?.symbol || 'N/A'}
                      </p>
                      <p className="text-sm text-green-600">
                        {analysisData.attribution.top_contributors?.[0]?.contribution_to_portfolio >= 0 ? '+' : ''}
                        {analysisData.attribution.top_contributors?.[0]?.contribution_to_portfolio?.toFixed(2)}%
                      </p>
                    </div>
                    <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-lg p-4">
                      <p className="text-sm text-red-700 mb-1">Bottom Contributor</p>
                      <p className="text-lg font-bold text-red-900">
                        {analysisData.attribution.worst_contributors?.[0]?.symbol || 'N/A'}
                      </p>
                      <p className="text-sm text-red-600">
                        {analysisData.attribution.worst_contributors?.[0]?.contribution_to_portfolio >= 0 ? '+' : ''}
                        {analysisData.attribution.worst_contributors?.[0]?.contribution_to_portfolio?.toFixed(2)}%
                      </p>
                    </div>
                  </div>

                  {/* Contribution Chart */}
                  <div className="bg-slate-50 rounded-lg p-4">
                    <h3 className="text-lg font-semibold text-slate-900 mb-4">Contribution to Portfolio Return</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={analysisData.attribution.assets}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis dataKey="symbol" tick={{ fontSize: 12 }} />
                        <YAxis tick={{ fontSize: 12 }} label={{ value: 'Contribution (%)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: 'white', border: '1px solid #e2e8f0' }}
                          formatter={(value) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`}
                        />
                        <Bar dataKey="contribution_to_portfolio" fill="#3b82f6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Asset Performance Over Time */}
                  {analysisData.historical_prices && analysisData.historical_prices.dates && (
                    <AssetPerformanceChart
                      data={(() => {
                        // Transform historical_prices to chart format
                        const { dates, assets } = analysisData.historical_prices;
                        return dates.map((date, idx) => {
                          const dataPoint = { date };
                          Object.keys(assets).forEach(symbol => {
                            dataPoint[symbol] = assets[symbol][idx];
                          });
                          return dataPoint;
                        });
                      })()}
                      symbols={Object.keys(analysisData.historical_prices.assets || {})}
                    />
                  )}

                  {/* Detailed Table */}
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr className="bg-slate-50">
                          <th className="p-3 text-left text-sm font-semibold text-slate-700 border border-slate-200">Symbol</th>
                          <th className="p-3 text-right text-sm font-semibold text-slate-700 border border-slate-200">Weight</th>
                          <th className="p-3 text-right text-sm font-semibold text-slate-700 border border-slate-200">Value</th>
                          <th className="p-3 text-right text-sm font-semibold text-slate-700 border border-slate-200">Total Return</th>
                          <th className="p-3 text-right text-sm font-semibold text-slate-700 border border-slate-200">Contribution</th>
                          <th className="p-3 text-right text-sm font-semibold text-slate-700 border border-slate-200">Gain/Loss</th>
                        </tr>
                      </thead>
                      <tbody>
                        {analysisData.attribution.assets.map((asset) => (
                          <tr key={asset.symbol} className="hover:bg-slate-50">
                            <td className="p-3 text-sm font-medium text-slate-900 border border-slate-200">{asset.symbol}</td>
                            <td className="p-3 text-sm text-right text-slate-700 border border-slate-200">{asset.weight.toFixed(2)}%</td>
                            <td className="p-3 text-sm text-right text-slate-700 border border-slate-200">
                              {getCurrencySymbol('EUR')}{asset.current_value.toLocaleString()}
                            </td>
                            <td className={`p-3 text-sm text-right font-semibold border border-slate-200 ${asset.total_return_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {asset.total_return_pct >= 0 ? '+' : ''}{asset.total_return_pct.toFixed(2)}%
                            </td>
                            <td className={`p-3 text-sm text-right font-semibold border border-slate-200 ${asset.contribution_to_portfolio >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {asset.contribution_to_portfolio >= 0 ? '+' : ''}{asset.contribution_to_portfolio.toFixed(2)}%
                            </td>
                            <td className={`p-3 text-sm text-right border border-slate-200 ${asset.gain_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {getCurrencySymbol('EUR')}{asset.gain_loss.toLocaleString()}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div className="text-center py-16 text-slate-500">
                  Performance attribution data not available. Portfolio needs positions with cost basis.
                </div>
              )}
            </div>
          </div>
        )
      )}
    </div>
  );
}

// Helper component for Correlation Heatmap
function CorrelationHeatmap({ data }) {
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

// Helper component for Monte Carlo Chart
function MonteCarloChart({ data }) {
  if (!data || !data.percentiles || !data.dates) return null;

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

// Helper component for Drawdown Chart
function DrawdownChart({ data }) {
  if (!data) return null;

  // Handle both array of objects [{date, drawdown}] and separate arrays {dates: [], drawdown: []}
  // Backend returns decimal values (e.g., -0.1234 for -12.34%), multiply by 100 for percentage display
  const chartData = Array.isArray(data)
    ? data.map(item => ({
        date: item.date,
        drawdown: (item.drawdown || 0) * 100  // Convert decimal to percentage
      }))
    : data.dates?.map((date, i) => ({
        date,
        drawdown: (data.drawdown[i] || 0) * 100
      }));

  if (!chartData || chartData.length === 0) return null;

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

// Asset Performance Over Time Chart with selectable legend
function AssetPerformanceChart({ data, symbols }) {
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

export default AnalyzePage;
