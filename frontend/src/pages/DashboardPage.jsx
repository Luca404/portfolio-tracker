import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { RefreshCw, Wallet, TrendingUp, TrendingDown, Layers, Percent } from 'lucide-react';
import { API_URL } from '../config';
import DashboardSkeleton from '../components/skeletons/DashboardSkeleton';
import { formatCurrencyValue, parseDateDMY, invalidatePortfolioCache } from '../utils/helpers';

function Dashboard({ token, portfolio, portfolios, onSelectPortfolio, onDeleted, refreshPortfolios }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
   const [actionLoading, setActionLoading] = useState(false);
  const [timeRange, setTimeRange] = useState('MAX');
  const [logScale, setLogScale] = useState(false);
  const [referenceCurrency, setReferenceCurrency] = useState('EUR');

  // Flag per evitare fetch multipli simultanei (React Strict Mode)
  const fetchingRef = React.useRef(false);

  useEffect(() => {
    let isMounted = true;

    const loadData = async () => {
      if (!isMounted) return;

      // Previeni fetch multipli simultanei
      if (fetchingRef.current) {
        console.log(`[CACHE] Portfolio ${portfolio.id}: fetch già in corso, skip`);
        return;
      }

      // Controlla cache sessionStorage (valida per sessione browser)
      const cacheKey = `portfolio_${portfolio.id}`;
      const cached = sessionStorage.getItem(cacheKey);
      console.log(`[CACHE] Portfolio ${portfolio.id}: cache present = ${!!cached}`);

      if (cached) {
        try {
          const cachedData = JSON.parse(cached);
          const cacheTime = cachedData._cacheTime || 0;
          const now = Date.now();
          const cacheAge = now - cacheTime;

          // Invalida cache se manca portfolio_xirr (nuova feature) o se la struttura è vecchia
          const hasXirr = cachedData.summary && cachedData.summary.portfolio_xirr !== undefined;
          const hasNewStructure = cachedData.portfolio && cachedData.summary && cachedData.history;

          // Cache valida per 1 giorno (dati EOD non cambiano durante giornata)
          if (cacheAge < 24 * 60 * 60 * 1000 && hasXirr && hasNewStructure) {
            const ageMinutes = Math.round(cacheAge / 60000);
            const ageHours = Math.round(ageMinutes / 60);
            const ageDisplay = ageHours > 0 ? `${ageHours}h fa` : `${ageMinutes}m fa`;
            console.log(`[CACHE] Portfolio ${portfolio.id}: usando cache (${ageDisplay})`);
            setData(cachedData);
            setLoading(false);
            return;
          } else if (!hasXirr) {
            console.log(`[CACHE] Portfolio ${portfolio.id}: cache invalidata (manca portfolio_xirr)`);
          } else if (!hasNewStructure) {
            console.log(`[CACHE] Portfolio ${portfolio.id}: cache invalidata (struttura vecchia)`);
          }
        } catch (e) {
          console.warn('[CACHE] Errore parsing cache:', e);
        }
      }

      // Cache miss o stale: fetch dal server
      fetchingRef.current = true;
      try {
        await fetchData();
      } finally {
        fetchingRef.current = false;
      }
    };

    loadData();

    return () => {
      isMounted = false;
    };
  }, [portfolio.id]); // Usa solo ID per evitare re-render non necessari

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/portfolios/${portfolio.id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const result = await res.json();

      // Aggiungi timestamp cache
      result._cacheTime = Date.now();

      // Salva in cache
      const cacheKey = `portfolio_${portfolio.id}`;
      try {
        sessionStorage.setItem(cacheKey, JSON.stringify(result));
        console.log(`[CACHE] Portfolio ${portfolio.id}: salvato in cache`);
      } catch (e) {
        console.warn('[CACHE] Errore salvataggio cache:', e);
      }

      setData(result);
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (data?.summary?.reference_currency) {
      setReferenceCurrency(data.summary.reference_currency);
    }
  }, [data]);

  const handleRefresh = async () => {
    // Invalida cache e ricarica
    invalidatePortfolioCache(portfolio.id);
    await fetchData();
  };

  const handleCurrencyChange = async (newCurrency) => {
    try {
      const res = await fetch(`${API_URL}/portfolios/${portfolio.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          name: data?.portfolio?.name || portfolio.name,
          description: data?.portfolio?.description || portfolio.description || '',
          reference_currency: newCurrency
        })
      });

      if (res.ok) {
        setReferenceCurrency(newCurrency);
        invalidatePortfolioCache(portfolio.id);
        await fetchData();
        await refreshPortfolios();
      }
    } catch (err) {
      console.error('Error updating currency:', err);
    }
  };

  const handleDelete = async () => {
    if (!portfolio?.id) return;
    setActionLoading(true);
    try {
      await fetch(`${API_URL}/portfolios/${portfolio.id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });
      // Invalida cache prima di refreshare
      invalidatePortfolioCache(portfolio.id);
      await refreshPortfolios();
      onDeleted();
    } catch (err) {
      console.error('Error deleting portfolio:', err);
    } finally {
      setActionLoading(false);
    }
  };

  const handlePortfolioChange = (e) => {
    const next = portfolios.find(p => p.id === Number(e.target.value));
    if (next) {
      onSelectPortfolio(next);
    }
  };

  const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];
  const pieData = data?.positions.map(p => ({ name: p.symbol, value: p.market_value })) || [];
  const typeMap = {};
  (data?.positions || []).forEach((p) => {
    const key = (p.instrument_type || 'Other').toUpperCase();
    typeMap[key] = (typeMap[key] || 0) + p.market_value;
  });
  const typeData = Object.entries(typeMap).map(([name, value]) => ({ name, value }));
  const summaryCurrency = data?.summary?.reference_currency || data?.positions?.[0]?.currency || 'EUR';
  const portfolioHistory = data?.history?.portfolio || [];
  const performanceHistory = data?.history?.performance || [];

  const filteredHistory = React.useMemo(() => {
    if (!portfolioHistory.length) return [];
    if (timeRange === 'MAX') return portfolioHistory;

    const now = new Date();
    const ranges = {
      '1W': 7,
      '1M': 30,
      '1Y': 365,
    };
    const days = ranges[timeRange] || 0;
    if (!days) return portfolioHistory;

    const cutoff = new Date(now);
    cutoff.setDate(cutoff.getDate() - days);
    return portfolioHistory.filter(p => {
      const d = parseDateDMY(p.date);
      return d && d >= cutoff;
    });
  }, [portfolioHistory, timeRange]);

  const chartHistory = React.useMemo(() => {
    if (!filteredHistory.length) return [];
    if (!logScale) return filteredHistory;
    // remove non-positive values to avoid log scale issues
    return filteredHistory.filter(p => p.value > 0);
  }, [filteredHistory, logScale]);

  const yAxisDomain = React.useMemo(() => {
    if (!chartHistory.length || !logScale) return ['auto', 'auto'];
    const positives = chartHistory.map(p => p.value).filter(v => v > 0);
    if (!positives.length) return ['auto', 'auto'];
    const minVal = Math.min(...positives);
    return [minVal, 'auto'];
  }, [chartHistory, logScale]);

  const filteredPerformance = React.useMemo(() => {
    if (!performanceHistory.length) return [];
    if (timeRange === 'MAX') return performanceHistory;

    const now = new Date();
    const ranges = {
      '1W': 7,
      '1M': 30,
      '1Y': 365,
    };
    const days = ranges[timeRange] || 0;
    if (!days) return performanceHistory;

    const cutoff = new Date(now);
    cutoff.setDate(cutoff.getDate() - days);
    return performanceHistory.filter(p => {
      const d = parseDateDMY(p.date);
      return d && d >= cutoff;
    });
  }, [performanceHistory, timeRange]);

  const perfPercentSeries = React.useMemo(() => {
    if (!filteredPerformance.length) return [];
    const base = filteredPerformance[0].value || 100;
    if (base === 0) return [];
    return filteredPerformance.map(p => ({
      date: p.date,
      value: ((p.value / base) - 1) * 100,
    }));
  }, [filteredPerformance]);

  if (loading) {
    return <DashboardSkeleton />;
  }

  return (
    <div>
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
        <div className="flex-1">
          <h1 className="text-3xl font-bold text-slate-900 mb-2">{data?.portfolio?.name || portfolio.name}</h1>
          <p className="text-slate-600">{data?.portfolio?.description || portfolio.description || 'Portfolio Overview'}</p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={referenceCurrency}
            onChange={(e) => handleCurrencyChange(e.target.value)}
            className="px-3 py-2 border border-slate-200 rounded-lg text-sm font-medium text-slate-700 bg-white hover:border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent cursor-pointer"
          >
            <option value="EUR">EUR (€)</option>
            <option value="USD">USD ($)</option>
            <option value="GBP">GBP (£)</option>
            <option value="CHF">CHF (Fr)</option>
            <option value="JPY">JPY (¥)</option>
            <option value="CNY">CNY (¥)</option>
          </select>
          <select
            value={portfolio.id}
            onChange={handlePortfolioChange}
            className="px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            {portfolios.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
          <button
            onClick={handleRefresh}
            disabled={loading}
            className="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition disabled:opacity-50 flex items-center gap-2"
            title="Aggiorna dati (ignora cache)"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
          <button
            onClick={handleDelete}
            disabled={actionLoading}
            className="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition disabled:opacity-50"
          >
            Delete
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {/* Total Value */}
        <div className="bg-white border border-slate-200 rounded-lg p-5 hover:border-slate-300 transition-colors">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-blue-50 rounded-lg">
              <Wallet className="w-5 h-5 text-blue-600" />
            </div>
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">Total Value</span>
          </div>
          <p className="text-2xl font-bold text-slate-900">{formatCurrencyValue(data?.summary.total_value, summaryCurrency)}</p>
        </div>

        {/* Total P&L */}
        <div className="bg-white border border-slate-200 rounded-lg p-5 hover:border-slate-300 transition-colors">
          <div className="flex items-center gap-3 mb-3">
            <div className={`p-2 rounded-lg ${data?.summary.total_gain_loss >= 0 ? 'bg-green-50' : 'bg-red-50'}`}>
              {data?.summary.total_gain_loss >= 0 ? (
                <TrendingUp className="w-5 h-5 text-green-600" />
              ) : (
                <TrendingDown className="w-5 h-5 text-red-600" />
              )}
            </div>
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">Profit & Loss</span>
          </div>
          <div className="flex items-baseline gap-2">
            <p className={`text-2xl font-bold ${data?.summary.total_gain_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatCurrencyValue(data?.summary.total_gain_loss, summaryCurrency)}
            </p>
            <span className={`text-sm font-semibold ${data?.summary.total_gain_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {data?.summary.total_gain_loss_pct >= 0 ? '+' : ''}{data?.summary.total_gain_loss_pct.toFixed(2)}%
            </span>
          </div>
        </div>

        {/* Annual Return (XIRR) */}
        <div className="bg-white border border-slate-200 rounded-lg p-5 hover:border-slate-300 transition-colors">
          <div className="flex items-center gap-3 mb-3">
            <div className={`p-2 rounded-lg ${(data?.summary.portfolio_xirr || 0) >= 0 ? 'bg-purple-50' : 'bg-orange-50'}`}>
              <Percent className={`w-5 h-5 ${(data?.summary.portfolio_xirr || 0) >= 0 ? 'text-purple-600' : 'text-orange-600'}`} />
            </div>
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">Annual Return</span>
          </div>
          <div className="flex items-baseline gap-2">
            <p className={`text-2xl font-bold ${(data?.summary.portfolio_xirr || 0) >= 0 ? 'text-purple-600' : 'text-orange-600'}`}>
              {data?.summary.portfolio_xirr !== null && data?.summary.portfolio_xirr !== undefined
                ? `${data.summary.portfolio_xirr >= 0 ? '+' : ''}${data.summary.portfolio_xirr.toFixed(2)}%`
                : 'N/A'}
            </p>
            <span className="text-xs text-slate-500">XIRR</span>
          </div>
        </div>

        {/* Positions */}
        <div className="bg-white border border-slate-200 rounded-lg p-5 hover:border-slate-300 transition-colors">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-slate-100 rounded-lg">
              <Layers className="w-5 h-5 text-slate-600" />
            </div>
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">Positions</span>
          </div>
          <p className="text-2xl font-bold text-slate-900">{data?.positions.length}</p>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-lg p-6 mb-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-6">
          <h2 className="text-lg font-semibold text-slate-900">Portfolio Performance</h2>
          <div className="flex flex-wrap items-center gap-2">
            {['1W', '1M', '1Y', 'MAX'].map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  timeRange === range
                    ? 'bg-slate-900 text-white'
                    : 'bg-slate-50 text-slate-600 hover:bg-slate-100'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
        </div>
        {perfPercentSeries.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={perfPercentSeries}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#64748b' }} stroke="#e2e8f0" />
              <YAxis tickFormatter={(v) => `${v.toFixed(1)}%`} tick={{ fontSize: 11, fill: '#64748b' }} stroke="#e2e8f0" />
              <Tooltip
                formatter={(value) => [`${Number(value).toFixed(2)}%`, 'Performance']}
                labelFormatter={(label) => label}
                contentStyle={{ border: '1px solid #e2e8f0', borderRadius: '8px', fontSize: '12px' }}
              />
              <Line type="monotone" dataKey="value" name="Performance" stroke="#10b981" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-slate-400 text-sm text-center py-12">Performance data not available</p>
        )}
      </div>

      <div className="bg-white border border-slate-200 rounded-lg p-6 mb-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-6">
          <h2 className="text-lg font-semibold text-slate-900">Portfolio Value</h2>
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => setLogScale(!logScale)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                logScale
                  ? 'bg-slate-900 text-white'
                  : 'bg-slate-50 text-slate-600 hover:bg-slate-100'
              }`}
              title="Toggle logarithmic scale"
            >
              {logScale ? 'Log' : 'Linear'}
            </button>
          </div>
        </div>
        {chartHistory.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={chartHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#64748b' }} stroke="#e2e8f0" />
              <YAxis
                tickFormatter={(v) => v.toLocaleString()}
                scale={logScale ? 'log' : 'linear'}
                allowDataOverflow={logScale}
                domain={yAxisDomain}
                type="number"
                tick={{ fontSize: 11, fill: '#64748b' }}
                stroke="#e2e8f0"
              />
              <Tooltip
                formatter={(value) => [formatCurrencyValue(value, summaryCurrency), 'Value']}
                contentStyle={{ border: '1px solid #e2e8f0', borderRadius: '8px', fontSize: '12px' }}
              />
              <Line type="monotone" dataKey="value" stroke="#2563eb" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-slate-400 text-sm text-center py-12">Historical data not available</p>
          )}
      </div>

      <div className="bg-white border border-slate-200 rounded-lg p-6 mb-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-5">Holdings</h2>
          <div className="inline-block w-full border border-slate-200 rounded-lg overflow-hidden">
            <table className="w-full text-[15px] leading-tight table-auto">
              <thead className="bg-slate-50 text-slate-600">
                <tr>
                  <th className="py-2 px-1.5 text-center whitespace-nowrap">Symbol</th>
                  <th className="py-2 px-1.5 text-center whitespace-nowrap">Qty</th>
                  <th className="py-2 px-1.5 text-center whitespace-nowrap">Avg Price</th>
                  <th className="py-2 px-1.5 text-center whitespace-nowrap">Curr Price</th>
                  <th className="py-2 px-1.5 text-center whitespace-nowrap">Init Value</th>
                  <th className="py-2 px-1.5 text-center whitespace-nowrap">Mkt Value</th>
                  <th className="py-2 px-1.5 text-center whitespace-nowrap">P/L</th>
                  <th className="py-2 px-1.5 text-center whitespace-nowrap">P/L %</th>
                  <th className="py-2 px-1.5 text-center whitespace-nowrap">XIRR</th>
                </tr>
              </thead>
              <tbody>
                {data?.positions.map((pos, idx) => (
                  <tr key={pos.symbol} className={idx % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                    <td className="py-2 px-1.5 text-center font-semibold text-slate-900 whitespace-nowrap">{pos.symbol}</td>
                    <td className="py-2 px-1.5 text-center text-slate-700 whitespace-nowrap">{pos.quantity}</td>
                    <td className="py-2 px-1.5 text-center text-slate-700 whitespace-nowrap">{formatCurrencyValue(pos.avg_price, pos.currency)}</td>
                    <td className="py-2 px-1.5 text-center text-slate-700 whitespace-nowrap">{formatCurrencyValue(pos.current_price, pos.currency)}</td>
                    <td className="py-2 px-1.5 text-center text-slate-700 whitespace-nowrap">{formatCurrencyValue(pos.cost_basis, pos.currency)}</td>
                    <td className="py-2 px-1.5 text-center text-slate-900 font-semibold whitespace-nowrap">{formatCurrencyValue(pos.market_value, pos.currency)}</td>
                    <td className={`py-2 px-1.5 text-center font-semibold whitespace-nowrap ${pos.gain_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {pos.gain_loss >= 0 ? '+' : ''}{formatCurrencyValue(pos.gain_loss, pos.currency)}
                    </td>
                    <td className={`py-2 px-1.5 text-center font-semibold whitespace-nowrap ${pos.gain_loss_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {pos.gain_loss_pct.toFixed(2)}%
                    </td>
                    <td className="py-2 px-1.5 text-center text-slate-700 whitespace-nowrap">
                      {pos.xirr ? `${pos.xirr.toFixed(2)}%` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold text-slate-900 mb-4">Allocation by Asset</h2>
          {pieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => {
                    const pct = data.summary.total_value ? (value / data.summary.total_value) * 100 : 0;
                    return pct >= 5 ? `${name}: ${pct.toFixed(1)}%` : '';
                  }}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => formatCurrencyValue(value, summaryCurrency)} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-center text-slate-500 py-16">No positions</p>
          )}
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold text-slate-900 mb-4">Allocation by Asset Type</h2>
          {typeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={typeData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => {
                    const pct = data.summary.total_value ? (value / data.summary.total_value) * 100 : 0;
                    return pct >= 5 ? `${name}: ${pct.toFixed(1)}%` : '';
                  }}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {typeData.map((entry, index) => (
                    <Cell key={`cell-type-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => formatCurrencyValue(value, summaryCurrency)} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-center text-slate-500 py-16">No positions</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
