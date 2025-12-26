import React, { useState, useEffect } from 'react';
import { Plus, Edit2, Trash2, Check, X } from 'lucide-react';
import { getCurrencySymbol } from '../utils/currency';
import { API_URL } from '../config';
import PortfolioCardSkeleton from './skeletons/PortfolioCardSkeleton';

// Portfolios List
function PortfoliosList({ token, onSelectPortfolio, portfolios, onRefresh, loading }) {
  const [showCreate, setShowCreate] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [portfolioCount, setPortfolioCount] = useState(0);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    reference_currency: 'EUR'
  });
  const [editData, setEditData] = useState({
    name: '',
    description: '',
    reference_currency: 'EUR',
    risk_free_source: 'auto',
    market_benchmark: 'auto'
  });

  // Initialize count from cache immediately
  const [countInitialized, setCountInitialized] = useState(false);

  useEffect(() => {
    // Set count from cache immediately (synchronous)
    const cacheKey = 'portfolio_count';
    const cached = sessionStorage.getItem(cacheKey);
    if (cached) {
      setPortfolioCount(Number(cached));
    }
    setCountInitialized(true);
  }, []);

  // Fetch portfolio count for skeleton loading
  useEffect(() => {
    if (!countInitialized) return;

    const fetchCount = async () => {
      try {
        const res = await fetch(`${API_URL}/portfolios/count`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        const data = await res.json();
        setPortfolioCount(data.count);
        sessionStorage.setItem('portfolio_count', data.count.toString());
      } catch (err) {
        console.error('Error fetching portfolio count:', err);
      }
    };

    fetchCount();
  }, [token, countInitialized]);

  const handleCreate = async () => {
    try {
      const res = await fetch(`${API_URL}/portfolios`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(formData)
      });

      if (res.ok) {
        setShowCreate(false);
        setFormData({ name: '', description: '', reference_currency: 'EUR' });
        // Update count cache
        const newCount = portfolioCount + 1;
        setPortfolioCount(newCount);
        sessionStorage.setItem('portfolio_count', newCount.toString());
        onRefresh();
      }
    } catch (err) {
      console.error('Error creating portfolio:', err);
    }
  };

  const handleUpdate = async (id) => {
    try {
      const res = await fetch(`${API_URL}/portfolios/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(editData)
      });

      if (res.ok) {
        setEditingId(null);
        onRefresh();
      }
    } catch (err) {
      console.error('Error updating portfolio:', err);
    }
  };

  const handleDelete = async (id) => {
    try {
      const res = await fetch(`${API_URL}/portfolios/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });

      if (res.ok) {
        setDeleteTarget(null);
        // Update count cache
        const newCount = Math.max(0, portfolioCount - 1);
        setPortfolioCount(newCount);
        sessionStorage.setItem('portfolio_count', newCount.toString());
        onRefresh();
      }
    } catch (err) {
      console.error('Error deleting portfolio:', err);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 mb-2">My Portfolios</h1>
          <p className="text-slate-600">Manage your investment portfolios</p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition"
        >
          <Plus className="w-5 h-5" />
          New Portfolio
        </button>
      </div>

      {showCreate && (
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6 border border-blue-200">
          <h2 className="text-xl font-bold text-slate-900 mb-4">Create New Portfolio</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Portfolio Name</label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({...formData, name: e.target.value})}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="My Growth Portfolio"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Description</label>
              <input
                type="text"
                value={formData.description}
                onChange={(e) => setFormData({...formData, description: e.target.value})}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Long-term growth investments"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Reference Currency</label>
              <select
                value={formData.reference_currency}
                onChange={(e) => setFormData({...formData, reference_currency: e.target.value})}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="EUR">EUR (€)</option>
                <option value="USD">USD ($)</option>
                <option value="GBP">GBP (£)</option>
                <option value="CHF">CHF (Fr)</option>
                <option value="JPY">JPY (¥)</option>
                <option value="CNY">CNY (¥)</option>
              </select>
            </div>
            <div className="flex gap-3">
              <button
                onClick={handleCreate}
                className="flex-1 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition"
              >
                Create Portfolio
              </button>
              <button
                onClick={() => setShowCreate(false)}
                className="flex-1 bg-slate-200 text-slate-700 py-2 rounded-lg hover:bg-slate-300 transition"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {loading && portfolios.length === 0 ? (
          // Show skeleton cards based on actual count
          Array.from({ length: portfolioCount || 3 }).map((_, i) => <PortfolioCardSkeleton key={i} />)
        ) : (
          portfolios.length === 0 ? (
            <div className="col-span-full text-center py-16 text-slate-500">
              <p className="text-lg mb-2">No portfolios yet</p>
              <p className="text-sm">Create your first portfolio to get started</p>
            </div>
          ) : portfolios.map((portfolio) => (
          <div
            key={portfolio.id}
            className="bg-white rounded-lg shadow hover:shadow-lg transition p-6 border-2 border-transparent hover:border-blue-500 relative group"
          >
            {editingId === portfolio.id ? null : (
              <>
                <div onClick={() => onSelectPortfolio(portfolio)} className="cursor-pointer">
                  <h3 className="text-xl font-bold text-slate-900 mb-2">{portfolio.name}</h3>
                  <p className="text-slate-600 text-sm mb-4">{portfolio.description || 'No description'}</p>
                  <div className="space-y-2 pt-4 border-t border-slate-200">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-600">Portfolio Value</span>
                      <span className="text-lg font-bold text-blue-600">
                        {getCurrencySymbol(portfolio.reference_currency || 'EUR')}{(portfolio.total_value || 0).toLocaleString()}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-600">P&L</span>
                      <span className={`text-sm font-semibold ${(portfolio.total_gain_loss || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {getCurrencySymbol(portfolio.reference_currency || 'EUR')}{(portfolio.total_gain_loss || 0).toLocaleString()} ({(portfolio.total_gain_loss_pct || 0) >= 0 ? '+' : ''}{(portfolio.total_gain_loss_pct || 0).toFixed(2)}%)
                      </span>
                    </div>
                  </div>
                </div>
                <div className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setEditData({
                        name: portfolio.name,
                        description: portfolio.description || '',
                        reference_currency: portfolio.reference_currency || 'EUR',
                        risk_free_source: portfolio.risk_free_source || 'auto',
                        market_benchmark: portfolio.market_benchmark || 'auto'
                      });
                      setEditingId(portfolio.id);
                    }}
                    className="p-2 bg-blue-100 text-blue-600 rounded-lg hover:bg-blue-200 transition"
                    title="Edit portfolio"
                  >
                    <Edit2 size={16} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setDeleteTarget(portfolio);
                    }}
                    className="p-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 transition"
                    title="Delete portfolio"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </>
            )}
          </div>
          ))
        )}
      </div>

      {/* Edit Portfolio Modal */}
      {editingId && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <h3 className="text-2xl font-bold text-slate-900 mb-6">Edit Portfolio Settings</h3>

              <div className="space-y-5">
                {/* Basic Info */}
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Portfolio Name *
                  </label>
                  <input
                    type="text"
                    value={editData.name}
                    onChange={(e) => setEditData({...editData, name: e.target.value})}
                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="My Investment Portfolio"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Description
                  </label>
                  <textarea
                    value={editData.description}
                    onChange={(e) => setEditData({...editData, description: e.target.value})}
                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Long-term growth portfolio..."
                    rows="2"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">
                    Reference Currency *
                  </label>
                  <select
                    value={editData.reference_currency}
                    onChange={(e) => setEditData({...editData, reference_currency: e.target.value})}
                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="EUR">EUR (€)</option>
                    <option value="USD">USD ($)</option>
                    <option value="GBP">GBP (£)</option>
                    <option value="CHF">CHF (Fr)</option>
                    <option value="JPY">JPY (¥)</option>
                    <option value="CNY">CNY (¥)</option>
                  </select>
                </div>

                {/* Divider */}
                <div className="border-t border-slate-200 my-4"></div>

                {/* Advanced Settings */}
                <div className="bg-slate-50 p-4 rounded-lg space-y-4">
                  <h4 className="font-semibold text-slate-900 mb-3">Advanced Financial Settings</h4>

                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      Risk-Free Rate Source
                    </label>
                    <select
                      value={editData.risk_free_source}
                      onChange={(e) => setEditData({...editData, risk_free_source: e.target.value})}
                      className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                    >
                      <option value="auto">Auto (based on currency)</option>
                      <option value="USD_TREASURY">US Treasury 10Y</option>
                      <option value="EUR_ECB">ECB Rate</option>
                      <option value="0">0% (No risk-free)</option>
                      <option value="2">2% (Custom)</option>
                      <option value="3">3% (Custom)</option>
                      <option value="4">4% (Custom)</option>
                      <option value="5">5% (Custom)</option>
                    </select>
                    <p className="text-xs text-slate-500 mt-1">
                      Used for Sharpe ratio and Sortino ratio calculations
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                      Market Benchmark
                    </label>
                    <select
                      value={editData.market_benchmark}
                      onChange={(e) => setEditData({...editData, market_benchmark: e.target.value})}
                      className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                    >
                      <option value="auto">Auto (based on currency)</option>
                      <option value="SP500">S&P 500</option>
                      <option value="VWCE">VWCE (All-World)</option>
                      <option value="QQQ">Nasdaq 100 (QQQ)</option>
                      <option value="^DJI">Dow Jones</option>
                      <option value="^FTSE">FTSE 100</option>
                      <option value="^N225">Nikkei 225</option>
                    </select>
                    <p className="text-xs text-slate-500 mt-1">
                      Used for Beta calculation and market correlation
                    </p>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-3 mt-6 pt-4 border-t border-slate-200">
                <button
                  onClick={() => handleUpdate(editingId)}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
                >
                  <Check size={18} />
                  Save Changes
                </button>
                <button
                  onClick={() => setEditingId(null)}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300 transition font-medium"
                >
                  <X size={18} />
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {deleteTarget && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-xl font-bold text-slate-900 mb-2">Delete Portfolio</h3>
            <p className="text-slate-600 mb-4">
              Are you sure you want to delete <strong>{deleteTarget.name}</strong>? This action cannot be undone.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => handleDelete(deleteTarget.id)}
                className="flex-1 bg-red-600 text-white py-2 rounded-lg hover:bg-red-700 transition"
              >
                Delete
              </button>
              <button
                onClick={() => setDeleteTarget(null)}
                className="flex-1 bg-slate-200 text-slate-700 py-2 rounded-lg hover:bg-slate-300 transition"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PortfoliosList;
