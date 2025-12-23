import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { PieChart as PieIcon, Plus, LogOut, BarChart3 } from 'lucide-react';

const API_URL = 'http://localhost:8000';

const formatCurrencyValue = (val, currency) => {
  const symbol = currency === 'EUR' ? '€' : currency === 'USD' ? '$' : currency ? `${currency} ` : '';
  const num = Number(val || 0);
  return `${symbol}${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

// Main App Component
export default function App() {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [user, setUser] = useState(null);
  const [currentView, setCurrentView] = useState('portfolios');
  const [selectedPortfolio, setSelectedPortfolio] = useState(null);
  const [portfolios, setPortfolios] = useState([]);
  const [portfoliosLoading, setPortfoliosLoading] = useState(false);

  useEffect(() => {
    if (token) {
      fetchUser();
      fetchPortfolios();
    }
  }, [token]);

  const fetchUser = async () => {
    try {
      const res = await fetch(`${API_URL}/auth/me`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setUser(data);
      } else {
        handleLogout();
      }
    } catch (err) {
      handleLogout();
    }
  };

  const fetchPortfolios = async () => {
    if (!token) return;
    setPortfoliosLoading(true);
    try {
      const res = await fetch(`${API_URL}/portfolios`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      setPortfolios(data.portfolios || []);
      if (selectedPortfolio) {
        const updated = data.portfolios?.find(p => p.id === selectedPortfolio.id);
        if (updated) {
          setSelectedPortfolio(updated);
        } else {
          setSelectedPortfolio(null);
          setCurrentView('portfolios');
        }
      }
    } catch (err) {
      console.error('Error fetching portfolios:', err);
    } finally {
      setPortfoliosLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
    setCurrentView('portfolios');
    setSelectedPortfolio(null);
    setPortfolios([]);
  };

  if (!token) {
    return <AuthPage onLogin={(newToken, userData) => {
      setToken(newToken);
      setUser(userData);
      localStorage.setItem('token', newToken);
    }} />;
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <Navbar
        user={user}
        onLogout={handleLogout}
        currentView={currentView}
        setCurrentView={setCurrentView}
        hasPortfolioSelected={!!selectedPortfolio}
        onGoPortfolios={() => {
          setSelectedPortfolio(null);
          setCurrentView('portfolios');
        }}
      />
      
      <div className="max-w-7xl mx-auto px-4 py-8">
        {currentView === 'portfolios' && (
          <PortfoliosList 
            token={token} 
            portfolios={portfolios}
            onRefresh={fetchPortfolios}
            loading={portfoliosLoading}
            onSelectPortfolio={(p) => {
              setSelectedPortfolio(p);
              setCurrentView('dashboard');
            }} 
          />
        )}
        
        {currentView === 'dashboard' && selectedPortfolio && (
          <Dashboard
            token={token}
            portfolio={selectedPortfolio}
            portfolios={portfolios}
            onSelectPortfolio={(p) => {
              setSelectedPortfolio(p);
              setCurrentView('dashboard');
            }}
            onDeleted={() => {
              fetchPortfolios();
              setSelectedPortfolio(null);
              setCurrentView('portfolios');
            }}
            refreshPortfolios={fetchPortfolios}
          />
        )}
        
        {currentView === 'orders' && selectedPortfolio && (
          <OrdersPage
            token={token}
            portfolio={selectedPortfolio}
            portfolios={portfolios}
            onSelectPortfolio={(p) => {
              setSelectedPortfolio(p);
              setCurrentView('orders');
            }}
            refreshPortfolios={fetchPortfolios}
          />
        )}
        
        {currentView === 'optimize' && selectedPortfolio && (
          <OptimizePage token={token} portfolio={selectedPortfolio} />
        )}
      </div>
    </div>
  );
}

// Auth Page
function AuthPage({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    username: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/register';
      const payload = isLogin
        ? { email: formData.email, password: formData.password }
        : { email: formData.email, password: formData.password, username: formData.username };
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();

      if (res.ok) {
        onLogin(data.access_token, data.user);
      } else {
        setError(data.detail || 'Authentication failed');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md p-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-600 rounded-full mb-4">
            <BarChart3 className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Portfolio Tracker</h1>
          <p className="text-slate-600">Professional portfolio management platform</p>
        </div>

        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setIsLogin(true)}
            className={`flex-1 py-2 rounded-lg font-medium transition ${
              isLogin ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            Sign In
          </button>
          <button
            onClick={() => setIsLogin(false)}
            className={`flex-1 py-2 rounded-lg font-medium transition ${
              !isLogin ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            Sign Up
          </button>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            {error}
          </div>
        )}

        <div className="space-y-4">
          {!isLogin && (
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Username</label>
              <input
                type="text"
                value={formData.username}
                onChange={(e) => setFormData({...formData, username: e.target.value})}
                className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="trader123"
              />
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Email</label>
            <input
              type="email"
              value={formData.email}
              onChange={(e) => setFormData({...formData, email: e.target.value})}
              className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="john@example.com"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Password</label>
            <input
              type="password"
              value={formData.password}
              onChange={(e) => setFormData({...formData, password: e.target.value})}
              className="w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="••••••••"
            />
          </div>

          <button
            onClick={handleSubmit}
            disabled={loading}
            className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 transition disabled:opacity-50"
          >
            {loading ? 'Please wait...' : (isLogin ? 'Sign In' : 'Create Account')}
          </button>
        </div>
      </div>
    </div>
  );
}

// Navbar
function Navbar({ user, onLogout, currentView, setCurrentView, hasPortfolioSelected, onGoPortfolios }) {
  return (
    <nav className="bg-white border-b border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-8">
            <button
              onClick={onGoPortfolios}
              className="flex items-center gap-2 hover:text-blue-600 transition"
            >
              <BarChart3 className="w-8 h-8 text-blue-600" />
              <span className="text-xl font-bold text-slate-900">Portfolio Tracker</span>
            </button>
            
            {hasPortfolioSelected && (
              <div className="flex items-center gap-4">
                <button
                  onClick={() => setCurrentView('dashboard')}
                  className={`px-4 py-2 rounded-lg font-medium transition ${
                    currentView === 'dashboard' ? 'bg-blue-50 text-blue-600' : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Dashboard
                </button>
                <button
                  onClick={() => setCurrentView('orders')}
                  className={`px-4 py-2 rounded-lg font-medium transition ${
                    currentView === 'orders' ? 'bg-blue-50 text-blue-600' : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Orders
                </button>
                <button
                  onClick={() => setCurrentView('optimize')}
                  className={`px-4 py-2 rounded-lg font-medium transition ${
                    currentView === 'optimize' ? 'bg-blue-50 text-blue-600' : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Optimize
                </button>
              </div>
            )}
            </div>

          <div className="flex items-center gap-4">
            <span className="text-sm text-slate-600">{user?.username}</span>
            <button
              onClick={onLogout}
              className="flex items-center gap-2 px-4 py-2 text-slate-600 hover:text-red-600 transition"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}

// Portfolios List
function PortfoliosList({ token, onSelectPortfolio, portfolios, onRefresh, loading }) {
  const [showCreate, setShowCreate] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    description: ''
  });

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
        setFormData({ name: '', description: '' });
        onRefresh();
      }
    } catch (err) {
      console.error('Error creating portfolio:', err);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 mb-2">My Portfolios</h1>
          <p className="text-slate-600">Manage your investment portfolios</p>
          {loading && <p className="text-sm text-slate-500 mt-1">Loading...</p>}
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
        {portfolios.map((portfolio) => (
          <div
            key={portfolio.id}
            onClick={() => onSelectPortfolio(portfolio)}
            className="bg-white rounded-lg shadow hover:shadow-lg transition p-6 cursor-pointer border-2 border-transparent hover:border-blue-500"
          >
            <h3 className="text-xl font-bold text-slate-900 mb-2">{portfolio.name}</h3>
            <p className="text-slate-600 text-sm mb-4">{portfolio.description || 'No description'}</p>
            <div className="flex items-center justify-between pt-4 border-t border-slate-200">
              <span className="text-sm text-slate-600">Portfolio Value</span>
              <span className="text-lg font-bold text-blue-600">${(portfolio.total_value || 0).toLocaleString()}</span>
            </div>
          </div>
        ))}
      </div>

      {portfolios.length === 0 && !showCreate && (
        <div className="text-center py-16">
          <PieIcon className="w-16 h-16 text-slate-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-slate-900 mb-2">No Portfolios Yet</h3>
          <p className="text-slate-600 mb-6">Create your first portfolio to get started</p>
          <button
            onClick={() => setShowCreate(true)}
            className="inline-flex items-center gap-2 bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition"
          >
            <Plus className="w-5 h-5" />
            Create Portfolio
          </button>
        </div>
      )}
    </div>
  );
}

// Dashboard Component
function Dashboard({ token, portfolio, portfolios, onSelectPortfolio, onDeleted, refreshPortfolios }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
   const [actionLoading, setActionLoading] = useState(false);

  useEffect(() => {
    fetchData();
  }, [portfolio]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/portfolios/${portfolio.id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const result = await res.json();
      setData(result);
    } catch (err) {
      console.error('Error:', err);
    } finally {
      setLoading(false);
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

  if (loading) {
    return <div className="text-center py-16 text-slate-600">Loading dashboard...</div>;
  }

  const COLORS = ['#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];
  const pieData = data?.positions.map(p => ({ name: p.symbol, value: p.market_value })) || [];
  const typeMap = {};
  (data?.positions || []).forEach((p) => {
    const key = (p.instrument_type || 'Other').toUpperCase();
    typeMap[key] = (typeMap[key] || 0) + p.market_value;
  });
  const typeData = Object.entries(typeMap).map(([name, value]) => ({ name, value }));
  const summaryCurrency = data?.positions?.[0]?.currency || '';

  return (
    <div>
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 mb-2">{portfolio.name}</h1>
          <p className="text-slate-600">{portfolio.description || 'Portfolio Overview'}</p>
        </div>
        <div className="flex items-center gap-3">
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
            onClick={handleDelete}
            disabled={actionLoading}
            className="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition disabled:opacity-50"
          >
            Delete
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg shadow-lg p-6 text-white">
          <p className="text-blue-100 text-sm mb-1">Total Value</p>
          <p className="text-3xl font-bold">{formatCurrencyValue(data?.summary.total_value, summaryCurrency)}</p>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-slate-600 text-sm mb-1">Total P&L</p>
          <p className={`text-3xl font-bold ${data?.summary.total_gain_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {formatCurrencyValue(data?.summary.total_gain_loss, summaryCurrency)}
          </p>
          <p className={`text-sm font-semibold ${data?.summary.total_gain_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {data?.summary.total_gain_loss_pct >= 0 ? '+' : ''}{data?.summary.total_gain_loss_pct.toFixed(2)}%
          </p>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-slate-600 text-sm mb-1">Positions</p>
          <p className="text-3xl font-bold text-slate-900">{data?.positions.length}</p>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-bold text-slate-900 mb-4">Holdings</h2>
          <div className="inline-block w-full border border-slate-100 rounded-lg overflow-hidden">
            <table className="w-full text-[13.5px] leading-tight table-auto">
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
                  label={(entry) => `${entry.name}: ${((entry.value / data.summary.total_value) * 100).toFixed(1)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `$${value.toFixed(2)}`} />
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
                  label={(entry) => `${entry.name}: ${((entry.value / data.summary.total_value) * 100).toFixed(1)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {typeData.map((entry, index) => (
                    <Cell key={`cell-type-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `$${value.toFixed(2)}`} />
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

// Orders Page Component  
function OrdersPage({ token, portfolio, portfolios, onSelectPortfolio, refreshPortfolios }) {
  const [orders, setOrders] = useState([]);
  const [showForm, setShowForm] = useState(false);
  const [editingOrderId, setEditingOrderId] = useState(null);
  const [symbolOptions, setSymbolOptions] = useState([]);
  const [symbolLoading, setSymbolLoading] = useState(false);
  const [ucitsCache, setUcitsCache] = useState([]);
  const [selectedInfo, setSelectedInfo] = useState({ name: '', exchange: '', currency: '' });
  const [skipSearch, setSkipSearch] = useState(false);
  const [lastChosenSymbol, setLastChosenSymbol] = useState('');
  const currencySymbol = selectedInfo.currency === 'EUR' ? '€' : selectedInfo.currency === 'USD' ? '$' : '';
  const [formData, setFormData] = useState({
    symbol: '',
    quantity: '',
    price: '',
    commission: '0',
    instrument_type: 'etf',
    order_type: 'buy',
    date: new Date().toISOString().split('T')[0]
  });
  const [touched, setTouched] = useState({});

  useEffect(() => {
    fetchOrders();
  }, [portfolio]);

  const handlePortfolioChange = (e) => {
    const next = portfolios.find(p => p.id === Number(e.target.value));
    if (next) {
      onSelectPortfolio(next);
    }
  };

  const fetchOrders = async () => {
    try {
      const res = await fetch(`${API_URL}/orders/${portfolio.id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      setOrders(data.orders || []);
    } catch (err) {
      console.error('Error:', err);
    }
  };

  useEffect(() => {
    const loadUcits = async () => {
      if (ucitsCache.length > 0) return;
      try {
        const res = await fetch(`${API_URL}/symbols/ucits`);
        if (res.ok) {
          const data = await res.json();
          setUcitsCache(data.results || []);
        }
      } catch (e) {
        console.error('UCITS cache load error', e);
      }
    };
    loadUcits();
  }, [ucitsCache.length]);

  const handleSubmit = async () => {
    const errs = {};
    const qtyVal = parseInt(formData.quantity, 10);
    if (!formData.symbol || !selectedInfo.name) errs.symbol = true;
    if (!formData.quantity || isNaN(qtyVal) || qtyVal <= 0) errs.quantity = true;
    if (!formData.price || isNaN(parseFloat(formData.price)) || parseFloat(formData.price) <= 0) errs.price = true;
    if (Object.keys(errs).length > 0) {
      setTouched({...touched, ...errs});
      return;
    }
    try {
      const payload = {
        portfolio_id: portfolio.id,
        symbol: formData.symbol,
        name: selectedInfo.name,
        exchange: selectedInfo.exchange,
        currency: selectedInfo.currency,
        quantity: qtyVal,
        price: parseFloat(formData.price),
        commission: parseFloat(formData.commission || 0),
        instrument_type: formData.instrument_type,
        order_type: formData.order_type,
        date: formData.date
      };

      const url = editingOrderId ? `${API_URL}/orders/${editingOrderId}` : `${API_URL}/orders`;
      const method = editingOrderId ? 'PUT' : 'POST';

      const res = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        setShowForm(false);
        setFormData({
          symbol: '', quantity: '', price: '', commission: '0', instrument_type: 'etf', order_type: 'buy',
          date: new Date().toISOString().split('T')[0]
        });
        setSelectedInfo({ name: '', exchange: '', currency: '' });
        setLastChosenSymbol('');
        setTouched({});
        setEditingOrderId(null);
        fetchOrders();
        refreshPortfolios();
      }
    } catch (err) {
      console.error('Error:', err);
    }
  };

  useEffect(() => {
    if (skipSearch) {
      setSkipSearch(false);
      return;
    }
    if (formData.symbol && formData.symbol === lastChosenSymbol) {
      return;
    }
    const controller = new AbortController();
    const fetchSuggestions = async () => {
      if (!formData.symbol || formData.symbol.length < 2) {
        setSymbolOptions([]);
        setSelectedInfo({ name: '', exchange: '', currency: '' });
        return;
      }
      // local search for etf using ucits cache
      if (formData.instrument_type === 'etf') {
        const q = formData.symbol.toUpperCase();
        const filtered = ucitsCache
          .filter(item => {
            const ticker = (item.symbol || '').toUpperCase();
            const isin = (item.isin || '').toUpperCase();
            const tickerMatch = ticker.startsWith(q);
            const isinMatch = q.length === 12 && isin === q;
            return tickerMatch || isinMatch;
          })
          .slice(0, 25);
        setSymbolOptions(filtered);
        return;
      }

      setSymbolLoading(true);
      try {
        const res = await fetch(
          `${API_URL}/symbols/search?q=${encodeURIComponent(formData.symbol)}&instrument_type=${formData.instrument_type}`,
          { signal: controller.signal }
        );
        if (res.ok) {
          const data = await res.json();
          setSymbolOptions(data.results || []);
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          console.error('Symbol search error:', err);
        }
      } finally {
        setSymbolLoading(false);
      }
    };
    const timer = setTimeout(fetchSuggestions, 250);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [formData.symbol, formData.instrument_type, skipSearch, ucitsCache]);

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Orders</h1>
          <p className="text-slate-600">Manage portfolio transactions</p>
          <div className="mt-2">
            <select
              value={portfolio.id}
              onChange={handlePortfolioChange}
              className="px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              {portfolios.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>
        </div>
        <button
          onClick={() => {
            setShowForm(!showForm);
            if (!showForm) {
              setEditingOrderId(null);
              setFormData({
                symbol: '', quantity: '', price: '', commission: '0', instrument_type: 'etf', order_type: 'buy',
                date: new Date().toISOString().split('T')[0]
              });
              setSelectedInfo({ name: '', exchange: '', currency: '' });
              setLastChosenSymbol('');
            }
          }}
          className="flex items-center gap-2 bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition"
        >
          <Plus className="w-5 h-5" />
          {editingOrderId ? 'Edit Order' : 'New Order'}
        </button>
      </div>

      {showForm && (
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">{editingOrderId ? 'Edit Order' : 'Create Order'}</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="relative">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                {formData.instrument_type === 'etf' ? 'Ticker or ISIN' : 'Ticker or Name'}
              </label>
              <input
                type="text"
                value={formData.symbol}
                onChange={(e) => {
                  setFormData({...formData, symbol: e.target.value.toUpperCase()});
                  setSelectedInfo({ name: '', exchange: '', currency: '' });
                }}
                className={`w-full px-4 py-2 border ${touched.symbol && (!formData.symbol || !selectedInfo.name) ? 'border-red-400' : 'border-slate-300'} rounded-lg focus:ring-2 focus:ring-blue-500`}
                placeholder="AAPL"
              />
              {symbolLoading && <p className="text-xs text-slate-500 mt-1">Searching...</p>}
              {symbolOptions.length > 0 && formData.symbol && (
                <div className="absolute z-10 mt-1 border border-slate-200 rounded-lg max-h-48 overflow-auto bg-white shadow w-full">
                  {symbolOptions.map((opt) => (
                    <button
                      key={`${opt.symbol}-${opt.exchange || ''}`}
                      type="button"
                      onClick={() => {
                        setFormData({...formData, symbol: opt.symbol.toUpperCase()});
                        setSelectedInfo({ name: opt.name || '', exchange: opt.exchange || '', currency: opt.currency || '' });
                        setSymbolOptions([]);
                        setSkipSearch(true);
                        setLastChosenSymbol(opt.symbol.toUpperCase());
                      }}
                      className="w-full text-left px-3 py-2 hover:bg-slate-100 text-sm"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex flex-col">
                          <span className="font-semibold text-slate-900">{opt.symbol}</span>
                          {opt.name && <span className="text-slate-600 text-xs">{opt.name}</span>}
                        </div>
                        <div className="text-xs text-slate-500 text-right">
                          {opt.exchange && <div>{opt.exchange}</div>}
                          {opt.currency && <div>{opt.currency}</div>}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2">
                <div>
                  <label className="block text-xs text-slate-500 mb-1">Name</label>
                  <input
                    type="text"
                    value={selectedInfo.name}
                    readOnly
                    className="w-full px-3 py-2 border border-slate-200 rounded bg-slate-50 text-sm cursor-default select-none pointer-events-none"
                    placeholder="—"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-500 mb-1">Exchange</label>
                  <input
                    type="text"
                    value={selectedInfo.exchange}
                    readOnly
                    className="w-full px-3 py-2 border border-slate-200 rounded bg-slate-50 text-sm cursor-default select-none pointer-events-none"
                    placeholder="—"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-500 mb-1">Currency</label>
                  <input
                    type="text"
                    value={selectedInfo.currency}
                    readOnly
                    className="w-full px-3 py-2 border border-slate-200 rounded bg-slate-50 text-sm cursor-default select-none pointer-events-none"
                    placeholder="—"
                  />
                </div>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Instrument</label>
              <select
                value={formData.instrument_type}
                onChange={(e) => {
                  setFormData({...formData, instrument_type: e.target.value});
                  setSymbolOptions([]);
                  setSelectedInfo({ name: '', exchange: '', currency: '' });
                }}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="stock">Stock</option>
                <option value="etf">ETF</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Type</label>
              <select
                value={formData.order_type}
                onChange={(e) => setFormData({...formData, order_type: e.target.value})}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              >
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Quantity</label>
              <input
                type="number"
                min="1"
                step="1"
                value={formData.quantity}
                onChange={(e) => setFormData({...formData, quantity: e.target.value})}
                className={`w-full px-4 py-2 border ${touched.quantity && (!formData.quantity || isNaN(parseInt(formData.quantity, 10)) || parseInt(formData.quantity, 10) <= 0) ? 'border-red-400' : 'border-slate-300'} rounded-lg focus:ring-2 focus:ring-blue-500`}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Price</label>
              <div className="relative">
                {currencySymbol && <span className="absolute inset-y-0 left-3 flex items-center text-slate-500">{currencySymbol}</span>}
                <input
                  type="number"
                  step="0.01"
                  value={formData.price}
                  onChange={(e) => setFormData({...formData, price: e.target.value})}
                  className={`w-full ${currencySymbol ? 'pl-8 pr-3' : 'px-4'} py-2 border ${touched.price && (!formData.price || parseFloat(formData.price) <= 0) ? 'border-red-400' : 'border-slate-300'} rounded-lg focus:ring-2 focus:ring-blue-500`}
                  placeholder=""
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Commission</label>
              <div className="relative">
                {currencySymbol && <span className="absolute inset-y-0 left-3 flex items-center text-slate-500">{currencySymbol}</span>}
                <input
                  type="number"
                  step="0.01"
                  value={formData.commission}
                  onChange={(e) => setFormData({...formData, commission: e.target.value})}
                  className={`w-full ${currencySymbol ? 'pl-8 pr-3' : 'px-4'} py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500`}
                  placeholder="0"
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">Date</label>
              <input
                type="date"
                value={formData.date}
                onChange={(e) => setFormData({...formData, date: e.target.value})}
                className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="flex items-end gap-3">
              <button
                onClick={handleSubmit}
                className="flex-1 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition"
              >
                Submit
              </button>
              <button
                onClick={() => setShowForm(false)}
                className="flex-1 bg-slate-200 text-slate-700 py-2 rounded-lg hover:bg-slate-300 transition"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="w-full">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr>
              <th className="text-left py-4 px-6 text-sm font-semibold text-slate-700">Date</th>
              <th className="text-left py-4 px-6 text-sm font-semibold text-slate-700">Symbol</th>
              <th className="text-left py-4 px-6 text-sm font-semibold text-slate-700">Type</th>
              <th className="text-right py-4 px-6 text-sm font-semibold text-slate-700">Quantity</th>
              <th className="text-right py-4 px-6 text-sm font-semibold text-slate-700">Price</th>
              <th className="text-right py-4 px-6 text-sm font-semibold text-slate-700">Commission</th>
              <th className="text-right py-4 px-6 text-sm font-semibold text-slate-700">Total (incl. comm)</th>
              <th className="text-right py-4 px-6 text-sm font-semibold text-slate-700"></th>
            </tr>
          </thead>
          <tbody>
            {orders.slice().reverse().map((order) => (
              <tr key={order.id} className="border-b border-slate-100 hover:bg-slate-50">
                <td className="py-4 px-6 text-slate-700">{order.date}</td>
                <td className="py-4 px-6 font-semibold text-slate-900">{order.symbol}</td>
                <td className="py-4 px-6">
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    order.order_type === 'buy' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                  }`}>
                    {order.order_type.toUpperCase()}
                  </span>
                </td>
                <td className="py-4 px-6 text-right text-slate-700">{order.quantity}</td>
                <td className="py-4 px-6 text-right text-slate-700">{formatCurrencyValue(order.price, order.currency)}</td>
                <td className="py-4 px-6 text-right text-slate-700">{formatCurrencyValue(order.commission || 0, order.currency)}</td>
                <td className="py-4 px-6 text-right font-semibold text-slate-900">
                  {(() => {
                    const net = order.order_type === 'buy'
                      ? order.quantity * order.price + (order.commission || 0)
                      : order.quantity * order.price - (order.commission || 0);
                    return formatCurrencyValue(net, order.currency);
                  })()}
                </td>
                <td className="py-4 px-6 text-right">
                  <div className="flex justify-end gap-3">
                    <button
                      onClick={() => {
                        setShowForm(true);
                        setEditingOrderId(order.id);
                        setFormData({
                          symbol: order.symbol,
                          quantity: order.quantity.toString(),
                          price: order.price.toString(),
                          commission: (order.commission || 0).toString(),
                          instrument_type: order.instrument_type || 'stock',
                          order_type: order.order_type,
                          date: order.date
                        });
                      }}
                      className="text-blue-600 hover:text-blue-800 text-sm"
                    >
                      Edit
                    </button>
                    <button
                      onClick={async () => {
                        await fetch(`${API_URL}/orders/${order.id}`, {
                          method: 'DELETE',
                          headers: { Authorization: `Bearer ${token}` }
                        });
                        fetchOrders();
                        refreshPortfolios();
                      }}
                      className="text-red-600 hover:text-red-800 text-sm"
                    >
                      Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {orders.length === 0 && (
          <div className="text-center py-16 text-slate-500">No orders yet</div>
        )}
      </div>
    </div>
  );
}

// Optimize Page
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
      const res = await fetch(`${API_URL}/portfolio/optimize`, {
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
