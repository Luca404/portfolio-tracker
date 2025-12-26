import { API_URL } from '../config';

/**
 * API Service Layer
 * Centralizza tutte le chiamate API per il backend
 */

// ============================================================================
// Authentication
// ============================================================================

export const authAPI = {
  /**
   * Register a new user
   */
  register: async (username, password) => {
    const res = await fetch(`${API_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Registration failed');
    }
    return res.json();
  },

  /**
   * Login user
   */
  login: async (username, password) => {
    const res = await fetch(`${API_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Login failed');
    }
    return res.json();
  },

  /**
   * Get current user info
   */
  getMe: async (token) => {
    const res = await fetch(`${API_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Unauthorized');
    return res.json();
  }
};

// ============================================================================
// Portfolios
// ============================================================================

export const portfolioAPI = {
  /**
   * Get all portfolios
   */
  getAll: async (token) => {
    const res = await fetch(`${API_URL}/portfolios`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Failed to fetch portfolios');
    return res.json();
  },

  /**
   * Get portfolios count
   */
  getCount: async (token) => {
    const res = await fetch(`${API_URL}/portfolios/count`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Failed to fetch count');
    return res.json();
  },

  /**
   * Get single portfolio by ID
   */
  getById: async (token, portfolioId) => {
    const res = await fetch(`${API_URL}/portfolios/${portfolioId}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Failed to fetch portfolio');
    return res.json();
  },

  /**
   * Create new portfolio
   */
  create: async (token, portfolio) => {
    const res = await fetch(`${API_URL}/portfolios`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify(portfolio)
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Failed to create portfolio');
    }
    return res.json();
  },

  /**
   * Update portfolio
   */
  update: async (token, portfolioId, portfolio) => {
    const res = await fetch(`${API_URL}/portfolios/${portfolioId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify(portfolio)
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Failed to update portfolio');
    }
    return res.json();
  },

  /**
   * Delete portfolio
   */
  delete: async (token, portfolioId) => {
    const res = await fetch(`${API_URL}/portfolios/${portfolioId}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Failed to delete portfolio');
    return res.json();
  }
};

// ============================================================================
// Orders
// ============================================================================

export const orderAPI = {
  /**
   * Get all orders for a portfolio
   */
  getByPortfolio: async (token, portfolioId) => {
    const res = await fetch(`${API_URL}/orders/${portfolioId}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Failed to fetch orders');
    return res.json();
  },

  /**
   * Create new order
   */
  create: async (token, order) => {
    const res = await fetch(`${API_URL}/orders`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify(order)
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Failed to create order');
    }
    return res.json();
  },

  /**
   * Update order
   */
  update: async (token, orderId, order) => {
    const res = await fetch(`${API_URL}/orders/${orderId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify(order)
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Failed to update order');
    }
    return res.json();
  },

  /**
   * Delete order
   */
  delete: async (token, orderId) => {
    const res = await fetch(`${API_URL}/orders/${orderId}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Failed to delete order');
    return res.json();
  }
};

// ============================================================================
// Analysis
// ============================================================================

export const analysisAPI = {
  /**
   * Get portfolio analysis
   */
  getPortfolioAnalysis: async (token, portfolioId) => {
    const res = await fetch(`${API_URL}/analysis/${portfolioId}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Failed to fetch analysis');
    return res.json();
  }
};

// ============================================================================
// Market Data
// ============================================================================

export const marketDataAPI = {
  /**
   * Get market data for a symbol
   */
  getSymbol: async (symbol) => {
    const res = await fetch(`${API_URL}/market-data/${symbol}`);
    if (!res.ok) throw new Error('Failed to fetch market data');
    return res.json();
  },

  /**
   * Get position history
   */
  getPositionHistory: async (token, portfolioId, symbol) => {
    const res = await fetch(`${API_URL}/portfolio/history/${portfolioId}/${symbol}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Failed to fetch position history');
    return res.json();
  },

  /**
   * Get risk-free rate
   */
  getRiskFreeRate: async (currency) => {
    const res = await fetch(`${API_URL}/market-data/risk-free-rate/${currency}`);
    if (!res.ok) throw new Error('Failed to fetch risk-free rate');
    return res.json();
  },

  /**
   * Get market benchmark
   */
  getBenchmark: async (currency) => {
    const res = await fetch(`${API_URL}/market-data/benchmark/${currency}`);
    if (!res.ok) throw new Error('Failed to fetch benchmark');
    return res.json();
  },

  /**
   * Get portfolio market context
   */
  getPortfolioContext: async (token, portfolioId) => {
    const res = await fetch(`${API_URL}/market-data/portfolio-context/${portfolioId}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) throw new Error('Failed to fetch market context');
    return res.json();
  }
};

// ============================================================================
// Symbols
// ============================================================================

export const symbolAPI = {
  /**
   * Search symbols
   */
  search: async (query, instrumentType = 'stock') => {
    const res = await fetch(`${API_URL}/symbols/search?q=${encodeURIComponent(query)}&instrument_type=${instrumentType}`);
    if (!res.ok) throw new Error('Failed to search symbols');
    return res.json();
  },

  /**
   * Get UCITS symbols
   */
  getUcits: async () => {
    const res = await fetch(`${API_URL}/symbols/ucits`);
    if (!res.ok) throw new Error('Failed to fetch UCITS symbols');
    return res.json();
  },

  /**
   * Get ETF list
   */
  getEtfList: async (etfType = 'all') => {
    const res = await fetch(`${API_URL}/symbols/etf-list?etf_type=${etfType}`);
    if (!res.ok) throw new Error('Failed to fetch ETF list');
    return res.json();
  },

  /**
   * Get ETF stats
   */
  getEtfStats: async () => {
    const res = await fetch(`${API_URL}/symbols/etf-stats`);
    if (!res.ok) throw new Error('Failed to fetch ETF stats');
    return res.json();
  }
};

// ============================================================================
// Optimization
// ============================================================================

export const optimizationAPI = {
  /**
   * Optimize portfolio
   */
  optimize: async (token, request) => {
    const res = await fetch(`${API_URL}/portfolio/optimize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify(request)
    });
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || 'Optimization failed');
    }
    return res.json();
  }
};

// ============================================================================
// Default export - tutte le API in un oggetto
// ============================================================================

export default {
  auth: authAPI,
  portfolio: portfolioAPI,
  order: orderAPI,
  analysis: analysisAPI,
  marketData: marketDataAPI,
  symbol: symbolAPI,
  optimization: optimizationAPI
};
