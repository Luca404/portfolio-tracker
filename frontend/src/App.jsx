import React, { useState, useEffect } from 'react';
import { API_URL } from './config';
import Navbar from './components/Navbar';
import PortfoliosList from './components/PortfoliosList';
import {
  AuthPage,
  DashboardPage,
  OrdersPage,
  AnalyzePage,
  ComparePage,
  OptimizePage
} from './pages';

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [user, setUser] = useState(null);
  const [currentView, setCurrentView] = useState('portfolios');
  const [selectedPortfolio, setSelectedPortfolio] = useState(null);
  const [portfolios, setPortfolios] = useState([]);
  const [portfoliosLoading, setPortfoliosLoading] = useState(false);

  // Flag per evitare fetch multipli simultanei
  const initializingRef = React.useRef(false);

  useEffect(() => {
    if (token && !initializingRef.current) {
      initializingRef.current = true;
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
          <DashboardPage
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

        {currentView === 'analyze' && selectedPortfolio && (
          <AnalyzePage
            token={token}
            portfolio={selectedPortfolio}
            portfolios={portfolios}
            onSelectPortfolio={setSelectedPortfolio}
          />
        )}

        {currentView === 'compare' && selectedPortfolio && (
          <ComparePage token={token} portfolio={selectedPortfolio} />
        )}

        {currentView === 'optimize' && selectedPortfolio && (
          <OptimizePage token={token} portfolio={selectedPortfolio} />
        )}
      </div>
    </div>
  );
}
