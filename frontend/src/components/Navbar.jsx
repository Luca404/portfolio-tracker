import React from 'react';
import { LogOut } from 'lucide-react';

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
              <img src="/logo1.svg" alt="pfTrackr" className="w-8 h-8" />
              <span className="text-xl font-bold text-slate-900">pfTrackr</span>
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
                  onClick={() => setCurrentView('analyze')}
                  className={`px-4 py-2 rounded-lg font-medium transition ${
                    currentView === 'analyze' ? 'bg-blue-50 text-blue-600' : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Analyze
                </button>
                <button
                  onClick={() => setCurrentView('compare')}
                  className={`px-4 py-2 rounded-lg font-medium transition ${
                    currentView === 'compare' ? 'bg-blue-50 text-blue-600' : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  Compare
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

export default Navbar;
