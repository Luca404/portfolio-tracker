import React, { useState } from 'react';
import { LogOut, Menu, X } from 'lucide-react';

function Navbar({ user, onLogout, currentView, setCurrentView, hasPortfolioSelected, onGoPortfolios }) {
  const [menuOpen, setMenuOpen] = useState(false);

  const navItems = hasPortfolioSelected ? [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'orders',    label: 'Orders' },
    { id: 'analyze',   label: 'Analyze' },
    { id: 'compare',   label: 'Compare' },
    { id: 'optimize',  label: 'Optimize' },
  ] : [];

  const handleNav = (id) => {
    setCurrentView(id);
    setMenuOpen(false);
  };

  return (
    <nav className="bg-white border-b border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <button onClick={() => { onGoPortfolios(); setMenuOpen(false); }} className="flex items-center gap-2 hover:text-blue-600 transition shrink-0">
            <img src="/logo1.svg" alt="pfTrackr" className="w-7 h-7 md:w-8 md:h-8" />
            <span className="text-lg md:text-xl font-bold text-slate-900">pfTrackr</span>
          </button>

          {/* Desktop nav */}
          {hasPortfolioSelected && (
            <div className="hidden md:flex items-center gap-1 lg:gap-4">
              {navItems.map(({ id, label }) => (
                <button
                  key={id}
                  onClick={() => setCurrentView(id)}
                  className={`px-3 py-2 rounded-lg font-medium text-sm lg:text-base transition ${
                    currentView === id ? 'bg-blue-50 text-blue-600' : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          )}

          {/* Right side */}
          <div className="flex items-center gap-2">
            <span className="hidden sm:block text-sm text-slate-600">{user?.username}</span>
            <button
              onClick={onLogout}
              className="flex items-center gap-1.5 px-3 py-2 text-slate-600 hover:text-red-600 transition text-sm"
            >
              <LogOut className="w-4 h-4" />
              <span className="hidden sm:inline">Logout</span>
            </button>
            {/* Hamburger — mobile only */}
            {hasPortfolioSelected && (
              <button
                onClick={() => setMenuOpen(o => !o)}
                className="md:hidden p-2 rounded-lg text-slate-600 hover:bg-slate-100 transition"
                aria-label="Menu"
              >
                {menuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Mobile dropdown menu */}
      {menuOpen && hasPortfolioSelected && (
        <div className="md:hidden border-t border-slate-200 bg-white">
          {navItems.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => handleNav(id)}
              className={`w-full text-left px-6 py-3 font-medium text-sm transition ${
                currentView === id ? 'text-blue-600 bg-blue-50' : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      )}
    </nav>
  );
}

export default Navbar;
