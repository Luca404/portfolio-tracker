import { type ReactNode } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { useData } from '../../contexts/DataContext';
import { useSwipeNavigation } from '../../hooks/useSwipeNavigation';

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { logout, user } = useAuth();
  const { clearCache } = useData();

  const navItems = [
    { path: '/accounts', label: 'Conti', icon: '🏦' },
    { path: '/categories', label: 'Categorie', icon: '🏷️' },
    { path: '/transactions', label: 'Transazioni', icon: '💰' },
    { path: '/stats', label: 'Recap', icon: '📊' },
    { path: '/portfolios', label: 'Investimenti', icon: '📈' },
  ];

  const routes = navItems.map(item => item.path);
  const { swipeOffset, isSwipingHorizontally } = useSwipeNavigation({
    threshold: 120,
    velocityThreshold: 0.3,
    routes
  });

  const handleLogout = () => {
    clearCache();
    logout();
  };

  return (
    <div
      className="flex flex-col bg-gray-50 dark:bg-gray-900"
      style={{
        minHeight: '100dvh'
      }}
    >
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm flex-shrink-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <button
            onClick={() => navigate('/transactions')}
            className="text-xl font-bold text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 transition-colors"
          >
            Trackr
          </button>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-600 dark:text-gray-400 hidden sm:inline">
              {user?.name}
            </span>
            <button
              onClick={() => navigate('/settings')}
              className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 transition-colors"
              title="Impostazioni"
            >
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main
        className="flex-1 max-w-7xl w-full mx-auto px-4 py-3"
        style={{
          paddingBottom: '6rem', // Spazio per navbar (64px) + margine extra
          transition: isSwipingHorizontally ? 'none' : 'transform 0.3s ease-out, opacity 0.3s ease-out',
          transform: `translateX(${Math.max(-30, Math.min(30, swipeOffset * 0.2))}px)`,
          opacity: isSwipingHorizontally ? Math.max(0.7, 1 - Math.abs(swipeOffset) / 800) : 1
        }}
      >
        {children}
      </main>

      {/* Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 right-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 safe-area-pb z-[60]">
        <div className="flex justify-around items-center h-16">
          {navItems.map((item) => (
            <button
              key={item.path}
              onClick={() => navigate(item.path)}
              className={`flex flex-col items-center justify-center flex-1 h-full transition-colors ${
                location.pathname === item.path
                  ? 'text-primary-600 dark:text-primary-400'
                  : 'text-gray-600 dark:text-gray-400'
              }`}
            >
              <span className="text-2xl mb-1">{item.icon}</span>
              <span className="text-xs">{item.label}</span>
            </button>
          ))}
        </div>
      </nav>
    </div>
  );
}
