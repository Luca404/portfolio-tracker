import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import Layout from '../components/layout/Layout';
import LoadingSpinner from '../components/common/LoadingSpinner';
import type { TransactionStats } from '../types';

export default function StatsPage() {
  const [stats, setStats] = useState<TransactionStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [period, setPeriod] = useState<'month' | 'year'>('month');

  useEffect(() => {
    loadStats();
  }, [period]);

  const loadStats = async () => {
    setIsLoading(true);
    try {
      const endDate = new Date();
      const startDate = new Date();

      if (period === 'month') {
        // Mese corrente
        startDate.setDate(1);
      } else {
        // Ultimi 12 mesi
        startDate.setMonth(startDate.getMonth() - 11);
        startDate.setDate(1);
      }

      const data = await apiService.getTransactionStats({
        startDate: startDate.toISOString().split('T')[0],
        endDate: endDate.toISOString().split('T')[0],
      });
      setStats(data);
    } catch (error) {
      console.error('Errore caricamento statistiche:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('it-IT', {
      style: 'currency',
      currency: 'EUR',
    }).format(amount);
  };

  if (isLoading) {
    return <LoadingSpinner />;
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Statistiche
          </h1>

          {/* Period selector */}
          <div className="flex bg-gray-200 dark:bg-gray-700 rounded-lg p-1">
            <button
              onClick={() => setPeriod('month')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                period === 'month'
                  ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-400'
              }`}
            >
              Mese
            </button>
            <button
              onClick={() => setPeriod('year')}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                period === 'year'
                  ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-400'
              }`}
            >
              Anno
            </button>
          </div>
        </div>

        {/* Overview cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="card bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 border-2 border-red-200 dark:border-red-800">
            <div className="text-sm text-red-600 dark:text-red-400 font-medium mb-1">
              Totale Uscite
            </div>
            <div className="text-3xl font-bold text-red-700 dark:text-red-300 mb-2">
              {formatCurrency(stats?.totalExpenses || 0)}
            </div>
          </div>

          <div className="card bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 border-2 border-green-200 dark:border-green-800">
            <div className="text-sm text-green-600 dark:text-green-400 font-medium mb-1">
              Totale Entrate
            </div>
            <div className="text-3xl font-bold text-green-700 dark:text-green-300 mb-2">
              {formatCurrency(stats?.totalIncome || 0)}
            </div>
          </div>

          <div className="card bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 border-2 border-purple-200 dark:border-purple-800">
            <div className="text-sm text-purple-600 dark:text-purple-400 font-medium mb-1">
              Bilancio
            </div>
            <div
              className={`text-3xl font-bold mb-2 ${
                (stats?.balance || 0) >= 0
                  ? 'text-purple-700 dark:text-purple-300'
                  : 'text-red-700 dark:text-red-300'
              }`}
            >
              {formatCurrency(stats?.balance || 0)}
            </div>
          </div>
        </div>

        {/* Spese per categoria - dettaglio completo */}
        {stats?.expensesByCategory && Object.keys(stats.expensesByCategory).length > 0 && (
          <div className="card">
            <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-gray-100">
              Distribuzione per Categoria
            </h2>
            <div className="space-y-4">
              {Object.entries(stats.expensesByCategory)
                .sort(([, a], [, b]) => b - a)
                .map(([category, amount]) => {
                  const percentage = ((amount / stats.totalExpenses) * 100).toFixed(1);
                  return (
                    <div key={category}>
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {category}
                        </span>
                        <div className="text-right">
                          <div className="font-bold text-gray-900 dark:text-gray-100">
                            {formatCurrency(amount)}
                          </div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">
                            {percentage}%
                          </div>
                        </div>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
                        <div
                          className="bg-gradient-to-r from-primary-600 to-primary-400 h-3 rounded-full transition-all"
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
            </div>
          </div>
        )}

        {/* Trend mensile */}
        {stats?.monthlyTrend && stats.monthlyTrend.length > 0 && (
          <div className="card">
            <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-gray-100">
              {period === 'month' ? 'Trend Mese Corrente' : 'Trend Ultimi 12 Mesi'}
            </h2>
            <div className="space-y-3">
              {stats.monthlyTrend.map((item) => {
                const maxAmount = Math.max(...stats.monthlyTrend.map(m => Math.max(m.expenses, m.income)));
                const expenseWidth = maxAmount > 0 ? (item.expenses / maxAmount * 100) : 0;
                const incomeWidth = maxAmount > 0 ? (item.income / maxAmount * 100) : 0;
                const monthName = new Date(item.month + '-01').toLocaleDateString('it-IT', {
                  month: 'long',
                  year: period === 'year' ? '2-digit' : 'numeric'
                });
                const balance = item.income - item.expenses;

                return (
                  <div key={item.month} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="font-medium text-gray-900 dark:text-gray-100 capitalize">
                        {monthName}
                      </span>
                      <div className="flex gap-4 text-sm">
                        <div className="text-right">
                          <span className="text-gray-500 dark:text-gray-400">Entrate: </span>
                          <span className="font-semibold text-green-600 dark:text-green-400">
                            {formatCurrency(item.income)}
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="text-gray-500 dark:text-gray-400">Uscite: </span>
                          <span className="font-semibold text-red-600 dark:text-red-400">
                            {formatCurrency(item.expenses)}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex gap-2 items-center">
                      <div className="flex-1">
                        <div className="flex gap-1 h-8 bg-gray-100 dark:bg-gray-800 rounded">
                          <div
                            className="bg-red-500 dark:bg-red-400 rounded transition-all"
                            style={{ width: `${expenseWidth}%` }}
                            title={`Uscite: ${formatCurrency(item.expenses)}`}
                          />
                          <div
                            className="bg-green-500 dark:bg-green-400 rounded transition-all"
                            style={{ width: `${incomeWidth}%` }}
                            title={`Entrate: ${formatCurrency(item.income)}`}
                          />
                        </div>
                      </div>
                      <div className={`text-sm font-bold w-24 text-right ${
                        balance >= 0
                          ? 'text-purple-600 dark:text-purple-400'
                          : 'text-red-600 dark:text-red-400'
                      }`}>
                        {formatCurrency(balance)}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="flex gap-4 mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-red-500 dark:bg-red-400 rounded"></div>
                <span className="text-gray-600 dark:text-gray-400">Uscite</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-green-500 dark:bg-green-400 rounded"></div>
                <span className="text-gray-600 dark:text-gray-400">Entrate</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-purple-500 dark:bg-purple-400 rounded"></div>
                <span className="text-gray-600 dark:text-gray-400">Bilancio</span>
              </div>
            </div>
          </div>
        )}

        {/* Investimenti */}
        {stats?.totalInvestments && stats.totalInvestments > 0 && (
          <div className="card bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 border-2 border-blue-200 dark:border-blue-800">
            <h2 className="text-xl font-semibold mb-2 text-blue-900 dark:text-blue-100">
              Investimenti
            </h2>
            <div className="text-3xl font-bold text-blue-700 dark:text-blue-300">
              {formatCurrency(stats.totalInvestments)}
            </div>
            <div className="text-sm text-blue-600 dark:text-blue-400 mt-2">
              💡 Sincronizzati con pfTrackr
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
