import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import Layout from '../components/layout/Layout';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PeriodSelector from '../components/common/PeriodSelector';
import DateRangePicker from '../components/common/DateRangePicker';
import { usePeriod } from '../hooks/usePeriod';
import type { Transaction } from '../types';

type StatsFilter = 'expense' | 'income' | 'investment';
type PeriodType = 'day' | 'week' | 'month' | 'year' | 'all' | 'custom';

interface CategoryStat {
  name: string;
  icon: string;
  amount: number;
  percentage: number;
  count: number;
}

export default function StatsPageNew() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState<StatsFilter>('expense');
  const [isDatePickerOpen, setIsDatePickerOpen] = useState(false);

  // Period state - condiviso tra le pagine
  const { startDate, endDate, setPeriod } = usePeriod();

  useEffect(() => {
    loadTransactions();
  }, [startDate, endDate]);

  const loadTransactions = async () => {
    setIsLoading(true);
    try {
      const data = await apiService.getTransactions({
        startDate: startDate.toISOString().split('T')[0],
        endDate: endDate.toISOString().split('T')[0],
      });
      setTransactions(data);
    } catch (error) {
      console.error('Errore caricamento transazioni:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePeriodChange = (start: Date, end: Date, type: PeriodType) => {
    setPeriod(start, end, type);
  };

  const handleCustomPeriodConfirm = (start: Date, end: Date) => {
    setPeriod(start, end, 'custom');
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('it-IT', {
      style: 'currency',
      currency: 'EUR',
    }).format(amount);
  };

  // Calcola totali del periodo
  const totalIncome = transactions
    .filter(t => t.type === 'income')
    .reduce((sum, t) => sum + t.amount, 0);

  const totalExpense = transactions
    .filter(t => t.type === 'expense')
    .reduce((sum, t) => sum + t.amount, 0);

  const totalInvestment = transactions
    .filter(t => t.type === 'investment')
    .reduce((sum, t) => sum + t.amount, 0);

  // Saldo del periodo (entrate - uscite, escludendo investimenti e trasferimenti)
  const periodBalance = totalIncome - totalExpense;

  // Calcola statistiche per categoria
  const filteredTransactions = transactions.filter(t => t.type === filter);
  const total = filteredTransactions.reduce((sum, t) => sum + Math.abs(t.amount), 0);

  const categoryStats: CategoryStat[] = filteredTransactions.reduce((acc, t) => {
    const existing = acc.find(c => c.name === t.category);
    if (existing) {
      existing.amount += Math.abs(t.amount);
      existing.count += 1;
    } else {
      acc.push({
        name: t.category,
        icon: '📌', // In un'app reale, potresti recuperare l'icona dalla categoria
        amount: Math.abs(t.amount),
        percentage: 0,
        count: 1,
      });
    }
    return acc;
  }, [] as CategoryStat[]);

  // Calcola percentuali
  categoryStats.forEach(stat => {
    stat.percentage = total > 0 ? (stat.amount / total) * 100 : 0;
  });

  // Ordina per amount decrescente
  categoryStats.sort((a, b) => b.amount - a.amount);

  // Trova il massimo per normalizzare le barre
  const maxAmount = categoryStats.length > 0 ? categoryStats[0].amount : 1;

  if (isLoading) {
    return <LoadingSpinner />;
  }

  return (
    <Layout>
      <div className="space-y-4">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Statistiche
        </h1>

        {/* Period Selector */}
        <PeriodSelector
          startDate={startDate}
          endDate={endDate}
          onPeriodChange={handlePeriodChange}
          onCustomClick={() => setIsDatePickerOpen(true)}
        />

        {/* Saldi periodo */}
        <div className="grid grid-cols-2 gap-3">
          <div className="card bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 border-green-200 dark:border-green-800">
            <div className="text-sm text-green-700 dark:text-green-300 mb-1">Entrate</div>
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {formatCurrency(totalIncome)}
            </div>
          </div>

          <div className="card bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 border-red-200 dark:border-red-800">
            <div className="text-sm text-red-700 dark:text-red-300 mb-1">Uscite</div>
            <div className="text-2xl font-bold text-red-600 dark:text-red-400">
              {formatCurrency(totalExpense)}
            </div>
          </div>

          <div className="card bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 border-blue-200 dark:border-blue-800">
            <div className="text-sm text-blue-700 dark:text-blue-300 mb-1">Investimenti</div>
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {formatCurrency(totalInvestment)}
            </div>
          </div>

          <div className="card bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 border-purple-200 dark:border-purple-800">
            <div className="text-sm text-purple-700 dark:text-purple-300 mb-1">Saldo Periodo</div>
            <div className={`text-2xl font-bold ${
              periodBalance >= 0
                ? 'text-green-600 dark:text-green-400'
                : 'text-red-600 dark:text-red-400'
            }`}>
              {formatCurrency(periodBalance)}
            </div>
          </div>
        </div>

        {/* Filtri */}
        <div className="flex gap-2">
          <button
            onClick={() => setFilter('expense')}
            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
              filter === 'expense'
                ? 'bg-red-500 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            💸 Uscite
          </button>
          <button
            onClick={() => setFilter('income')}
            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
              filter === 'income'
                ? 'bg-green-500 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            💰 Entrate
          </button>
          <button
            onClick={() => setFilter('investment')}
            className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
              filter === 'investment'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            📈 Investimenti
          </button>
        </div>

        {/* Totale */}
        <div className="card bg-gradient-to-br from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20 border-primary-200 dark:border-primary-800">
          <div className="text-sm text-primary-700 dark:text-primary-300 mb-1">
            Totale {filter === 'expense' ? 'Uscite' : filter === 'income' ? 'Entrate' : 'Investimenti'}
          </div>
          <div className="text-3xl font-bold text-primary-600 dark:text-primary-400">
            {formatCurrency(total)}
          </div>
          <div className="text-xs text-primary-600 dark:text-primary-400 mt-1">
            {filteredTransactions.length} transazioni
          </div>
        </div>

        {/* Grafico a barre per categoria */}
        {categoryStats.length === 0 ? (
          <div className="card text-center py-12">
            <div className="text-gray-500 dark:text-gray-400">
              Nessuna transazione in questo periodo
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Per Categoria
            </h2>

            {categoryStats.map((stat, index) => (
              <div key={stat.name} className="card">
                {/* Header categoria */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xl">{stat.icon}</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {stat.name}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className={`font-bold ${
                      filter === 'expense'
                        ? 'text-red-600 dark:text-red-400'
                        : filter === 'income'
                        ? 'text-green-600 dark:text-green-400'
                        : 'text-blue-600 dark:text-blue-400'
                    }`}>
                      {formatCurrency(stat.amount)}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {stat.count} trans.
                    </div>
                  </div>
                </div>

                {/* Barra di progresso */}
                <div className="relative w-full h-8 bg-gray-200 dark:bg-gray-700 rounded-lg overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      filter === 'expense'
                        ? 'bg-red-500'
                        : filter === 'income'
                        ? 'bg-green-500'
                        : 'bg-blue-500'
                    }`}
                    style={{ width: `${(stat.amount / maxAmount) * 100}%` }}
                  />
                  <div className="absolute inset-0 flex items-center justify-end pr-3">
                    <span className={`text-sm font-bold ${
                      stat.percentage > 30
                        ? 'text-white'
                        : filter === 'expense'
                        ? 'text-red-600'
                        : filter === 'income'
                        ? 'text-green-600'
                        : 'text-blue-600'
                    }`}>
                      {stat.percentage.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Date Range Picker */}
        <DateRangePicker
          isOpen={isDatePickerOpen}
          onClose={() => setIsDatePickerOpen(false)}
          onConfirm={handleCustomPeriodConfirm}
          initialStart={startDate}
          initialEnd={endDate}
        />
      </div>
    </Layout>
  );
}
