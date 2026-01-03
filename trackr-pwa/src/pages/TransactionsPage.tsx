import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import Layout from '../components/layout/Layout';
import FAB from '../components/common/FAB';
import Modal from '../components/common/Modal';
import TransactionForm from '../components/transactions/TransactionForm';
import LoadingSpinner from '../components/common/LoadingSpinner';
import MonthYearPicker from '../components/common/MonthYearPicker';
import type { Transaction, TransactionFormData } from '../types';

export default function TransactionsPage() {
  const now = new Date();
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [filterCategory, setFilterCategory] = useState('');
  const [filterType, setFilterType] = useState('');
  const [selectedMonth, setSelectedMonth] = useState(now.getMonth());
  const [selectedYear, setSelectedYear] = useState(now.getFullYear());

  useEffect(() => {
    loadTransactions();
  }, [filterCategory, filterType, selectedMonth, selectedYear]);

  const loadTransactions = async () => {
    setIsLoading(true);
    try {
      // Calcola start_date e end_date per il mese selezionato
      const startDate = new Date(selectedYear, selectedMonth, 1);
      const endDate = new Date(selectedYear, selectedMonth + 1, 0);

      const data = await apiService.getTransactions({
        category: filterCategory || undefined,
        type: filterType || undefined,
        start_date: startDate.toISOString().split('T')[0],
        end_date: endDate.toISOString().split('T')[0],
      });
      setTransactions(data);
    } catch (error) {
      console.error('Errore caricamento transazioni:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateTransaction = async (data: TransactionFormData) => {
    await apiService.createTransaction(data);
    await loadTransactions();
  };

  const handleDeleteTransaction = async (id: string) => {
    if (confirm('Sei sicuro di voler eliminare questa transazione?')) {
      await apiService.deleteTransaction(id);
      await loadTransactions();
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('it-IT', {
      style: 'currency',
      currency: 'EUR',
    }).format(amount);
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('it-IT', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
    });
  };

  if (isLoading) {
    return <LoadingSpinner />;
  }

  return (
    <Layout>
      <div className="space-y-4">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Transazioni
        </h1>

        {/* Selettore mese/anno */}
        <MonthYearPicker
          selectedMonth={selectedMonth}
          selectedYear={selectedYear}
          onMonthChange={setSelectedMonth}
          onYearChange={setSelectedYear}
        />

        {/* Filtri */}
        <div className="flex gap-3 overflow-x-auto pb-2">
          <button
            onClick={() => setFilterType('')}
            className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-colors ${
              filterType === ''
                ? 'bg-primary-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            Tutte
          </button>
          <button
            onClick={() => setFilterType('expense')}
            className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-colors ${
              filterType === 'expense'
                ? 'bg-red-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            Uscite
          </button>
          <button
            onClick={() => setFilterType('income')}
            className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-colors ${
              filterType === 'income'
                ? 'bg-green-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            Entrate
          </button>
          <button
            onClick={() => setFilterType('investment')}
            className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-colors ${
              filterType === 'investment'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            Investimenti
          </button>
        </div>

        {/* Lista transazioni */}
        {transactions.length === 0 ? (
          <div className="card text-center py-12">
            <div className="text-6xl mb-4">📋</div>
            <div className="text-gray-500 dark:text-gray-400">
              Nessuna transazione trovata
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {transactions.map((transaction) => (
              <div
                key={transaction.id}
                className="card flex items-center justify-between"
              >
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold text-gray-900 dark:text-gray-100">
                      {transaction.category}
                    </span>
                    {transaction.ticker && (
                      <span className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 px-2 py-0.5 rounded">
                        {transaction.ticker}
                      </span>
                    )}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {transaction.description || '-'}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                    {formatDate(transaction.date)}
                  </div>
                  {transaction.ticker && (
                    <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                      {transaction.quantity} × {formatCurrency(transaction.price || 0)}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <div
                    className={`text-lg font-bold ${
                      transaction.type === 'income'
                        ? 'text-green-600 dark:text-green-400'
                        : 'text-red-600 dark:text-red-400'
                    }`}
                  >
                    {transaction.type === 'income' ? '+' : '-'}
                    {formatCurrency(transaction.amount)}
                  </div>
                  <button
                    onClick={() => handleDeleteTransaction(transaction.id)}
                    className="text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 p-2 rounded-lg"
                  >
                    🗑️
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <FAB onClick={() => setIsModalOpen(true)} />

      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title="Nuova Transazione"
      >
        <TransactionForm
          onSubmit={handleCreateTransaction}
          onCancel={() => setIsModalOpen(false)}
        />
      </Modal>
    </Layout>
  );
}
