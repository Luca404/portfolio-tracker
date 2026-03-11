import { useState, useMemo } from 'react';
import { apiService } from '../services/api';
import { useData } from '../contexts/DataContext';
import Layout from '../components/layout/Layout';
import FAB from '../components/common/FAB';
import Modal from '../components/common/Modal';
import TransactionForm from '../components/transactions/TransactionForm';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PeriodSelector from '../components/common/PeriodSelector';
import DateRangePicker from '../components/common/DateRangePicker';
import { usePeriod } from '../hooks/usePeriod';
import type { Transaction, TransactionFormData } from '../types';

type PeriodType = 'day' | 'week' | 'month' | 'year' | 'all' | 'custom';

export default function TransactionsPage() {
  const { transactions: allTransactions, accounts, categories, isLoading: dataLoading, addTransaction, updateTransaction: updateTransactionCache, deleteTransaction: deleteTransactionCache } = useData();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDatePickerOpen, setIsDatePickerOpen] = useState(false);
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [isEditMode, setIsEditMode] = useState(false);

  // Period state - condiviso tra le pagine
  const { startDate, endDate, setPeriod } = usePeriod();

  // Filtra le transazioni in base al periodo selezionato
  const transactions = useMemo(() => {
    return allTransactions.filter(transaction => {
      const transactionDate = new Date(transaction.date);
      return transactionDate >= startDate && transactionDate <= endDate;
    });
  }, [allTransactions, startDate, endDate]);

  const getAccountName = (accountId: number) => {
    if (!accountId) return '';
    const account = accounts.find((a) => a.id === accountId);
    return account ? `${account.icon} ${account.name}` : `Conto #${accountId}`;
  };

  const getCategoryIcon = (categoryName: string) => {
    const category = categories.find((c) => c.name === categoryName);
    return category?.icon || '📌';
  };

  const handlePeriodChange = (start: Date, end: Date, type: PeriodType) => {
    setPeriod(start, end, type);
  };

  const handleCustomPeriodConfirm = (start: Date, end: Date) => {
    setPeriod(start, end, 'custom');
  };

  const handleCreateTransaction = async (data: TransactionFormData) => {
    const newTransaction = await apiService.createTransaction(data);
    addTransaction(newTransaction);
  };

  const handleUpdateTransaction = async (data: TransactionFormData) => {
    if (selectedTransaction) {
      const updated = await apiService.updateTransaction(selectedTransaction.id, data);
      updateTransactionCache(updated);
      setIsModalOpen(false);
      setSelectedTransaction(null);
      setIsEditMode(false);
    }
  };

  const handleDeleteTransaction = async () => {
    if (selectedTransaction) {
      await apiService.deleteTransaction(selectedTransaction.id);
      deleteTransactionCache(selectedTransaction.id);
      setIsModalOpen(false);
      setSelectedTransaction(null);
      setIsEditMode(false);
    }
  };

  const handleTransactionClick = (transaction: Transaction) => {
    setSelectedTransaction(transaction);
    setIsEditMode(true);
    setIsModalOpen(true);
  };

  const handleNewTransaction = () => {
    setSelectedTransaction(null);
    setIsEditMode(false);
    setIsModalOpen(true);
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
    });
  };

  if (dataLoading) {
    return <LoadingSpinner />;
  }

  return (
    <Layout>
      <div className="space-y-4">
        {/* Period Selector */}
        <PeriodSelector
          startDate={startDate}
          endDate={endDate}
          onPeriodChange={handlePeriodChange}
          onCustomClick={() => setIsDatePickerOpen(true)}
        />

        {/* Lista transazioni */}
        <div>
          {transactions.length === 0 ? (
            <div className="card text-center py-12">
              <div className="text-6xl mb-4">📋</div>
              <div className="text-gray-500 dark:text-gray-400">
                Nessuna transazione in questo periodo
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              {transactions.map((transaction) => (
                <div
                  key={transaction.id}
                  className="card flex items-center justify-between hover:shadow-md transition-shadow cursor-pointer"
                  onClick={() => handleTransactionClick(transaction)}
                >
                  <div className="flex items-center gap-3 flex-1">
                    <span className="text-2xl">{getCategoryIcon(transaction.category)}</span>
                    <div className="flex-1">
                      <div className="font-medium text-gray-900 dark:text-gray-100">
                        {transaction.category}
                        {transaction.subcategory && (
                          <span className="text-sm text-gray-500 dark:text-gray-400"> ({transaction.subcategory})</span>
                        )}
                      </div>
                      {transaction.description && (
                        <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          {transaction.description}
                        </div>
                      )}
                      {transaction.ticker && (
                        <div className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                          {transaction.ticker} • {transaction.quantity} x {formatCurrency(transaction.price || 0)}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="text-right ml-4">
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                      {getAccountName(transaction.account_id)}
                    </div>
                    <div className={`font-bold text-lg ${
                      transaction.type === 'income'
                        ? 'text-green-600 dark:text-green-400'
                        : transaction.type === 'expense'
                        ? 'text-red-600 dark:text-red-400'
                        : transaction.type === 'investment'
                        ? 'text-blue-600 dark:text-blue-400'
                        : 'text-purple-600 dark:text-purple-400'
                    }`}>
                      {transaction.type === 'income' ? '+' : '-'}{formatCurrency(Math.abs(transaction.amount))}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {formatDate(transaction.date)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* FAB */}
        <FAB onClick={handleNewTransaction} />

        {/* Modal transazione */}
        <Modal
          isOpen={isModalOpen}
          onClose={() => {
            setIsModalOpen(false);
            setSelectedTransaction(null);
            setIsEditMode(false);
          }}
          title={isEditMode ? "Modifica Transazione" : "Nuova Transazione"}
        >
          <TransactionForm
            onSubmit={isEditMode ? handleUpdateTransaction : handleCreateTransaction}
            onCancel={() => {
              setIsModalOpen(false);
              setSelectedTransaction(null);
              setIsEditMode(false);
            }}
            initialData={selectedTransaction ? {
              type: selectedTransaction.type,
              category: selectedTransaction.category,
              subcategory: selectedTransaction.subcategory,
              amount: Math.abs(selectedTransaction.amount),
              description: selectedTransaction.description || '',
              date: selectedTransaction.date,
              account_id: selectedTransaction.account_id,
              ticker: selectedTransaction.ticker,
              quantity: selectedTransaction.quantity,
              price: selectedTransaction.price,
            } : undefined}
            isEditMode={isEditMode}
            onDelete={isEditMode ? handleDeleteTransaction : undefined}
          />
        </Modal>

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
