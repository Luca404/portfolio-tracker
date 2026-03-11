import { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import { apiService } from '../services/api';
import type { Account, Category, Transaction } from '../types';

interface DataContextType {
  // Data
  accounts: Account[];
  categories: Category[];
  transactions: Transaction[];

  // Loading states
  isLoading: boolean;
  isInitialized: boolean;

  // CRUD operations
  addAccount: (account: Account) => void;
  updateAccount: (account: Account) => void;
  deleteAccount: (id: number) => void;

  addCategory: (category: Category) => void;
  updateCategory: (category: Category) => void;
  deleteCategory: (id: number) => void;

  addTransaction: (transaction: Transaction) => void;
  updateTransaction: (transaction: Transaction) => void;
  deleteTransaction: (id: number) => void;

  // Refresh functions
  refreshAccounts: () => Promise<void>;
  refreshCategories: () => Promise<void>;
  refreshTransactions: (startDate?: string, endDate?: string) => Promise<void>;
  refreshAll: () => Promise<void>;

  // Clear cache
  clearCache: () => void;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

const CACHE_KEYS = {
  ACCOUNTS: 'trackr_accounts',
  CATEGORIES: 'trackr_categories',
  TRANSACTIONS: 'trackr_transactions',
  LAST_SYNC: 'trackr_last_sync',
};

interface DataProviderProps {
  children: ReactNode;
}

export function DataProvider({ children }: DataProviderProps) {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [categories, setCategories] = useState<Category[]>([]);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isInitialized, setIsInitialized] = useState(false);

  // Check if user is authenticated before fetching
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (token) {
      loadFromCache();
      fetchAllData();
    } else {
      setIsLoading(false);
      setIsInitialized(false);
    }

    // Listen for storage changes (login from another tab)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'access_token') {
        if (e.newValue) {
          // User logged in
          loadFromCache();
          fetchAllData();
        } else {
          // User logged out
          clearCache();
        }
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  // Ricalcola i current_balance degli account quando cambiano le transazioni
  useEffect(() => {
    if (!isInitialized || accounts.length === 0) return;

    setAccounts(prevAccounts => {
      return prevAccounts.map(account => {
        // Calcola il saldo corrente dalle transazioni
        const accountTransactions = transactions.filter(t => t.account_id === account.id);
        let currentBalance = account.initial_balance;

        accountTransactions.forEach(transaction => {
          if (transaction.type === 'income') {
            currentBalance += transaction.amount;
          } else if (transaction.type === 'expense' || transaction.type === 'investment') {
            currentBalance -= transaction.amount;
          }
        });

        // Solo aggiorna se il saldo è cambiato
        if (currentBalance !== account.current_balance) {
          return { ...account, current_balance: currentBalance };
        }
        return account;
      });
    });
  }, [transactions, isInitialized]);

  // Save to localStorage whenever data changes
  useEffect(() => {
    if (isInitialized) {
      saveToCache();
    }
  }, [accounts, categories, transactions, isInitialized]);

  const loadFromCache = () => {
    try {
      const cachedAccounts = localStorage.getItem(CACHE_KEYS.ACCOUNTS);
      const cachedCategories = localStorage.getItem(CACHE_KEYS.CATEGORIES);
      const cachedTransactions = localStorage.getItem(CACHE_KEYS.TRANSACTIONS);

      if (cachedAccounts) setAccounts(JSON.parse(cachedAccounts));
      if (cachedCategories) setCategories(JSON.parse(cachedCategories));
      if (cachedTransactions) setTransactions(JSON.parse(cachedTransactions));
    } catch (error) {
      console.error('Error loading from cache:', error);
    }
  };

  const saveToCache = () => {
    try {
      localStorage.setItem(CACHE_KEYS.ACCOUNTS, JSON.stringify(accounts));
      localStorage.setItem(CACHE_KEYS.CATEGORIES, JSON.stringify(categories));
      localStorage.setItem(CACHE_KEYS.TRANSACTIONS, JSON.stringify(transactions));
      localStorage.setItem(CACHE_KEYS.LAST_SYNC, new Date().toISOString());
    } catch (error) {
      console.error('Error saving to cache:', error);
    }
  };

  const fetchAllData = async () => {
    setIsLoading(true);
    try {
      await Promise.all([
        refreshAccounts(),
        refreshCategories(),
        refreshTransactions(),
      ]);
    } catch (error) {
      console.error('Error fetching all data:', error);
    } finally {
      setIsLoading(false);
      setIsInitialized(true);
    }
  };

  const refreshAccounts = async () => {
    try {
      const data = await apiService.getAccounts();
      setAccounts(data);
    } catch (error) {
      console.error('Error refreshing accounts:', error);
      throw error;
    }
  };

  const refreshCategories = async () => {
    try {
      const data = await apiService.getCategories();
      setCategories(data);
    } catch (error) {
      console.error('Error refreshing categories:', error);
      throw error;
    }
  };

  const refreshTransactions = async (startDate?: string, endDate?: string) => {
    try {
      const data = await apiService.getTransactions(
        startDate && endDate ? { startDate, endDate } : undefined
      );
      setTransactions(data);
    } catch (error) {
      console.error('Error refreshing transactions:', error);
      throw error;
    }
  };

  const refreshAll = async () => {
    await fetchAllData();
  };

  const clearCache = () => {
    setAccounts([]);
    setCategories([]);
    setTransactions([]);
    setIsInitialized(false);
    localStorage.removeItem(CACHE_KEYS.ACCOUNTS);
    localStorage.removeItem(CACHE_KEYS.CATEGORIES);
    localStorage.removeItem(CACHE_KEYS.TRANSACTIONS);
    localStorage.removeItem(CACHE_KEYS.LAST_SYNC);
  };

  // Account operations
  const addAccount = (account: Account) => {
    setAccounts(prev => [...prev, account]);
  };

  const updateAccount = (account: Account) => {
    setAccounts(prev => prev.map(a => a.id === account.id ? account : a));
  };

  const deleteAccount = (id: number) => {
    setAccounts(prev => prev.filter(a => a.id !== id));
  };

  // Category operations
  const addCategory = (category: Category) => {
    setCategories(prev => [...prev, category]);
  };

  const updateCategory = (category: Category) => {
    setCategories(prev => prev.map(c => c.id === category.id ? category : c));
  };

  const deleteCategory = (id: number) => {
    setCategories(prev => prev.filter(c => c.id !== id));
  };

  // Transaction operations
  const addTransaction = (transaction: Transaction) => {
    setTransactions(prev => {
      const newTransactions = [...prev, transaction];
      // Sort by date descending, then by created_at descending
      return newTransactions.sort((a, b) => {
        const dateCompare = new Date(b.date).getTime() - new Date(a.date).getTime();
        if (dateCompare !== 0) return dateCompare;
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
    });
  };

  const updateTransaction = (transaction: Transaction) => {
    setTransactions(prev => {
      const updated = prev.map(t => t.id === transaction.id ? transaction : t);
      // Re-sort after update
      return updated.sort((a, b) => {
        const dateCompare = new Date(b.date).getTime() - new Date(a.date).getTime();
        if (dateCompare !== 0) return dateCompare;
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
    });
  };

  const deleteTransaction = (id: number) => {
    setTransactions(prev => prev.filter(t => t.id !== id));
  };

  const value: DataContextType = {
    accounts,
    categories,
    transactions,
    isLoading,
    isInitialized,
    addAccount,
    updateAccount,
    deleteAccount,
    addCategory,
    updateCategory,
    deleteCategory,
    addTransaction,
    updateTransaction,
    deleteTransaction,
    refreshAccounts,
    refreshCategories,
    refreshTransactions,
    refreshAll,
    clearCache,
  };

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
}

export function useData() {
  const context = useContext(DataContext);
  if (context === undefined) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
}
