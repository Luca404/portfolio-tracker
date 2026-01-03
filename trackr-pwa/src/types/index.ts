export interface User {
  id: string;
  email: string;
  name: string;
  createdAt: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterCredentials extends LoginCredentials {
  username: string;
}

export type TransactionCategory = string;

export type TransactionType = 'expense' | 'income' | 'investment' | 'transfer';

export interface Transaction {
  id: string;
  userId?: string;
  type: TransactionType;
  category: string;
  subcategory?: string;
  amount: number;
  description?: string;
  date: string;
  createdAt?: string;
  updatedAt?: string;

  // Campi specifici per investimenti
  ticker?: string;
  quantity?: number;
  price?: number;
}

export interface TransactionFormData {
  type: TransactionType;
  category: string;
  subcategory?: string;
  amount: number;
  description: string;
  date: string;

  // Campi opzionali per investimenti
  ticker?: string;
  quantity?: number;
  price?: number;
}

export interface TransactionStats {
  totalExpenses: number;
  totalIncome: number;
  totalInvestments: number;
  balance: number;
  expensesByCategory: Record<string, number>;
  monthlyTrend: Array<{
    month: string;
    expenses: number;
    income: number;
  }>;
}

export interface ApiError {
  message: string;
  status?: number;
  errors?: Record<string, string[]>;
}

export interface Subcategory {
  id: number;
  category_id: number;
  name: string;
  icon: string;
  created_at: string;
  updated_at: string;
}

export interface Category {
  id: number;
  user_id: number;
  name: string;
  icon: string;
  category_type?: string | null;  // 'expense', 'income', 'investment', 'transfer', or null for all
  created_at: string;
  updated_at: string;
  subcategories: Subcategory[];
}

export interface CategoryWithStats extends Category {
  total_amount: number;
  transaction_count: number;
}

export interface CategoryFormData {
  name: string;
  icon: string;
  category_type?: string | null;
}

export interface SubcategoryFormData {
  name: string;
  icon: string;
}

export interface Account {
  id: number;
  user_id: number;
  name: string;
  icon: string;
  initial_balance: number;
  created_at: string;
  updated_at?: string;
}

export interface AccountFormData {
  name: string;
  icon: string;
  initial_balance: number;
}
