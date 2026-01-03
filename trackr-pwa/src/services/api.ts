import axios from 'axios';
import type { AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import type {
  AuthResponse,
  LoginCredentials,
  RegisterCredentials,
  Transaction,
  TransactionFormData,
  TransactionStats,
  User,
  Category,
  CategoryWithStats,
  CategoryFormData,
  Subcategory,
  SubcategoryFormData,
  Account,
  AccountFormData,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

class ApiService {
  private api: AxiosInstance;

  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor per aggiungere JWT token
    this.api.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        const token = localStorage.getItem('authToken');
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor per gestire errori di autenticazione
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Token scaduto o non valido
          localStorage.removeItem('authToken');
          localStorage.removeItem('user');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Auth endpoints
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const { data } = await this.api.post<AuthResponse>('/api/auth/login', credentials);
    localStorage.setItem('authToken', data.access_token);
    localStorage.setItem('user', JSON.stringify(data.user));
    return data;
  }

  async register(credentials: RegisterCredentials): Promise<AuthResponse> {
    const { data } = await this.api.post<AuthResponse>('/api/auth/register', credentials);
    localStorage.setItem('authToken', data.access_token);
    localStorage.setItem('user', JSON.stringify(data.user));
    return data;
  }

  logout() {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    window.location.href = '/login';
  }

  getCurrentUser(): User | null {
    const userStr = localStorage.getItem('user');
    if (!userStr) return null;
    try {
      return JSON.parse(userStr);
    } catch {
      return null;
    }
  }

  isAuthenticated(): boolean {
    return !!localStorage.getItem('authToken');
  }

  // Transaction endpoints
  async getTransactions(params?: {
    startDate?: string;
    endDate?: string;
    category?: string;
    type?: string;
  }): Promise<Transaction[]> {
    const { data } = await this.api.get<Transaction[]>('/api/transactions', { params });
    return data;
  }

  async getTransaction(id: string): Promise<Transaction> {
    const { data } = await this.api.get<Transaction>(`/api/transactions/${id}`);
    return data;
  }

  async createTransaction(transaction: TransactionFormData): Promise<Transaction> {
    const { data } = await this.api.post<Transaction>('/api/transactions', transaction);
    return data;
  }

  async updateTransaction(id: string, transaction: Partial<TransactionFormData>): Promise<Transaction> {
    const { data } = await this.api.put<Transaction>(`/api/transactions/${id}`, transaction);
    return data;
  }

  async deleteTransaction(id: string): Promise<void> {
    await this.api.delete(`/api/transactions/${id}`);
  }

  async getTransactionStats(params?: {
    startDate?: string;
    endDate?: string;
  }): Promise<TransactionStats> {
    const { data } = await this.api.get<TransactionStats>('/api/transactions/stats', { params });
    return data;
  }

  // Category endpoints
  async getCategories(params?: {
    start_date?: string;
    end_date?: string;
  }): Promise<CategoryWithStats[]> {
    const { data } = await this.api.get<CategoryWithStats[]>('/api/categories', { params });
    return data;
  }

  async createCategory(category: CategoryFormData): Promise<Category> {
    const { data } = await this.api.post<Category>('/api/categories', category);
    return data;
  }

  async updateCategory(id: number, category: Partial<CategoryFormData>): Promise<Category> {
    const { data } = await this.api.put<Category>(`/api/categories/${id}`, category);
    return data;
  }

  async deleteCategory(id: number): Promise<void> {
    await this.api.delete(`/api/categories/${id}`);
  }

  // Subcategory endpoints
  async getSubcategories(categoryId: number): Promise<Subcategory[]> {
    const { data } = await this.api.get<Subcategory[]>(`/api/categories/${categoryId}/subcategories`);
    return data;
  }

  async createSubcategory(categoryId: number, subcategory: SubcategoryFormData): Promise<Subcategory> {
    const { data } = await this.api.post<Subcategory>(`/api/categories/${categoryId}/subcategories`, subcategory);
    return data;
  }

  async updateSubcategory(categoryId: number, subcategoryId: number, subcategory: Partial<SubcategoryFormData>): Promise<Subcategory> {
    const { data } = await this.api.put<Subcategory>(`/api/categories/${categoryId}/subcategories/${subcategoryId}`, subcategory);
    return data;
  }

  async deleteSubcategory(categoryId: number, subcategoryId: number): Promise<void> {
    await this.api.delete(`/api/categories/${categoryId}/subcategories/${subcategoryId}`);
  }

  // ==================== ACCOUNTS ====================

  async getAccounts(): Promise<Account[]> {
    const response = await this.api.get('/accounts/');
    return response.data;
  }

  async createAccount(data: AccountFormData): Promise<Account> {
    const response = await this.api.post('/accounts/', data);
    return response.data;
  }

  async updateAccount(id: number, data: Partial<AccountFormData>): Promise<Account> {
    const response = await this.api.put(`/accounts/${id}`, data);
    return response.data;
  }

  async deleteAccount(id: number): Promise<void> {
    await this.api.delete(`/accounts/${id}`);
  }
}

export const apiService = new ApiService();
