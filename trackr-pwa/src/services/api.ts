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
  Portfolio,
  PortfolioFormData,
} from '../types';
import { localStorageService } from './localStorage';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// Modalità offline: usa IndexedDB invece di chiamate HTTP
// Cambia a false per usare il server backend
const OFFLINE_MODE = import.meta.env.VITE_OFFLINE_MODE !== 'false';

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
    if (OFFLINE_MODE) {
      return localStorageService.login(credentials);
    }

    const { data } = await this.api.post<AuthResponse>('/api/auth/login', credentials);
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('authToken', data.access_token);
    localStorage.setItem('user', JSON.stringify(data.user));
    return data;
  }

  async register(credentials: RegisterCredentials): Promise<AuthResponse> {
    if (OFFLINE_MODE) {
      return localStorageService.register(credentials);
    }

    const { data } = await this.api.post<AuthResponse>('/api/auth/register', credentials);
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('authToken', data.access_token);
    localStorage.setItem('user', JSON.stringify(data.user));
    return data;
  }

  logout() {
    if (OFFLINE_MODE) {
      return localStorageService.logout();
    }

    localStorage.removeItem('access_token');
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    window.location.href = '/login';
  }

  getCurrentUser(): User | null {
    return localStorageService.getCurrentUser();
  }

  isAuthenticated(): boolean {
    return localStorageService.isAuthenticated();
  }

  // Transaction endpoints
  async getTransactions(params?: {
    startDate?: string;
    endDate?: string;
    category?: string;
    type?: string;
  }): Promise<Transaction[]> {
    if (OFFLINE_MODE) {
      return localStorageService.getTransactions(params);
    }

    // Converti camelCase a snake_case per il backend
    const backendParams = params ? {
      start_date: params.startDate,
      end_date: params.endDate,
      category: params.category,
      type: params.type,
    } : undefined;

    const { data } = await this.api.get<Transaction[]>('/api/transactions', { params: backendParams });
    return data;
  }

  async getTransaction(id: string): Promise<Transaction> {
    if (OFFLINE_MODE) {
      return localStorageService.getTransaction(id);
    }

    const { data } = await this.api.get<Transaction>(`/api/transactions/${id}`);
    return data;
  }

  async createTransaction(transaction: TransactionFormData): Promise<Transaction> {
    if (OFFLINE_MODE) {
      return localStorageService.createTransaction(transaction);
    }

    const { data } = await this.api.post<Transaction>('/api/transactions', transaction);
    return data;
  }

  async updateTransaction(id: string, transaction: Partial<TransactionFormData>): Promise<Transaction> {
    if (OFFLINE_MODE) {
      return localStorageService.updateTransaction(id, transaction);
    }

    const { data } = await this.api.put<Transaction>(`/api/transactions/${id}`, transaction);
    return data;
  }

  async deleteTransaction(id: string): Promise<void> {
    if (OFFLINE_MODE) {
      return localStorageService.deleteTransaction(id);
    }

    await this.api.delete(`/api/transactions/${id}`);
  }

  async getTransactionStats(params?: {
    startDate?: string;
    endDate?: string;
  }): Promise<TransactionStats> {
    if (OFFLINE_MODE) {
      return localStorageService.getTransactionStats(params);
    }

    const { data } = await this.api.get<TransactionStats>('/api/transactions/stats', { params });
    return data;
  }

  // Category endpoints
  async getCategories(params?: {
    start_date?: string;
    end_date?: string;
  }): Promise<CategoryWithStats[]> {
    if (OFFLINE_MODE) {
      return localStorageService.getCategories(params);
    }

    const { data } = await this.api.get<CategoryWithStats[]>('/api/categories', { params });
    return data;
  }

  async createCategory(category: CategoryFormData): Promise<Category> {
    if (OFFLINE_MODE) {
      return localStorageService.createCategory(category);
    }

    const { data } = await this.api.post<Category>('/api/categories', category);
    return data;
  }

  async updateCategory(id: number, category: Partial<CategoryFormData>): Promise<Category> {
    if (OFFLINE_MODE) {
      return localStorageService.updateCategory(id, category);
    }

    const { data } = await this.api.put<Category>(`/api/categories/${id}`, category);
    return data;
  }

  async deleteCategory(id: number): Promise<void> {
    if (OFFLINE_MODE) {
      return localStorageService.deleteCategory(id);
    }

    await this.api.delete(`/api/categories/${id}`);
  }

  // Subcategory endpoints
  async getSubcategories(categoryId: number): Promise<Subcategory[]> {
    if (OFFLINE_MODE) {
      return localStorageService.getSubcategories(categoryId);
    }

    const { data } = await this.api.get<Subcategory[]>(`/api/categories/${categoryId}/subcategories`);
    return data;
  }

  async createSubcategory(categoryId: number, subcategory: SubcategoryFormData): Promise<Subcategory> {
    if (OFFLINE_MODE) {
      return localStorageService.createSubcategory(categoryId, subcategory);
    }

    const { data } = await this.api.post<Subcategory>(`/api/categories/${categoryId}/subcategories`, subcategory);
    return data;
  }

  async updateSubcategory(categoryId: number, subcategoryId: number, subcategory: Partial<SubcategoryFormData>): Promise<Subcategory> {
    if (OFFLINE_MODE) {
      return localStorageService.updateSubcategory(categoryId, subcategoryId, subcategory);
    }

    const { data } = await this.api.put<Subcategory>(`/api/categories/${categoryId}/subcategories/${subcategoryId}`, subcategory);
    return data;
  }

  async deleteSubcategory(categoryId: number, subcategoryId: number): Promise<void> {
    if (OFFLINE_MODE) {
      return localStorageService.deleteSubcategory(categoryId, subcategoryId);
    }

    await this.api.delete(`/api/categories/${categoryId}/subcategories/${subcategoryId}`);
  }

  // ==================== ACCOUNTS ====================

  async getAccounts(): Promise<Account[]> {
    if (OFFLINE_MODE) {
      return localStorageService.getAccounts();
    }

    const response = await this.api.get('/api/accounts/');
    return response.data;
  }

  async createAccount(data: AccountFormData): Promise<Account> {
    if (OFFLINE_MODE) {
      return localStorageService.createAccount(data);
    }

    const response = await this.api.post('/api/accounts/', data);
    return response.data;
  }

  async updateAccount(id: number, data: Partial<AccountFormData>): Promise<Account> {
    if (OFFLINE_MODE) {
      return localStorageService.updateAccount(id, data);
    }

    const response = await this.api.put(`/api/accounts/${id}`, data);
    return response.data;
  }

  async deleteAccount(id: number): Promise<void> {
    if (OFFLINE_MODE) {
      return localStorageService.deleteAccount(id);
    }

    await this.api.delete(`/api/accounts/${id}`);
  }

  // ==================== PORTFOLIOS ====================

  async getPortfolios(): Promise<Portfolio[]> {
    if (OFFLINE_MODE) {
      return localStorageService.getPortfolios();
    }

    const response = await this.api.get('/api/portfolios');
    return response.data.portfolios || [];
  }

  async createPortfolio(data: PortfolioFormData): Promise<Portfolio> {
    if (OFFLINE_MODE) {
      return localStorageService.createPortfolio(data);
    }

    const response = await this.api.post('/api/portfolios', data);
    return response.data;
  }

  async updatePortfolio(id: number, data: Partial<PortfolioFormData>): Promise<Portfolio> {
    if (OFFLINE_MODE) {
      return localStorageService.updatePortfolio(id, data);
    }

    const response = await this.api.put(`/api/portfolios/${id}`, data);
    return response.data;
  }

  async deletePortfolio(id: number): Promise<void> {
    if (OFFLINE_MODE) {
      return localStorageService.deletePortfolio(id);
    }

    await this.api.delete(`/api/portfolios/${id}`);
  }

  // ==================== EXPORT/IMPORT ====================

  async exportData(): Promise<void> {
    if (OFFLINE_MODE) {
      return localStorageService.exportData();
    }
    throw new Error('Export disponibile solo in modalità offline');
  }

  async importData(file: File): Promise<void> {
    if (OFFLINE_MODE) {
      return localStorageService.importData(file);
    }
    throw new Error('Import disponibile solo in modalità offline');
  }
}

export const apiService = new ApiService();
